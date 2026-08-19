"""Разбор сцены: паноптическая разметка и поиск объектов геометрией, без обучения."""
from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from .data import find_segment
from .lidar import xyz

# классы паноптической разметки Waymo
SEMANTIC = {
    0: "не размечено", 1: "наша машина", 2: "легковая", 3: "грузовик", 4: "автобус",
    5: "крупный транспорт", 6: "велосипед", 7: "мотоцикл", 8: "прицеп",
    9: "пешеход", 10: "велосипедист", 11: "мотоциклист", 12: "птица",
    13: "животное", 14: "конус", 15: "столб", 16: "вещь пешехода", 17: "знак",
    18: "светофор", 19: "здание", 20: "дорога", 21: "разметка полосы",
    22: "дорожный знак на асфальте", 23: "тротуар", 24: "растительность",
    25: "небо", 26: "земля", 27: "подвижное", 28: "неподвижное",
}

_CMAP = plt.get_cmap("tab20")


def panoptic(clip, frame: int = 0, segment=None):
    """Эталонная разметка сцены: что за класс у каждого пикселя и какой это экземпляр."""
    segment = segment or find_segment()
    ts = clip.timestamps[frame]
    f = next((segment / "camera_segmentation").glob("*.parquet"))
    df = pq.read_table(f, filters=[("key.camera_name", "=", clip.camera),
                                   ("key.frame_timestamp_micros", "=", ts)]).to_pandas()
    if not len(df):
        return None

    row = df.iloc[0]
    divisor = int(row["[CameraSegmentationLabelComponent].panoptic_label_divisor"])
    label = np.array(Image.open(io.BytesIO(
        row["[CameraSegmentationLabelComponent].panoptic_label"])))
    return {"семантика": label // divisor, "экземпляры": label % divisor}


def labeled_frames(clip, segment=None) -> list[int]:
    """Кадры отрезка, у которых есть паноптическая разметка (она положена на 5 Гц)."""
    segment = segment or find_segment()
    f = next((segment / "camera_segmentation").glob("*.parquet"))
    have = set(pq.read_table(f, columns=["key.frame_timestamp_micros", "key.camera_name"]
                             ).to_pandas().query("`key.camera_name` == @clip.camera")
               ["key.frame_timestamp_micros"])
    return [i for i, t in enumerate(clip.timestamps) if t in have]


def show_panoptic(clip, frame: int = 0, alpha: float = 0.55, segment=None):
    """Вся сцена, раскрашенная по классам."""
    p = panoptic(clip, frame, segment)
    if p is None:
        raise ValueError(f"У кадра {frame} нет разметки. Доступные: {labeled_frames(clip)}")

    sem = p["семантика"]
    if sem.ndim == 3:
        sem = sem[..., 0]
    present = [c for c in np.unique(sem) if c in SEMANTIC]

    rgb = np.zeros((*sem.shape, 3), dtype=np.uint8)
    for c in present:
        rgb[sem == c] = (np.array(_CMAP(c % 20)[:3]) * 255).astype(np.uint8)

    base = clip.frames[frame]
    mix = (base * (1 - alpha) + rgb * alpha).astype(np.uint8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 6.5))
    ax1.imshow(base); ax1.set_title("Кадр", fontsize=13); ax1.axis("off")
    ax2.imshow(mix); ax2.set_title(f"Разметка человеком: {len(present)} классов", fontsize=13)
    ax2.axis("off")

    from matplotlib.patches import Patch
    order = sorted(present, key=lambda c: -(sem == c).sum())[:12]
    ax2.legend(handles=[Patch(color=_CMAP(c % 20), label=SEMANTIC.get(c, c)) for c in order],
               loc="upper left", fontsize=8, ncol=2, framealpha=0.85)
    fig.tight_layout()
    return fig


def class_areas(clip, frame: int = 0, segment=None) -> pd.DataFrame:
    """Сколько процента кадра занимает каждый класс."""
    p = panoptic(clip, frame, segment)
    sem = p["семантика"]
    if sem.ndim == 3:
        sem = sem[..., 0]
    vals, counts = np.unique(sem, return_counts=True)
    return (pd.DataFrame({"класс": [SEMANTIC.get(int(v), str(v)) for v in vals],
                          "доля кадра, %": (100 * counts / sem.size).round(1)})
            .sort_values("доля кадра, %", ascending=False).reset_index(drop=True))


# ---------------------------------------------------------------- геометрия

def find_by_geometry(clip, frame: int = 0, eps: float = 0.7, min_points: int = 12,
                     max_range: float = 45, segment=None):
    """Объекты без единой обученной модели: убрать дорогу, разбить остальное на скопления.

    Дорога снимается порогом по высоте, объекты выделяются связностью в пространстве.
    Тот самый классический подход из §2 доклада.
    """
    from sklearn.cluster import DBSCAN

    pts = xyz(clip, frame, segment)
    near = (np.linalg.norm(pts[:, :2], axis=1) < max_range)
    ground = pts[:, 2] < np.percentile(pts[near, 2], 12) + 0.35
    obj = pts[near & ~ground]

    if not len(obj):
        return pd.DataFrame(), obj, np.array([])

    labels = DBSCAN(eps=eps, min_samples=min_points).fit_predict(obj)

    rows = []
    for c in sorted(set(labels) - {-1}):
        p = obj[labels == c]
        size = p.max(0) - p.min(0)
        # объект, а не стена: компактный в плане и не выше пятиэтажки
        if not (0.4 < size[2] < 5 and size[0] < 6 and size[1] < 6):
            continue
        rows.append({
            "скопление": int(c), "точек": len(p),
            "вперёд, м": round(float(p[:, 0].mean()), 1),
            "вбок, м": round(float(p[:, 1].mean()), 1),
            "дальность, м": round(float(np.linalg.norm(p[:, :2].mean(0))), 1),
            "ширина, м": round(float(size[1]), 1),
            "высота, м": round(float(size[2]), 1),
        })
    return (pd.DataFrame(rows).sort_values("дальность, м").reset_index(drop=True),
            obj, labels)


def show_geometry(clip, frame: int = 0, span: float = 35, **kw):
    """Вид сверху с объектами, найденными одной геометрией."""
    table, pts, labels = find_by_geometry(clip, frame, **kw)

    fig, ax = plt.subplots(figsize=(9.5, 9.5))
    noise = labels == -1
    ax.scatter(pts[noise, 1], pts[noise, 0], s=0.5, c="#C9CCD1")
    keep = set(table["скопление"]) if len(table) else set()
    for c in sorted(set(labels) - {-1}):
        if c not in keep:
            continue
        p = pts[labels == c]
        ax.scatter(p[:, 1], p[:, 0], s=3, color=_CMAP(c % 20))
        ax.text(p[:, 1].mean(), p[:, 0].mean() + 0.9,
                f"{np.linalg.norm(p[:, :2].mean(0)):.0f} м", fontsize=8, ha="center",
                clip_on=True)

    ax.plot(0, 0, marker="^", ms=16, color="#E63946")
    ax.set_xlim(-span, span); ax.set_ylim(-span / 3, span * 1.6)
    ax.set_xlabel("влево / вправо, м"); ax.set_ylabel("вперёд, м")
    ax.set_title(f"Найдено геометрией, без обучения: {len(table)} объектов", fontsize=13)
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, table
