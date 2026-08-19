"""Лидар: измеренное расстояние поверх кадра, вид сверху, дальность до объектов."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from .data import find_segment

TOP_LASER = 1


_cache: dict = {}


def _table(segment, component, laser=TOP_LASER):
    """Компонент читается целиком один раз и держится в памяти.

    Раньше файл перечитывался на каждый кадр, и вид сверху собирался минутами.
    """
    key = (str(segment), component, laser)
    if key not in _cache:
        f = next((segment / component).glob("*.parquet"))
        df = pq.read_table(f, filters=[("key.laser_name", "=", laser)]).to_pandas()
        _cache[key] = df.set_index("key.frame_timestamp_micros")
    return _cache[key]


def _read(segment, component, ts, laser=TOP_LASER):
    """Одна строка компонента по метке времени."""
    return _table(segment, component, laser).loc[[ts]]


def _grids(clip, frame: int = 0, segment=None):
    """Дальность, проекции на камеры и координаты в метрах — на одной сетке.

    Раньше проекции и облако точек считались по разным выборкам, из-за чего
    сопоставить пиксель с точкой в пространстве было нельзя.
    """
    key = ("grids", id(clip), frame)
    if key in _cache:
        return _cache[key]

    segment = segment or find_segment()
    ts = clip.timestamps[frame]
    lid = _read(segment, "lidar", ts)
    prj = _read(segment, "lidar_camera_projection", ts)

    ck = ("calib", str(segment))
    if ck not in _cache:
        c = pq.read_table(next((segment / "lidar_calibration").glob("*.parquet"))).to_pandas()
        _cache[ck] = c[c["key.laser_name"] == TOP_LASER].iloc[0]
    cal = _cache[ck]

    shape = np.array(lid["[LiDARComponent].range_image_return1.shape"].iloc[0])
    rng = np.array(lid["[LiDARComponent].range_image_return1.values"].iloc[0]).reshape(shape)
    proj = np.array(prj["[LiDARCameraProjectionComponent].range_image_return1.values"].iloc[0]).reshape(
        np.array(prj["[LiDARCameraProjectionComponent].range_image_return1.shape"].iloc[0]))
    dist = rng[..., 0]

    incl = np.array(cal["[LiDARCalibrationComponent].beam_inclination.values"])
    h, w = dist.shape
    if len(incl) != h:
        incl = np.linspace(cal["[LiDARCalibrationComponent].beam_inclination.min"],
                           cal["[LiDARCalibrationComponent].beam_inclination.max"], h)
    az = np.linspace(np.pi, -np.pi, w)
    A, I = np.meshgrid(az, incl)

    x = dist * np.cos(I) * np.cos(A)
    # в данных ось направлена влево. Разворачиваем, чтобы на карте право было справа
    y = -dist * np.cos(I) * np.sin(A)
    z = dist * np.sin(I)

    _cache[key] = (dist, proj, np.stack([x, y, z], axis=-1))
    return _cache[key]


def points3d(clip, frame: int = 0, segment=None):
    """Точки, попавшие в кадр: пиксель u, пиксель v, дальность и координаты в метрах.

    Координаты берутся из самого измерения, а не восстанавливаются по углу обзора.
    Поэтому работают для любой камеры, а не только для передней.
    """
    key = ("p3d", id(clip), frame)
    if key in _cache:
        return _cache[key]

    dist, proj, xyz_grid = _grids(clip, frame, segment)
    h, w = clip.frames[frame].shape[:2]
    sel = (proj[..., 0] == clip.camera) & (dist > 0)

    u = proj[..., 1][sel].astype(int)
    v = proj[..., 2][sel].astype(int)
    d = dist[sel]
    P = xyz_grid[sel]

    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    _cache[key] = (u[ok], v[ok], d[ok], P[ok])
    return _cache[key]


def points(clip, frame: int = 0, segment=None):
    """Точки лидара, попавшие в кадр: пиксель u, пиксель v, расстояние в метрах."""
    u, v, d, _ = points3d(clip, frame, segment)
    return u, v, d


def overlay(clip, frame: int = 0, max_m: float = 60, size: float = 15, segment=None):
    """Кадр с наложенной развёрткой лидара, цвет — расстояние."""
    u, v, d = points(clip, frame, segment)
    fig, ax = plt.subplots(figsize=(size, size * 0.66))
    ax.imshow(clip.frames[frame])
    s = ax.scatter(u, v, c=d, s=3, cmap="turbo", vmin=2, vmax=max_m)
    fig.colorbar(s, ax=ax, fraction=0.03, pad=0.01).set_label("расстояние, м", fontsize=12)
    ax.set_title(f"Кадр {frame}: {len(d)} измерений, от {d.min():.1f} до {d.max():.1f} м",
                 fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    return fig


def distance_to_objects(result, clip, frame: int = 0, segment=None):
    """Сколько метров до каждого объекта, который модель нашла по слову.

    Модель отвечает «что и где», лидар — «как далеко». Связка идёт по маске.
    """
    import pandas as pd

    u, v, d = points(clip, frame, segment)
    masks = result.masks.get(frame)
    det = result.detections[result.detections["кадр"] == frame].reset_index(drop=True)
    if masks is None or not len(det):
        return pd.DataFrame(columns=["трек", "метров", "точек"])

    rows = []
    for i, tid in enumerate(det["трек"]):
        if i >= len(masks):
            break
        inside = masks[i][v, u]
        if inside.sum() >= 3:
            rows.append({"трек": int(tid), "метров": round(float(np.median(d[inside])), 1),
                         "точек": int(inside.sum())})
        else:
            rows.append({"трек": int(tid), "метров": np.nan, "точек": int(inside.sum())})
    return pd.DataFrame(rows).sort_values("метров")


def speed_of_objects(result, clip, frames=(0, 10), segment=None):
    """Скорость по изменению расстояния между двумя кадрами, м/с."""
    import pandas as pd

    a, b = frames
    dt = (clip.timestamps[b] - clip.timestamps[a]) / 1e6
    da = distance_to_objects(result, clip, a, segment).set_index("трек")["метров"]
    db = distance_to_objects(result, clip, b, segment).set_index("трек")["метров"]
    common = da.index.intersection(db.index)

    out = pd.DataFrame({
        "трек": common,
        "было, м": da[common].values,
        "стало, м": db[common].values,
        "м/с": ((db[common].values - da[common].values) / dt).round(1),
    }).dropna()
    out["направление"] = np.where(out["м/с"] < 0, "приближается", "удаляется")
    return out.reset_index(drop=True)


def xyz(clip, frame: int = 0, segment=None):
    """Облако точек в метрах: x вперёд, y влево, z вверх."""
    dist, _, xyz_grid = _grids(clip, frame, segment)
    return xyz_grid[dist > 0]


def trajectories(result, clip, segment=None):
    """Где каждый объект находился на плоскости в каждом кадре.

    Положение берётся по точкам дальномера, попавшим внутрь контура объекта.
    """
    import pandas as pd

    rows = []
    for frame in sorted(result.masks.keys()):
        masks = result.masks[frame]
        det = result.detections[result.detections["кадр"] == frame].reset_index(drop=True)
        if not len(det):
            continue
        u, v, d, P = points3d(clip, frame, segment)
        for i, tid in enumerate(det["трек"]):
            if i >= len(masks):
                break
            inside = masks[i][v, u]
            if inside.sum() < 3:
                continue
            # медиана устойчива к точкам фона, попавшим в контур по краю
            xy = np.median(P[inside][:, :2], axis=0)
            rows.append({"кадр": frame, "трек": int(tid),
                         "вперёд, м": round(float(xy[0]), 1),
                         "вбок, м": round(float(xy[1]), 1),
                         "дальность, м": round(float(np.median(d[inside])), 1)})
    # пустая таблица тоже должна иметь колонки, иначе обращение к ним падает
    return pd.DataFrame(rows, columns=["кадр", "трек", "вперёд, м", "вбок, м", "дальность, м"])


def show_trajectories(result, clip, span: float = 30, segment=None):
    """Путь каждого объекта на плоскости за весь отрезок."""
    from .viz import color_of

    tr = trajectories(result, clip, segment)
    fig, ax = plt.subplots(figsize=(9, 9.5))
    if len(tr):
        base = xyz(clip, 0, segment)
        ax.scatter(base[:, 1], base[:, 0], s=0.3, c="#DDE1E6")
        for tid, g in tr.groupby("трек"):
            c = color_of(tid) / 255
            ax.plot(g["вбок, м"], g["вперёд, м"], "-o", ms=4, lw=2, color=c)
            last = g.iloc[-1]
            ax.text(last["вбок, м"], last["вперёд, м"] + 1.1, f"№{tid}",
                    fontsize=10, ha="center", color=c, clip_on=True)
    ax.plot(0, 0, marker="^", ms=16, color="#E63946")
    ax.set_xlim(-span, span)
    ax.set_ylim(-4, span * 1.8)
    ax.set_xlabel("влево / вправо, м")
    ax.set_ylabel("вперёд, м")
    ax.set_title(f"Пути объектов по запросу «{result.prompt}»", fontsize=13)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, tr


def zone_report(result, clip, forward=(5.0, 25.0), side=(-4.0, 4.0), segment=None):
    """Сколько объектов попало в заданную зону и когда.

    Зона задаётся в метрах: сколько вперёд и сколько вбок от нашей машины.
    Модель находит объекты, а решение принимает обычная проверка координат.
    """
    import pandas as pd

    tr = trajectories(result, clip, segment)
    if not len(tr):
        return pd.DataFrame(), pd.DataFrame()

    tr = tr.copy()
    tr["в зоне"] = (tr["вперёд, м"].between(*forward) & tr["вбок, м"].between(*side))

    по_кадрам = (tr.groupby("кадр")["в зоне"].sum()
                 .rename("объектов в зоне").reset_index())
    события = (tr[tr["в зоне"]].groupby("трек")
               .agg(**{"первый кадр": ("кадр", "min"), "последний кадр": ("кадр", "max"),
                       "ближе всего, м": ("дальность, м", "min")})
               .reset_index())
    return по_кадрам, события


def show_zone(result, clip, forward=(5.0, 25.0), side=(-4.0, 4.0), span: float = 30,
              segment=None):
    """Зона на плоскости и пути объектов относительно неё."""
    from matplotlib.patches import Rectangle

    fig, tr = show_trajectories(result, clip, span=span, segment=segment)
    ax = fig.axes[0]
    ax.add_patch(Rectangle((side[0], forward[0]), side[1] - side[0],
                           forward[1] - forward[0], facecolor="#E63946", alpha=0.12,
                           edgecolor="#E63946", lw=2, ls="--"))
    ax.text(side[0], forward[1] + 0.6, "зона контроля", color="#E63946", fontsize=11)
    по_кадрам, события = zone_report(result, clip, forward, side, segment)
    n = len(события)
    ax.set_title(f"Запрос «{result.prompt}»: в зону попало объектов — {n}", fontsize=13)
    return fig, события


def bev(clip, frame: int = 0, span: float = 40, segment=None):
    """Вид сверху: та же сцена с высоты птичьего полёта."""
    x, y, z = xyz(clip, frame, segment).T

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(y, x, c=z, s=0.4, cmap="viridis", vmin=-3, vmax=4)
    ax.plot(0, 0, marker="^", ms=16, color="#E63946")
    ax.set_xlim(-span, span)
    ax.set_ylim(-span / 2, span * 1.5)
    ax.set_xlabel("влево / вправо, м")
    ax.set_ylabel("вперёд, м")
    ax.set_title(f"Вид сверху, кадр {frame}. Красный треугольник — наша машина", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig


# --------------------------------------------- разметка объёма, сделанная людьми

LIDAR_CLASSES = {0: "unknown", 1: "vehicle", 2: "pedestrian", 3: "sign", 4: "cyclist"}


def annotated_boxes(clip, frame: int = 0, segment=None):
    """Объёмные рамки, размеченные людьми: положение, размер, разворот, номер, скорость.

    Разметка круговая и не зависит от того, какая камера выбрана.
    """
    import pandas as pd

    segment = segment or find_segment()
    key = ("boxes", str(segment))
    if key not in _cache:
        df = pq.read_table(next((segment / "lidar_box").glob("*.parquet"))).to_pandas()
        _cache[key] = df.set_index("key.frame_timestamp_micros")
    table = _cache[key]

    ts = clip.timestamps[frame]
    if ts not in table.index:
        return pd.DataFrame()
    rows = table.loc[[ts]]

    B = "[LiDARBoxComponent]"
    return pd.DataFrame({
        "объект": rows["key.laser_object_id"].values,
        "класс": rows[f"{B}.type"].map(LIDAR_CLASSES).fillna("unknown").values,
        "вперёд, м": rows[f"{B}.box.center.x"].round(1).values,
        "вбок, м": (-rows[f"{B}.box.center.y"]).round(1).values,
        "длина, м": rows[f"{B}.box.size.x"].round(1).values,
        "ширина, м": rows[f"{B}.box.size.y"].round(1).values,
        "разворот": (-rows[f"{B}.box.heading"]).values,
        "точек": rows[f"{B}.num_lidar_points_in_box"].values,
        "м/с": np.hypot(rows[f"{B}.speed.x"], rows[f"{B}.speed.y"]).round(1).values,
    }).reset_index(drop=True)


def association(clip, frame: int = 0, segment=None):
    """Какой объект на камере соответствует какому объекту в облаке точек."""
    import pandas as pd

    segment = segment or find_segment()
    key = ("assoc", str(segment))
    if key not in _cache:
        f = next((segment / "camera_to_lidar_box_association").glob("*.parquet"))
        _cache[key] = pq.read_table(f).to_pandas()
    df = _cache[key]

    sel = df[(df["key.frame_timestamp_micros"] == clip.timestamps[frame])
             & (df["key.camera_name"] == clip.camera)]
    return pd.DataFrame({"объект на камере": sel["key.camera_object_id"].values,
                         "объект в облаке": sel["key.laser_object_id"].values})
