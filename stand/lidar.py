"""Лидар: измеренное расстояние поверх кадра, вид сверху, дальность до объектов."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from .data import find_segment

TOP_LASER = 1


def _read(segment, component, ts, laser=TOP_LASER):
    f = next((segment / component).glob("*.parquet"))
    return pq.read_table(f, filters=[("key.laser_name", "=", laser),
                                     ("key.frame_timestamp_micros", "=", ts)]).to_pandas()


def points(clip, frame: int = 0, segment=None):
    """Точки лидара, попавшие в кадр: пиксель u, пиксель v, расстояние в метрах."""
    segment = segment or find_segment()
    ts = clip.timestamps[frame]
    h, w = clip.frames[frame].shape[:2]

    lid = _read(segment, "lidar", ts)
    prj = _read(segment, "lidar_camera_projection", ts)

    rng = np.array(lid["[LiDARComponent].range_image_return1.values"].iloc[0]).reshape(
        np.array(lid["[LiDARComponent].range_image_return1.shape"].iloc[0]))
    proj = np.array(prj["[LiDARCameraProjectionComponent].range_image_return1.values"].iloc[0]).reshape(
        np.array(prj["[LiDARCameraProjectionComponent].range_image_return1.shape"].iloc[0]))

    dist, cam_id = rng[..., 0], proj[..., 0]
    sel = (cam_id == clip.camera) & (dist > 0)
    u = proj[..., 1][sel].astype(int)
    v = proj[..., 2][sel].astype(int)
    d = dist[sel]

    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    return u[ok], v[ok], d[ok]


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
    segment = segment or find_segment()
    ts = clip.timestamps[frame]
    lid = _read(segment, "lidar", ts)
    cal = pq.read_table(next((segment / "lidar_calibration").glob("*.parquet"))).to_pandas()
    cal = cal[cal["key.laser_name"] == TOP_LASER].iloc[0]

    shape = np.array(lid["[LiDARComponent].range_image_return1.shape"].iloc[0])
    rng = np.array(lid["[LiDARComponent].range_image_return1.values"].iloc[0]).reshape(shape)
    dist = rng[..., 0]

    incl = np.array(cal["[LiDARCalibrationComponent].beam_inclination.values"])
    h, w = dist.shape
    if len(incl) != h:
        incl = np.linspace(cal["[LiDARCalibrationComponent].beam_inclination.min"],
                           cal["[LiDARCalibrationComponent].beam_inclination.max"], h)
    az = np.linspace(np.pi, -np.pi, w)

    sel = dist > 0
    A, I = np.meshgrid(az, incl)
    return np.stack([
        dist[sel] * np.cos(I[sel]) * np.cos(A[sel]),
        dist[sel] * np.cos(I[sel]) * np.sin(A[sel]),
        dist[sel] * np.sin(I[sel]),
    ], axis=1)


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
        u, v, d = points(clip, frame, segment)
        pts = xyz(clip, frame, segment)
        # порядок точек в points и xyz совпадает не всегда, поэтому считаем по дальности
        for i, tid in enumerate(det["трек"]):
            if i >= len(masks):
                break
            inside = masks[i][v, u]
            if inside.sum() < 3:
                continue
            dist = float(np.median(d[inside]))
            cx = float(np.median(u[inside]))
            # угол в плане по положению пикселя относительно центра кадра
            w = clip.frames[frame].shape[1]
            fov = np.deg2rad(50.4)  # передняя камера Waymo
            ang = -(cx - w / 2) / w * fov
            rows.append({"кадр": frame, "трек": int(tid),
                         "вперёд, м": round(dist * np.cos(ang), 1),
                         "вбок, м": round(dist * np.sin(ang), 1),
                         "дальность, м": round(dist, 1)})
    return pd.DataFrame(rows)


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
