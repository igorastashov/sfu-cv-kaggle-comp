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
