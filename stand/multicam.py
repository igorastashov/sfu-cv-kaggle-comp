"""Пять камер сразу: круговой обзор и переход объекта между камерами."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import CAMERAS, load_clip
from .viz import draw

# порядок склейки слева направо
PANORAMA = [4, 2, 1, 3, 5]


def load_all(n_frames: int = 8, start: int = 0, segment=None) -> dict:
    """Один и тот же отрезок времени со всех пяти камер."""
    return {c: load_clip(n_frames=n_frames, camera=c, start=start, segment=segment)
            for c in PANORAMA}


def panorama(clips: dict, frame: int = 0, height: int = 420) -> np.ndarray:
    """Склейка пяти кадров в одну ленту. Стыки не сглаживаются: камеры смотрят под углом."""
    parts = []
    for cam in PANORAMA:
        img = clips[cam].frames[frame]
        w = int(img.shape[1] * height / img.shape[0])
        idx_y = (np.arange(height) * img.shape[0] / height).astype(int)
        idx_x = (np.arange(w) * img.shape[1] / w).astype(int)
        parts.append(img[np.ix_(idx_y, idx_x)])
    return np.concatenate(parts, axis=1)


def show_panorama(clips: dict, frame: int = 0, results: dict | None = None,
                  height: int = 420, size: float = 22):
    """Круговой обзор. Если переданы результаты поиска, объекты обводятся."""
    parts = []
    for cam in PANORAMA:
        clip = clips[cam]
        img = clip.frames[frame]
        if results and cam in results:
            r = results[cam]
            det = r.detections[r.detections["кадр"] == frame]
            img = draw(img, det, r.masks.get(frame), thickness=6)
        w = int(img.shape[1] * height / img.shape[0])
        idx_y = (np.arange(height) * img.shape[0] / height).astype(int)
        idx_x = (np.arange(w) * img.shape[1] / w).astype(int)
        parts.append(img[np.ix_(idx_y, idx_x)])

    strip = np.concatenate(parts, axis=1)
    fig, ax = plt.subplots(figsize=(size, size * strip.shape[0] / strip.shape[1]))
    ax.imshow(strip)

    x = 0
    for cam, p in zip(PANORAMA, parts):
        ax.axvline(x, color="white", lw=2)
        ax.text(x + p.shape[1] / 2, 22, CAMERAS[cam], color="white", fontsize=12,
                ha="center", bbox=dict(facecolor="black", alpha=0.5, pad=2))
        x += p.shape[1]
    ax.axis("off")
    ax.set_title(f"Круговой обзор, кадр {frame}", fontsize=14)
    fig.tight_layout()
    return fig


def search_all(clips: dict, prompt: str, conf: float = 0.5) -> dict:
    """Один запрос по всем пяти камерам."""
    from .run import find
    return {cam: find(clips[cam], prompt, conf=conf) for cam in PANORAMA}


def summary(results: dict, clips: dict, prompt: str = "") -> pd.DataFrame:
    """Сколько объектов нашлось на каждой камере и сколько их размечено эталоном."""
    rows = []
    for cam in PANORAMA:
        r, clip = results[cam], clips[cam]
        rows.append({
            "камера": CAMERAS[cam],
            "найдено": r.n_objects,
            "размечено": clip.boxes["трек"].nunique(),
            "секунд": round(r.seconds, 1),
        })
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"камера": "всего", "найдено": df["найдено"].sum(),
                       "размечено": df["размечено"].sum(), "секунд": df["секунд"].sum()}
    return df
