"""Глубина по одному снимку и сверка её с измерением лидара."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .lidar import points

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "depth-anything-v2-small"
_cache: dict = {}


def load_model(model_dir=None):
    if "model" not in _cache:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        path = str(model_dir or MODEL_DIR)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _cache["proc"] = AutoImageProcessor.from_pretrained(path)
        _cache["model"] = AutoModelForDepthEstimation.from_pretrained(path).to(device).eval()
        _cache["device"] = device
    return _cache["model"], _cache["proc"]


def predict(clip, frame: int = 0) -> np.ndarray:
    """Карта относительной глубины по одному обычному снимку."""
    model, proc = load_model()
    img = clip.frames[frame]
    inputs = proc(images=img, return_tensors="pt").to(_cache["device"])
    with torch.inference_mode():
        out = model(**inputs).predicted_depth
    return torch.nn.functional.interpolate(
        out[None], size=img.shape[:2], mode="bicubic", align_corners=False
    )[0, 0].float().cpu().numpy()


def calibrate(clip, frame: int = 0, segment=None):
    """Модель выдаёт относительную величину. Лидар даёт метры — по нему и градуируем.

    Подбираем два числа так, чтобы предсказание легло на измерение.
    Это ровно §14 доклада: камера становится прибором только после привязки к эталону.
    """
    rel = predict(clip, frame)
    u, v, d = points(clip, frame, segment)
    r = rel[v, u]

    # обратная глубина линейно связана с расстоянием
    a, b = np.polyfit(r, 1.0 / d, 1)
    metres = 1.0 / np.clip(a * rel + b, 1e-3, None)

    pred = 1.0 / np.clip(a * r + b, 1e-3, None)
    err = pred - d
    stats = {
        "точек сверки": len(d),
        "средняя ошибка, м": round(float(np.mean(np.abs(err))), 2),
        "медианная ошибка, м": round(float(np.median(np.abs(err))), 2),
        "ошибка в пределах 10%": f"{100 * np.mean(np.abs(err) / d < 0.1):.0f}%",
        "диапазон лидара, м": f"{d.min():.1f}–{d.max():.1f}",
    }
    return metres, stats, (d, pred)


def compare(clip, frame: int = 0, max_m: float = 60, segment=None):
    """Три картинки: снимок, предсказание модели, измерение лидара."""
    metres, stats, (d, pred) = calibrate(clip, frame, segment)
    u, v, _ = points(clip, frame, segment)

    fig, axes = plt.subplots(1, 3, figsize=(21, 5.2))
    axes[0].imshow(clip.frames[frame]); axes[0].set_title("Обычный снимок", fontsize=13)
    im = axes[1].imshow(np.clip(metres, 0, max_m), cmap="turbo_r")
    axes[1].set_title("Предсказано по одному кадру", fontsize=13)
    fig.colorbar(im, ax=axes[1], fraction=0.03).set_label("метры")
    axes[2].imshow(clip.frames[frame])
    s = axes[2].scatter(u, v, c=d, s=2, cmap="turbo_r", vmin=0, vmax=max_m)
    axes[2].set_title("Измерено лидаром", fontsize=13)
    fig.colorbar(s, ax=axes[2], fraction=0.03).set_label("метры")
    for a in axes:
        a.axis("off")
    fig.suptitle("Средняя ошибка {} м, в пределах 10% попадает {}".format(
        stats["средняя ошибка, м"], stats["ошибка в пределах 10%"]), fontsize=14)
    fig.tight_layout()
    return fig, stats


def scatter(clip, frame: int = 0, segment=None):
    """Предсказание против измерения: чем ближе к диагонали, тем лучше."""
    _, stats, (d, pred) = calibrate(clip, frame, segment)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(d, pred, s=1.5, alpha=0.15, color="#2A9D8F")
    lim = [0, float(np.percentile(d, 99))]
    ax.plot(lim, lim, color="#E63946", lw=2, label="идеальное совпадение")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("измерено лидаром, м"); ax.set_ylabel("предсказано моделью, м")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, stats


def distance_table(result, clip, frame: int = 0, segment=None) -> pd.DataFrame:
    """До каждого найденного словом объекта: предсказание модели против лидара."""
    from .lidar import distance_to_objects

    metres, _, _ = calibrate(clip, frame, segment)
    truth = distance_to_objects(result, clip, frame, segment)
    masks = result.masks.get(frame)
    det = result.detections[result.detections["кадр"] == frame].reset_index(drop=True)

    pred = []
    for i in range(len(det)):
        m = masks[i].astype(bool) if masks is not None and i < len(masks) else None
        pred.append(round(float(np.median(metres[m])), 1) if m is not None and m.any() else np.nan)

    out = truth.copy()
    out["по камере, м"] = pred[:len(out)]
    out = out.rename(columns={"метров": "по лидару, м"})
    out["расхождение, м"] = (out["по камере, м"] - out["по лидару, м"]).round(1)
    return out[["трек", "по лидару, м", "по камере, м", "расхождение, м"]]
