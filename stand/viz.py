"""Картинки. Всё, что участник видит глазами."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

# устойчивая палитра: номер трека -> цвет
_PALETTE = np.array([
    [0xE6, 0x39, 0x46], [0x2A, 0x9D, 0x8F], [0xE9, 0xC4, 0x6A], [0x45, 0x7B, 0x9D],
    [0xF4, 0xA2, 0x61], [0x8E, 0x7D, 0xBE], [0x43, 0xAA, 0x8B], [0xBC, 0x4B, 0x51],
    [0x27, 0x6F, 0xBF], [0xD6, 0x80, 0x2C], [0x5B, 0x8E, 0x7B], [0xC1, 0x66, 0xA5],
], dtype=np.uint8)


def color_of(track_id: int) -> np.ndarray:
    return _PALETTE[int(track_id) % len(_PALETTE)]


def draw(frame: np.ndarray, detections=None, masks=None, alpha: float = 0.45,
         thickness: int = 4) -> np.ndarray:
    """Наложить на кадр маски и рамки, каждый трек своим цветом."""
    out = frame.copy()

    if masks is not None and len(masks):
        ids = detections["трек"].tolist() if detections is not None else range(len(masks))
        for m, tid in zip(masks, ids):
            c = color_of(tid)
            sel = m.astype(bool)
            out[sel] = (out[sel] * (1 - alpha) + c * alpha).astype(np.uint8)

    if detections is not None:
        h, w = out.shape[:2]
        for _, d in detections.iterrows():
            c = color_of(d["трек"])
            x1, y1 = max(int(d["x1"]), 0), max(int(d["y1"]), 0)
            x2, y2 = min(int(d["x2"]), w - 1), min(int(d["y2"]), h - 1)
            if x2 <= x1 or y2 <= y1:
                continue
            for t in range(thickness):
                if y1 + t < h:
                    out[y1 + t, x1:x2] = c
                if y2 - t >= 0:
                    out[y2 - t, x1:x2] = c
                if x1 + t < w:
                    out[y1:y2, x1 + t] = c
                if x2 - t >= 0:
                    out[y1:y2, x2 - t] = c
    return out


def show(frame: np.ndarray, title: str = "", size: float = 14):
    fig, ax = plt.subplots(figsize=(size, size * frame.shape[0] / frame.shape[1]))
    ax.imshow(frame)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    return fig


def grid(clip, result, frames=(0, 10, 20, 30), size: float = 16):
    """Несколько кадров подряд — видно, что номера не сбиваются."""
    frames = [f for f in frames if f < len(clip)]
    cols = 2
    rows = (len(frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(size, size * 0.36 * rows))
    for ax, idx in zip(np.ravel(axes), frames):
        det = result.detections[result.detections["кадр"] == idx]
        ax.imshow(draw(clip.frames[idx], det, result.masks.get(idx)))
        ax.set_title(f"кадр {idx} — объектов {len(det)}", fontsize=11)
        ax.axis("off")
    for ax in np.ravel(axes)[len(frames):]:
        ax.axis("off")
    fig.suptitle(f"Запрос «{result.prompt}», порог {result.conf}", fontsize=14)
    fig.tight_layout()
    return fig


def counts_plot(result, clip, target: str = "VEHICLE"):
    """Сколько объектов в кадре: модель против эталона."""
    truth = clip.boxes[clip.boxes["класс"] == target].groupby("кадр").size()
    got = result.per_frame()
    idx = range(len(clip))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(idx, [truth.get(i, 0) for i in idx], lw=2.5, label=f"эталон ({target})")
    ax.plot(idx, [got.get(i, 0) for i in idx], lw=2.5, ls="--",
            label=f"модель («{result.prompt}»)")
    ax.set_xlabel("кадр")
    ax.set_ylabel("объектов в кадре")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def sweep_plot(table, target: str = "VEHICLE"):
    """Кривая «порог против числа объектов» — управленческое решение в одной картинке."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(table["порог"], table["насчитано"], marker="o", lw=2.5, label="насчитано")
    ax.axhline(table["на самом деле"].iloc[0], color="#E63946", ls="--", lw=2,
               label=f"на самом деле ({target})")
    ax.set_xlabel("порог уверенности")
    ax.set_ylabel("объектов за отрезок")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    return fig


def prompts_plot(table):
    """Столбики по запросам — то самое «одно слово меняет всё»."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#2A9D8F" if e >= 0 else "#E63946" for e in table["ошибка"]]
    ax.bar(table["запрос"], table["насчитано"], color=colors)
    ax.axhline(table["на самом деле"].iloc[0], color="#264653", ls="--", lw=2,
               label="на самом деле")
    for i, v in enumerate(table["насчитано"]):
        ax.text(i, v + 0.15, str(v), ha="center", fontsize=11)
    ax.set_ylabel("объектов за отрезок")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig
