"""Сравнение изображений: поиск похожего по векторному представлению."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "dinov3-vitb16"
_cache: dict = {}


def load_model(model_dir=None):
    """Базовая сеть, обученная без разметки. Ничего не распознаёт, только описывает."""
    if "model" not in _cache:
        from transformers import AutoImageProcessor, AutoModel
        path = str(model_dir or MODEL_DIR)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _cache["proc"] = AutoImageProcessor.from_pretrained(path)
        _cache["model"] = AutoModel.from_pretrained(path).to(device).eval()
        _cache["device"] = device
    return _cache["model"], _cache["proc"]


def _crops(result, clip, pad: int = 8):
    """Вырезанные участки вокруг каждого найденного объекта."""
    out = []
    for _, d in result.detections.iterrows():
        img = clip.frames[int(d["кадр"])]
        h, w = img.shape[:2]
        x1 = max(int(d["x1"]) - pad, 0)
        y1 = max(int(d["y1"]) - pad, 0)
        x2 = min(int(d["x2"]) + pad, w)
        y2 = min(int(d["y2"]) + pad, h)
        if x2 - x1 < 12 or y2 - y1 < 12:
            continue
        out.append({"кадр": int(d["кадр"]), "трек": int(d["трек"]),
                    "фрагмент": img[y1:y2, x1:x2]})
    return out


def describe(result, clip, batch: int = 32) -> tuple[np.ndarray, pd.DataFrame]:
    """Каждый найденный объект превращается в набор чисел.

    Похожие по виду объекты получают близкие наборы. После этого сравнение
    изображений сводится к арифметике.
    """
    model, proc = load_model()
    items = _crops(result, clip)
    if not items:
        return np.zeros((0, 1)), pd.DataFrame()

    vecs = []
    for i in range(0, len(items), batch):
        imgs = [it["фрагмент"] for it in items[i:i + batch]]
        inputs = proc(images=imgs, return_tensors="pt").to(_cache["device"])
        with torch.inference_mode():
            out = model(**inputs)
        v = out.pooler_output if getattr(out, "pooler_output", None) is not None \
            else out.last_hidden_state.mean(1)
        vecs.append(v.float().cpu().numpy())

    v = np.concatenate(vecs)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    meta = pd.DataFrame([{"кадр": it["кадр"], "трек": it["трек"]} for it in items])
    _cache["items"] = items
    return v, meta


def similar(vectors, meta, query: int = 0, top: int = 5) -> pd.DataFrame:
    """Самые похожие на выбранный объект, по убыванию сходства."""
    sim = vectors @ vectors[query]
    order = np.argsort(-sim)
    order = [i for i in order if i != query][:top]
    return pd.DataFrame({
        "номер": order,
        "сходство": np.round(sim[order], 3),
        "кадр": meta.loc[order, "кадр"].values,
        "трек": meta.loc[order, "трек"].values,
        "тот же объект": meta.loc[order, "трек"].values == meta.loc[query, "трек"],
    })


def show_similar(vectors, meta, query: int = 0, top: int = 5, size: float = 16):
    """Слева искомый участок, справа ближайшие по виду."""
    items = _cache.get("items")
    if not items:
        raise RuntimeError("Сначала выполните describe")
    table = similar(vectors, meta, query, top)

    fig, axes = plt.subplots(1, top + 1, figsize=(size, size / (top + 1) * 1.6))
    axes[0].imshow(items[query]["фрагмент"])
    axes[0].set_title(f"искомый\nкадр {meta.loc[query, 'кадр']}, №{meta.loc[query, 'трек']}",
                      fontsize=10)
    axes[0].axis("off")
    for ax, (_, row) in zip(axes[1:], table.iterrows()):
        ax.imshow(items[int(row["номер"])]["фрагмент"])
        цвет = "#2A9D8F" if row["тот же объект"] else "#E63946"
        ax.set_title(f"сходство {row['сходство']:.2f}\nкадр {row['кадр']}, №{row['трек']}",
                     fontsize=10, color=цвет)
        ax.axis("off")
    fig.suptitle("Поиск похожего: зелёным отмечен тот же объект, красным — другой",
                 fontsize=12)
    fig.tight_layout()
    return fig, table


def quality(vectors, meta) -> dict:
    """Насколько поиск похожего попадает в тот же объект.

    Проверка честная: правильным считается только совпадение номера объекта.
    """
    if len(vectors) < 3:
        return {}
    sim = vectors @ vectors.T
    np.fill_diagonal(sim, -2)
    best = np.argmax(sim, axis=1)
    top1 = (meta["трек"].values[best] == meta["трек"].values).mean()

    top3 = []
    for i in range(len(vectors)):
        cand = np.argsort(-sim[i])[:3]
        top3.append(meta["трек"].values[i] in meta["трек"].values[cand])
    return {
        "участков сравнено": len(vectors),
        "длина описания": int(vectors.shape[1]),
        "первый ответ верен, %": round(100 * float(top1), 1),
        "верный среди трёх, %": round(100 * float(np.mean(top3)), 1),
    }


def map_objects(vectors, meta, size: float = 8):
    """Карта сходства: похожие участки располагаются рядом."""
    from sklearn.decomposition import PCA
    from .viz import color_of

    if len(vectors) < 3:
        raise ValueError("Слишком мало участков для карты")
    xy = PCA(n_components=2).fit_transform(vectors)

    fig, ax = plt.subplots(figsize=(size, size))
    for tid, g in meta.assign(x=xy[:, 0], y=xy[:, 1]).groupby("трек"):
        ax.scatter(g["x"], g["y"], s=70, color=color_of(tid) / 255, label=f"№{tid}")
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.set_title("Карта сходства участков. Один цвет — один объект", fontsize=12)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig
