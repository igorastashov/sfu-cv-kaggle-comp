"""Запуск модели по текстовому запросу и подсчёт результата."""
from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import session

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "sam3_hf"

_cache: dict[str, object] = {}

def load_model(model_dir: Path | str | None = None):
    """Модель грузится один раз за сессию и кешируется."""
    if "model" not in _cache:
        from transformers import AutoModel, AutoProcessor
        path = str(model_dir or MODEL_DIR)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        _cache["processor"] = AutoProcessor.from_pretrained(path)
        _cache["model"] = AutoModel.from_pretrained(path, dtype=dtype).to(device).eval()
        _cache["device"], _cache["dtype"] = device, dtype
    return _cache["model"], _cache["processor"]


def gpu_report() -> None:
    """Сколько памяти ускорителя свободно.

    Нужно ведущему до занятия. Каждый участник работает в своём вычислительном
    ядре, и каждое ядро держит собственную копию моделей. Память освобождается
    только при перезапуске ядра, а не по окончании расчёта.
    """
    if not torch.cuda.is_available():
        print("Ускоритель не виден. Расчёт пойдёт на процессоре, это минуты на отрезок.")
        return

    free, total = torch.cuda.mem_get_info()
    гб = 1024 ** 3
    свой = torch.cuda.memory_reserved() / гб
    print(f"Ускоритель: {torch.cuda.get_device_name(0)}")
    print(f"Памяти всего {total / гб:.1f} ГБ, свободно {free / гб:.1f} ГБ, "
          f"занято этим ядром {свой:.1f} ГБ")


@dataclass
class Result:
    """Что нашла модель на отрезке."""
    prompt: str
    conf: float
    min_track_len: int
    detections: pd.DataFrame          # кадр, трек, скор, рамка xyxy
    masks: dict = field(repr=False, default_factory=dict)   # кадр -> (K,H,W) bool
    seconds: float = 0.0

    @property
    def n_objects(self) -> int:
        """Сколько уникальных объектов прошло — это и есть ответ участника."""
        return self.detections["трек"].nunique()

    def per_frame(self) -> pd.Series:
        return self.detections.groupby("кадр").size()

    def track_lengths(self) -> pd.Series:
        return self.detections.groupby("трек").size().sort_values(ascending=False)


def find(clip, prompt: str, conf: float = 0.5, min_track_len: int = 1,
         keep_masks: bool = True) -> Result:
    """Найти на отрезке всё, что подходит под текстовый запрос.

    conf           — порог уверенности: ниже него объект отбрасывается
    min_track_len  — трек короче стольких кадров считается ложным
    """
    model, processor = load_model()
    device, dtype = _cache["device"], _cache["dtype"]

    session = processor.init_video_session(video=clip.frames,
                                           inference_device=device, dtype=dtype)
    processor.add_text_prompt(session, prompt)

    rows, masks, t0 = [], {}, time.time()
    with torch.inference_mode():
        for out in model.propagate_in_video_iterator(session):
            r = processor.postprocess_outputs(session, out)
            idx = out.frame_idx
            scores = r["scores"].float().cpu().numpy()
            ids = r["object_ids"].cpu().numpy()
            boxes = r["boxes"].float().cpu().numpy()
            n = min(len(ids), len(scores), len(boxes))
            keep = scores[:n] >= conf
            if keep.any():
                for oid, sc, bb in zip(ids[:n][keep], scores[:n][keep], boxes[:n][keep]):
                    rows.append({"кадр": idx, "трек": int(oid), "скор": float(sc),
                                 "x1": bb[0], "y1": bb[1], "x2": bb[2], "y2": bb[3]})
                if keep_masks:
                    masks[idx] = r["masks"][:n][torch.as_tensor(keep)].cpu().numpy()
    seconds = time.time() - t0

    del session
    if device == "cuda":
        torch.cuda.empty_cache()

    det = pd.DataFrame(rows, columns=["кадр", "трек", "скор", "x1", "y1", "x2", "y2"])
    if min_track_len > 1 and len(det):
        long_enough = det.groupby("трек").size()
        det = det[det["трек"].isin(long_enough[long_enough >= min_track_len].index)]

    return Result(prompt=prompt, conf=conf, min_track_len=min_track_len,
                  detections=det.reset_index(drop=True), masks=masks, seconds=seconds)


def score(result: Result, clip, target: str = "VEHICLE") -> dict:
    """Сравнить ответ участника с эталоном разметки."""
    truth = clip.boxes[clip.boxes["класс"] == target]["трек"].nunique()
    got = result.n_objects
    return {
        "запрос": result.prompt,
        "порог": result.conf,
        "мин. длина трека": result.min_track_len,
        "насчитано": got,
        "на самом деле": truth,
        "ошибка": got - truth,
        "точность, %": round(100 * (1 - abs(got - truth) / max(truth, 1)), 1),
        "секунд": round(result.seconds, 1),
    }


def save_submission(result: Result, clip, target: str = "VEHICLE",
                    path: str | None = None) -> pd.DataFrame:
    """Показатели одного прогона в файл, чтобы сравнивать попытки между собой.

    Имя файла по умолчанию содержит имя ноутбука: на занятии участники работают
    в одной среде, и общее имя означало бы, что они затирают результаты друг друга.
    """
    if path is None:
        path = f"результат-{session.name()}.csv"
    row = score(result, clip, target=target)
    row["цель"] = target
    df = pd.DataFrame([row])
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Сохранено в {path}\n")
    print(df.to_string(index=False))
    return df


def sweep_conf(clip, prompt: str, thresholds=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
               target: str = "VEHICLE") -> pd.DataFrame:
    """Как порог влияет на счёт. Одна модель, один прогон, разные отсечки."""
    full = find(clip, prompt, conf=0.0, min_track_len=1, keep_masks=False)
    truth = clip.boxes[clip.boxes["класс"] == target]["трек"].nunique()

    rows = []
    for t in thresholds:
        kept = full.detections[full.detections["скор"] >= t]
        got = kept["трек"].nunique()
        rows.append({"порог": t, "насчитано": got, "на самом деле": truth,
                     "ошибка": got - truth})
    return pd.DataFrame(rows)


def compare_prompts(clip, prompts: list[str], conf: float = 0.5,
                    target: str = "VEHICLE") -> pd.DataFrame:
    """Главная таблица стенда: одно слово меняет ответ."""
    truth = clip.boxes[clip.boxes["класс"] == target]["трек"].nunique()
    rows = []
    for p in prompts:
        r = find(clip, p, conf=conf, keep_masks=False)
        rows.append({"запрос": p, "насчитано": r.n_objects,
                     "на самом деле": truth, "ошибка": r.n_objects - truth,
                     "секунд": round(r.seconds, 1)})
    return pd.DataFrame(rows)
