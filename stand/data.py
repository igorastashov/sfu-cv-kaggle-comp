"""Чтение сегмента Waymo v2. Участник этот файл не видит — в ноутбуке только вызовы."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

CAMERAS = {1: "передняя", 2: "передняя левая", 3: "передняя правая",
           4: "боковая левая", 5: "боковая правая"}

CLASSES = {1: "VEHICLE", 2: "PEDESTRIAN", 3: "SIGN", 4: "CYCLIST"}

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "waymo"


def find_segment(root: Path | str | None = None) -> Path:
    """Первый скачанный сегмент."""
    root = Path(root) if root else DATA_ROOT
    segments = [p for p in root.iterdir() if p.is_dir() and (p / "camera_image").exists()]
    if not segments:
        raise FileNotFoundError(
            f"В {root} нет сегментов. Скачайте по инструкции docs/10-waymo-download.md"
        )
    return segments[0]


def _component(segment: Path, name: str) -> Path:
    files = list((segment / name).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Нет компонента {name} в {segment}")
    return files[0]


@dataclass
class Clip:
    """Отрезок видео с одной камеры плюс эталонная разметка."""
    frames: np.ndarray          # (N, H, W, 3)
    timestamps: list[int]
    camera: int
    boxes: pd.DataFrame         # эталон: кадр, класс, номер трека, рамка xyxy

    def __len__(self) -> int:
        return len(self.frames)

    @property
    def size(self) -> tuple[int, int]:
        return self.frames.shape[2], self.frames.shape[1]

    def truth_counts(self) -> pd.Series:
        """Сколько уникальных объектов каждого класса реально прошло по отрезку."""
        return (self.boxes.groupby("класс")["трек"].nunique()
                .rename("объектов").sort_values(ascending=False))

    def truth_per_frame(self) -> pd.Series:
        return self.boxes.groupby("кадр").size()


def load_clip(n_frames: int = 40, camera: int = 1, start: int = 0,
              segment: Path | None = None) -> Clip:
    """Загрузить отрезок: кадры одной камеры и эталонные рамки к ним."""
    segment = segment or find_segment()

    img_file = _component(segment, "camera_image")
    meta = pq.read_table(
        img_file, columns=["key.frame_timestamp_micros", "key.camera_name"]
    ).to_pandas()
    all_ts = sorted(meta.loc[meta["key.camera_name"] == camera,
                             "key.frame_timestamp_micros"].unique())
    ts = all_ts[start:start + n_frames]
    if not ts:
        raise ValueError(f"Нет кадров: start={start} при всего {len(all_ts)} кадрах")

    table = pq.read_table(img_file, filters=[("key.camera_name", "=", camera),
                                             ("key.frame_timestamp_micros", "in", ts)])
    img_col = next(c for c in table.column_names if c.endswith(".image"))
    df = (table.select(["key.frame_timestamp_micros", img_col]).to_pandas()
          .sort_values("key.frame_timestamp_micros"))
    frames = np.stack([np.array(Image.open(io.BytesIO(b)).convert("RGB"))
                       for b in df[img_col]])

    return Clip(frames=frames, timestamps=list(df["key.frame_timestamp_micros"]),
                camera=camera, boxes=_load_boxes(segment, camera, ts))


def _load_boxes(segment: Path, camera: int, ts: list[int]) -> pd.DataFrame:
    raw = pd.read_parquet(_component(segment, "camera_box"))
    sel = raw[(raw["key.camera_name"] == camera)
              & (raw["key.frame_timestamp_micros"].isin(ts))].copy()

    cx = sel["[CameraBoxComponent].box.center.x"]
    cy = sel["[CameraBoxComponent].box.center.y"]
    w = sel["[CameraBoxComponent].box.size.x"]
    h = sel["[CameraBoxComponent].box.size.y"]

    order = {t: i for i, t in enumerate(sorted(ts))}
    return pd.DataFrame({
        "кадр": sel["key.frame_timestamp_micros"].map(order),
        "класс": sel["[CameraBoxComponent].type"].map(CLASSES).fillna("UNKNOWN"),
        "трек": sel["key.camera_object_id"],
        "x1": cx - w / 2, "y1": cy - h / 2, "x2": cx + w / 2, "y2": cy + h / 2,
    }).sort_values(["кадр", "трек"]).reset_index(drop=True)


def segment_summary(segment: Path | None = None) -> pd.DataFrame:
    """Что вообще есть в сегменте — для первой ячейки ноутбука."""
    segment = segment or find_segment()
    meta = pq.read_table(_component(segment, "camera_image"),
                         columns=["key.frame_timestamp_micros", "key.camera_name"]).to_pandas()
    box = pd.read_parquet(_component(segment, "camera_box"))
    box["класс"] = box["[CameraBoxComponent].type"].map(CLASSES).fillna("UNKNOWN")

    rows = []
    for cam, name in CAMERAS.items():
        b = box[box["key.camera_name"] == cam]
        rows.append({
            "камера": name,
            "кадров": int((meta["key.camera_name"] == cam).sum()),
            "рамок": len(b),
            "объектов": b["key.camera_object_id"].nunique(),
            "машин": int((b["класс"] == "VEHICLE").sum()),
            "пешеходов": int((b["класс"] == "PEDESTRIAN").sum()),
            "велосипедистов": int((b["класс"] == "CYCLIST").sum()),
        })
    return pd.DataFrame(rows)
