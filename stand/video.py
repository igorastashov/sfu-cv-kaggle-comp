"""Видеоролики прямо в ноутбуке: исходная запись, сопровождение объектов, вид сверху."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .viz import color_of, draw

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "video"


def _writer(path: Path, fps: int):
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(str(path), fps=fps, codec="libx264",
                              macro_block_size=None, quality=7)


def show(path: str | Path, width: int = 900):
    """Встроить готовый ролик в вывод ячейки."""
    from IPython.display import Video
    return Video(str(path), embed=True, width=width,
                 html_attributes="controls loop autoplay muted")


def _even(img: np.ndarray) -> np.ndarray:
    """Кодек принимает только чётные стороны. У боковых камер высота другая."""
    return img[:img.shape[0] // 2 * 2, :img.shape[1] // 2 * 2]


def _fit(img: np.ndarray, max_width: int) -> np.ndarray:
    if img.shape[1] <= max_width:
        return _even(img)
    h = int(img.shape[0] * max_width / img.shape[1])
    ys = (np.arange(h) * img.shape[0] / h).astype(int)
    xs = (np.arange(max_width) * img.shape[1] / max_width).astype(int)
    return _even(img[np.ix_(ys, xs)])


def raw(clip, fps: int = 10, max_width: int = 960, name: str = "запись.mp4"):
    """Выбранный отрезок как обычное видео, без разметки и без модели."""
    path = OUT / name
    with _writer(path, fps) as w:
        for f in clip.frames:
            w.append_data(_fit(f, max_width))
    print(f"Готово: {len(clip)} кадров, {len(clip) / fps:.1f} с")
    return show(path)


def tracking(clip, result, fps: int = 10, max_width: int = 960,
             name: str = "сопровождение.mp4"):
    """Найденные объекты с номерами. Номер держится, пока модель не потеряла объект."""
    from PIL import Image, ImageDraw

    path = OUT / name
    with _writer(path, fps) as w:
        for i in range(len(clip)):
            det = result.detections[result.detections["кадр"] == i]
            frame = draw(clip.frames[i], det, result.masks.get(i), thickness=5)
            img = Image.fromarray(_fit(frame, max_width))
            d = ImageDraw.Draw(img)
            k = img.size[0] / clip.frames[i].shape[1]
            for _, r in det.iterrows():
                c = tuple(int(v) for v in color_of(r["трек"]))
                x, y = int(r["x1"] * k), max(int(r["y1"] * k) - 22, 0)
                d.rectangle([x, y, x + 74, y + 20], fill=(0, 0, 0))
                d.text((x + 5, y + 4), f"№{int(r['трек'])}", fill=c)
            d.rectangle([6, 6, 190, 30], fill=(0, 0, 0))
            d.text((12, 12), f"кадр {i}   объектов {len(det)}", fill=(255, 255, 255))
            w.append_data(np.array(img))

    n_ids = result.detections["трек"].nunique()
    print(f"Уникальных номеров за отрезок: {n_ids}")
    return show(path)


def lidar_tracks(clip, result, fps: int = 10, span: float = 30,
                 name: str = "лидар.mp4", segment=None):
    """Вид сверху во времени. Серое — облако точек, цветное — найденные объекты.

    Объекты найдены по камере. Их положение на плоскости берётся по точкам
    дальномера, попавшим внутрь контура. По самому облаку точек поиск не ведётся.
    """
    from .lidar import trajectories, xyz

    tr = trajectories(result, clip, segment)
    path = OUT / name
    seen: dict[int, list] = {}

    matplotlib.use("Agg")
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    with _writer(path, fps) as w:
        for i in range(len(clip)):
            ax.clear()
            pts = xyz(clip, i, segment)
            ax.scatter(pts[:, 1], pts[:, 0], s=0.4, c="#D5D9DE")

            for _, r in tr[tr["кадр"] == i].iterrows():
                tid = int(r["трек"])
                seen.setdefault(tid, []).append((r["вбок, м"], r["вперёд, м"]))
                c = color_of(tid) / 255
                path_pts = np.array(seen[tid])
                ax.plot(path_pts[:, 0], path_pts[:, 1], "-", lw=2, color=c, alpha=0.6)
                ax.scatter([r["вбок, м"]], [r["вперёд, м"]], s=150, color=c,
                           edgecolors="white", linewidths=1.5, zorder=3)
                ax.text(r["вбок, м"], r["вперёд, м"] + 1.4, f"№{tid}", fontsize=10,
                        ha="center", color=c, zorder=4)

            ax.plot(0, 0, marker="^", ms=15, color="#E63946")
            ax.set_xlim(-span, span)
            ax.set_ylim(-4, span * 1.7)
            ax.set_xlabel("влево / вправо, м")
            ax.set_ylabel("вперёд, м")
            ax.set_title(f"Кадр {i}. Объекты найдены камерой, расстояние измерено дальномером",
                         fontsize=10)
            ax.set_aspect("equal")
            ax.grid(alpha=0.2)
            fig.canvas.draw()
            frame_rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            # кодек принимает только чётные размеры
            frame_rgb = frame_rgb[:frame_rgb.shape[0] // 2 * 2, :frame_rgb.shape[1] // 2 * 2]
            w.append_data(frame_rgb.copy())
    plt.close(fig)

    print(f"Объектов на плоскости: {len(seen)}")
    return show(path)


def cameras(cams: dict, results: dict | None = None, fps: int = 10,
            max_width: int = 640):
    """Отдельный ролик на каждую камеру. Видно, как объект переходит из одной в соседнюю."""
    from IPython.display import HTML, display

    from .data import CAMERAS
    from .multicam import PANORAMA

    for cam in PANORAMA:
        clip = cams[cam]
        name = f"камера-{cam}.mp4"
        path = OUT / name
        res = results.get(cam) if results else None
        with _writer(path, fps) as w:
            for i in range(len(clip)):
                frame = clip.frames[i]
                if res is not None:
                    det = res.detections[res.detections["кадр"] == i]
                    frame = draw(frame, det, res.masks.get(i), thickness=5)
                w.append_data(_fit(frame, max_width))
        display(HTML(f"<b>{CAMERAS[cam]}</b>"))
        display(show(path, width=max_width))
