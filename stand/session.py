"""Кто именно работает.

На занятии несколько участников открывают в одной среде свои копии ноутбука.
Ролики и файлы результата у них должны быть разными, иначе они перезаписывают
друг друга, а при одновременной записи получается битый файл.

Разделение идёт по имени ноутбука. Оно определяется само, участник ничего
не задаёт и ошибиться не может.
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

_cache: dict[str, str] = {}

# Внутри контейнера среда ноутбуков всегда слушает этот порт, снаружи он может
# быть другим. Обращение идёт изнутри, поэтому значение постоянное.
_PORT = 8888


def set_name(name: str) -> str:
    """Задать имя вручную. Нужно, если определить его не удалось."""
    _cache["name"] = _clean(name)
    return _cache["name"]


def name() -> str:
    """Имя ноутбука, из которого идёт вызов."""
    if "name" not in _cache:
        _cache["name"] = _detect()
    return _cache["name"]


def _clean(value: str) -> str:
    """Имя используется как имя папки, поэтому лишние знаки убираются."""
    value = re.sub(r"[^\w\-. ]", "_", value).strip(" .")
    return value or "общий"


def _detect() -> str:
    """Спросить у среды ноутбуков, какому файлу принадлежит это ядро.

    Ядро своего имени не знает. Знает среда: у неё есть список сеансов,
    где номер ядра сопоставлен пути файла. Номер ядра берётся из имени
    файла подключения.
    """
    try:
        from ipykernel import get_connection_file
        kernel_id = Path(get_connection_file()).stem.replace("kernel-", "")

        url = f"http://localhost:{_PORT}/api/sessions"
        token = os.environ.get("JUPYTER_TOKEN", "")
        if token:
            url += f"?token={token}"

        with urllib.request.urlopen(url, timeout=3) as answer:
            for item in json.load(answer):
                if item.get("kernel", {}).get("id") == kernel_id:
                    return _clean(Path(item["path"]).stem)
    except Exception:
        # Ноутбук могут запустить вне контейнера или без пароля. Общая папка
        # хуже разделения, но работать это не мешает.
        pass
    return "общий"
