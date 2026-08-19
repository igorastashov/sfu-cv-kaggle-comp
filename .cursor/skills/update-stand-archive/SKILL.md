---
name: update-stand-archive
description: >-
  Полное обновление стенда sfu-kaggle на машине 118 через архив: упаковка
  локально, перенос, распаковка, перезапуск контейнера и проверка. Использовать
  когда нужно синхронизировать весь репозиторий включая data/models, или когда
  пользователь просит обновить архив на 118.
---

# Обновление стенда через архив (118)

Целевая машина: `192.168.52.118`, пользователь `dmd`, каталог `/home/dmd/sfu-kaggle`.

## Когда какой способ

| Задача | Команда | Время |
| --- | --- | --- |
| Правки `stand/`, `notebooks/` | `python tools/sync_remote.py` | секунды |
| Полное обновление (data, models, всё) | `python tools/deploy_archive.py` | ~10–20 мин |
| Код + пересоздание контейнера + GPU-check | `python tools/verify_remote.py` | ~1 мин |

## Полный цикл (архив)

Перед упаковкой пересоберите ноутбук, если меняли `tools/build_notebook.py`:

```powershell
python tools/build_notebook.py
```

Учётные данные — через переменную окружения, не в командной строке:

```powershell
cd d:\__projects__\sfu-kaggle
$env:SFU_REMOTE_PASSWORD = '<пароль ssh dmd>'
python tools/deploy_archive.py
```

Скрипт делает по порядку:

1. `tools/pack_transfer.ps1` — архив `..\sfu-kaggle-YYYYMMDD.zip` + `.sha256`
2. Загрузка архива на 118 в `/home/dmd/`
3. `sha256sum -c` на удалённой машине
4. Остановка контейнера, замена `/home/dmd/sfu-kaggle`, распаковка
5. `docker compose down && up -d` (fallback на `docker-compose`)
6. Проверка: CUDA, Jupyter, контрольные суммы ключевых файлов

Флаги:

| Флаг | Действие |
| --- | --- |
| `--pack-only` | только собрать архив локально |
| `--deploy-only ПУТЬ.zip` | только загрузить и распаковать готовый архив |
| `--skip-pack` | не пересобирать, взять последний `..\sfu-kaggle-*.zip` |
| `--verify-only` | сверить sha256 ключевых файлов локаль ↔ 118 + GPU/Jupyter |

Если ноутбук менялся после упаковки — досинхронизируйте код:

```powershell
python tools/build_notebook.py
python tools/sync_remote.py
python tools/deploy_archive.py --verify-only
```

## Проверка после деплоя

```powershell
$env:SFU_REMOTE_PASSWORD = '...'
python tools/sync_remote.py --status
```

На 118 внутри контейнера:

```bash
docker exec sfu-cv-stand python -c "import torch; print(torch.cuda.is_available())"
docker exec sfu-cv-stand nvidia-smi -L
```

Ожидается: `True` и `RTX 3090`.

## GPU на RedOS (118)

В `docker/compose.yaml` обязательны:

```yaml
privileged: true
runtime: nvidia
```

Без них при `no-cgroups = true` в `/etc/nvidia-container-runtime/config.toml` CUDA в контейнере будет `False`.

## Jupyter

Токен в `docker/.env` на 118. Не меняется при деплое, если локальный `docker/.env` тот же.

```
http://192.168.52.118:8888/lab?token=<JUPYTER_TOKEN>
```

Ноутбук: `notebooks/1_chto-vidit-mashina.ipynb`

## Если распаковка на 118

`unzip` может отсутствовать — скрипт использует `python3 -m zipfile`.

Папку `notebooks/.ipynb_checkpoints` контейнер иногда создаёт от root — перед распаковкой скрипт чистит каталог через `docker run alpine rm -rf`.

## Связанные файлы

- [tools/pack_transfer.ps1](../../tools/pack_transfer.ps1) — упаковка
- [tools/deploy_archive.py](../../tools/deploy_archive.py) — полный деплой
- [tools/sync_remote.py](../../tools/sync_remote.py) — быстрая синхронизация кода
- [docs/40-perenos-stenda.md](../../docs/40-perenos-stenda.md) — документация организатора
