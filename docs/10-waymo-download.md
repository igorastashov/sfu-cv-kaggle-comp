# Как скачать данные Waymo для стенда

Инструкция для организатора. Скачивается **один валидационный сегмент**, этого достаточно, чтобы собрать и проверить стенд целиком.

Полный набор весит около 2,29 ТБ — качать его не нужно и не следует.

---

## 0. Что и почему берём

Формат v2 колоночный: данные разложены по компонентам, и каждый качается отдельно. Берём только нужное.

| Компонент | Зачем | Вес |
|---|---|---|
| `camera_image` | сами кадры | средний |
| `camera_box` | рамки с номерами треков → детекция и трекинг | малый |
| `camera_segmentation` | паноптические маски → сегментация | средний |
| `camera_calibration` | параметры камер | крошечный |
| `vehicle_pose` | положение машины | крошечный |
| `lidar_camera_projection` | привязка точек лидара к пикселям → **глубина** | средний |
| `lidar` | сами облака точек, нужны вместе с проекцией для глубины | **большой** |

Первые пять — обязательны. Последние два нужны только для ступени глубины, и `lidar` — самая тяжёлая часть набора. Если по объёму не проходим, ступень глубины делается без эталона: монокулярная модель рисует карту расстояний, но сверять её будет не с чем.

**Почему валидационный сплит, а не обучающий.** Разметка сегментации камер положена неравномерно: в обучающем сплите это последовательности всего по 5 кадров (25 размеченных изображений на сегмент), в валидационном — полные прогоны на 5 Гц, 100 кадров на 5 камер, то есть 500 размеченных изображений. Для трекинга нужна длинная последовательность, поэтому берём валидацию.

---

## 1. Доступ

1. Завести или взять Google-аккаунт.
2. Зайти на [waymo.com/open](https://waymo.com/open/), зарегистрироваться и **принять условия использования**. Без этого бакет не откроется.
3. Поставить Google Cloud CLI: [cloud.google.com/sdk/docs/install](https://cloud.google.com/sdk/docs/install).
4. Авторизоваться:

```powershell
gcloud auth login
```

---

## 2. Какую строку брать на странице загрузки

На [waymo.com/open/download](https://waymo.com/open/download/) нужна ровно одна строка:

> **Perception Dataset — V 2.0.1 (Modular without maps), March 2024**

Остальное не брать:

| Строка | Почему нет |
|---|---|
| Motion Dataset | траектории агентов, изображений нет |
| Perception **V 1.4.3** | старый формат TFRecord с привязкой к TensorFlow |
| End-to-End Driving | другая задача |
| Community Contributions | надстройки поверх основного набора |

Бакет: **`gs://waymo_open_dataset_v_2_0_1`**. Внутри папки `training/`, `validation/`, `testing/`.

```powershell
$BUCKET = "gs://waymo_open_dataset_v_2_0_1"
gcloud storage ls $BUCKET/validation/
```

Ожидаем список компонентов: `camera_image/`, `camera_box/`, `camera_segmentation/` и прочие.

**Без командной строки тоже можно.** Кнопка Download ведёт в браузер Google Cloud Storage. Там те же папки, и нужные пять файлов скачиваются кликами. CLI удобнее только тем, что список сегментов проще посмотреть и отфильтровать.

---

## 3. Выбрать сегмент

Имя сегмента (контекста) выглядит как `10023947602400723454_1120_000_1140_000`. Нам нужен такой, у которого **есть разметка сегментации**, поэтому список берём именно из `camera_segmentation`:

```powershell
gcloud storage ls $BUCKET/validation/camera_segmentation/ | Select-Object -First 5
```

Берём **первое** имя из выдачи, отрезаем путь и расширение `.parquet`. Например, если вывелось

```
gs://waymo_open_dataset_v_2_0_1/validation/camera_segmentation/550171902340535682_2640_000_2660_000.parquet
```

то

```powershell
$CTX = "550171902340535682_2640_000_2660_000"
```

---

## 4. Скачать

```powershell
$DST = "d:\__projects__\sfu-kaggle\data\waymo\$CTX"
New-Item -ItemType Directory -Force $DST | Out-Null

foreach ($c in @("camera_image","camera_box","camera_segmentation","camera_calibration","vehicle_pose")) {
    New-Item -ItemType Directory -Force "$DST\$c" | Out-Null
    gcloud storage cp "$BUCKET/validation/$c/$CTX.parquet" "$DST\$c\"
}
```

Отдельно, для ступени глубины — **сначала посмотрите размер**, это самая тяжёлая часть:

```powershell
gcloud storage du -s -h "$BUCKET/validation/lidar/$CTX.parquet"
gcloud storage du -s -h "$BUCKET/validation/lidar_camera_projection/$CTX.parquet"
```

Если суммарно приемлемо — качаем:

```powershell
foreach ($c in @("lidar","lidar_camera_projection","lidar_calibration")) {
    New-Item -ItemType Directory -Force "$DST\$c" | Out-Null
    gcloud storage cp "$BUCKET/validation/$c/$CTX.parquet" "$DST\$c\"
}
```

---

## 5. Замерить и сообщить

```powershell
Get-ChildItem $DST -Recurse -File |
    Group-Object { $_.Directory.Name } |
    ForEach-Object {
        "{0,-28} {1,8:N1} МБ" -f $_.Name, (($_.Group | Measure-Object Length -Sum).Sum / 1MB)
    }
"ИТОГО: {0:N1} МБ" -f ((Get-ChildItem $DST -Recurse -File | Measure-Object Length -Sum).Sum / 1MB)
```

**Это и есть ответ на открытый вопрос ресерча.** От суммы зависит:

- влезает ли подвыборка в лимит датасета на Codenrock;
- берём один сегмент или несколько;
- оставляем ступень глубины с эталоном или без.

Если общий вес без лидара выходит за пару гигабайт — переходим на KITTI, структура стенда при этом не меняется.

---

## 6. Куда всё кладётся

```
data/waymo/<context_name>/
    camera_image/<context_name>.parquet
    camera_box/<context_name>.parquet
    camera_segmentation/<context_name>.parquet
    camera_calibration/<context_name>.parquet
    vehicle_pose/<context_name>.parquet
    lidar/…                        (опционально)
    lidar_camera_projection/…      (опционально)
    lidar_calibration/…            (опционально)
```

Папка `data/` закрыта от git — в репозиторий ничего не уйдёт.

---

## Если что-то не открывается

| Симптом | Причина | Что делать |
|---|---|---|
| `AccessDeniedError` на бакете | не приняты условия использования | зайти на waymo.com/open под тем же аккаунтом и принять |
| Бакета `v_2_0_1` нет | вышла другая версия | взять `v_2_0_0`, структура та же |
| В `camera_segmentation` пусто | смотрите не тот сплит | нужен `validation`, а не `training` |
| `gcloud` не найден | CLI не в PATH | перезапустить терминал после установки |
