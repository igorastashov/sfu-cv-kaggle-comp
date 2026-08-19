#!/usr/bin/env bash
# Подъём стенда на целевой машине. Запускается после распаковки архива:
#
#   bash tools/start_stand.sh
#
# Скрипт проверяет условия, собирает образ и поднимает ноутбук на порту 8888.
# Первая сборка идёт долго и требует выхода в интернет.
#
# Имена переменных латиницей намеренно: оболочка допускает в именах только
# латинские буквы, цифры и подчёркивание.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

fail() { printf '\nОстановлено: %s\n' "$1" >&2; exit 1; }

# --- проверка условий --------------------------------------------------------

command -v docker >/dev/null 2>&1 || fail "docker не установлен."

if docker compose version >/dev/null 2>&1; then
    compose=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    compose=(docker-compose)
else
    fail "не найден docker compose."
fi

docker info >/dev/null 2>&1 || fail "демон docker не отвечает. Проверьте службу и права пользователя."

# Без доступа к ускорителю расчёт уходит на процессор и одна ячейка считается
# минутами вместо секунд. Это не отказ, но занятие в таком виде не проводится.
if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
    printf 'Ускоритель:     доступен контейнерам\n'
else
    printf 'Ускоритель:     НЕ доступен контейнерам\n'
    printf 'Нужен пакет nvidia-container-toolkit и перезапуск docker.\n'
    printf 'Продолжить без ускорителя? Расчёт будет идти минутами. [y/N] '
    read -r answer
    case "$answer" in
        y|Y) ;;
        *) fail "ускоритель не настроен." ;;
    esac
fi

# --- состав ------------------------------------------------------------------

for item in stand notebooks docker/compose.yaml requirements.txt; do
    [ -e "$item" ] || fail "нет $item. Архив распакован не полностью."
done

size_mb() { du -sm "$1" 2>/dev/null | cut -f1 || echo 0; }

data_mb=$(size_mb data)
models_mb=$(size_mb models)
data_mb=${data_mb:-0}
models_mb=${models_mb:-0}

[ "$data_mb"   -ge 100 ]  || fail "набор данных пуст или неполон (${data_mb} МБ)."
[ "$models_mb" -ge 1000 ] || fail "веса моделей отсутствуют (${models_mb} МБ)."

printf 'Набор данных:   %s МБ\n' "$data_mb"
printf 'Веса моделей:   %s МБ\n' "$models_mb"

# --- пароль доступа ----------------------------------------------------------

if [ ! -f docker/.env ]; then
    token=$(head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9')
    cat > docker/.env <<EOF
JUPYTER_TOKEN=$token
JUPYTER_PORT=8888
DATA_PATH=../data
MODELS_PATH=../models
EOF
    printf 'Пароль доступа: создан новый\n'
fi

token=$(grep '^JUPYTER_TOKEN=' docker/.env | head -1 | cut -d= -f2- | tr -d '\r')
port=$(grep '^JUPYTER_PORT=' docker/.env | head -1 | cut -d= -f2- | tr -d '\r' || true)
port=${port:-8888}
[ -n "$token" ] || fail "в docker/.env пустой JUPYTER_TOKEN."

# --- сборка и запуск ---------------------------------------------------------

printf '\nСобираю образ. Первый раз это занимает продолжительное время.\n\n'
"${compose[@]}" -f docker/compose.yaml up -d --build

printf '\nЖду готовности ноутбука.\n'
# Ответ 403 тоже означает готовность: страница закрыта паролем, но отвечает.
for _ in $(seq 1 60); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://localhost:${port}/api" || echo 000)
    case "$code" in
        200|403) break ;;
    esac
    sleep 5
done

ip=$(hostname -I 2>/dev/null | awk '{print $1}')
ip=${ip:-АДРЕС-МАШИНЫ}

printf '\nСтенд поднят.\n\n'
printf '  Ссылка:  http://%s:%s/lab?token=%s\n' "$ip" "$port" "$token"
printf '  Ноутбук: notebooks/1_chto-vidit-mashina.ipynb\n\n'
printf 'Журнал:    %s -f docker/compose.yaml logs -f stand\n' "${compose[*]}"
printf 'Остановка: %s -f docker/compose.yaml down\n' "${compose[*]}"
