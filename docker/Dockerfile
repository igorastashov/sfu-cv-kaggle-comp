# Основа выбрана под то, что проверено локально: Ubuntu 22.04 несёт Python 3.10,
# ускоритель работает через CUDA 12.6. Те же версии стоят в рабочем окружении.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-dev \
        libgl1 libglib2.0-0 \
        git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python3

RUN python -m pip install --upgrade pip setuptools wheel

# Отдельной строкой: обычный индекс отдаёт сборку без ускорителя.
RUN python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install -r /tmp/requirements.txt

WORKDIR /workspace

EXPOSE 8888

# Токен задаётся снаружи через переменную окружения. Пустой токен означает открытый
# доступ, поэтому в рабочем запуске он обязателен.
CMD ["sh", "-c", "jupyter lab \
     --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
     --ServerApp.root_dir=/workspace \
     --ServerApp.token=${JUPYTER_TOKEN} \
     --ServerApp.password='' \
     --ServerApp.allow_origin='*'"]
