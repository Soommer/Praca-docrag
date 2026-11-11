# docrag - obraz z GPU (CUDA 12.8). Działa na RTX 3050 (sm_86) i RTX 5070 (sm_120).
# Wymaga na hoście: sterownik NVIDIA >= 570 i nvidia-container-toolkit (GPU przez CDI w Podmanie).
#
#   podman compose -f compose.podman-3050.yml up -d --build      (laptop, RTX 3050)
#   docker compose -f compose.windows-5070.yml up -d --build     (Windows, RTX 5070)
#
FROM docker.io/library/python:3.12-slim

# torch z indeksu PyTorch (build CUDA) - wersje przypięte, bo docling ciągnie najnowszy
# torchvision, który wymaga nowszego torcha, a pip podmieniłby build CUDA na inny
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.11.0
ARG TORCHVISION_VERSION=0.26.0
# api = serwer HTTP; dopisz ",docling" dla ingestu DOCX/PPTX/HTML w kontenerze (+ok. 2 GB)
ARG EXTRAS=api

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models \
    DOCRAG_CONFIG=/app/config.yaml

WORKDIR /app

# 1. torch (największa warstwa, zmienia się najrzadziej)
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url ${TORCH_INDEX} \
 && printf 'torch==%s\ntorchvision==%s\n' "${TORCH_VERSION}" "${TORCHVISION_VERSION}" > /tmp/constraints.txt

# 2. zależności projektu - tylko pyproject.toml, żeby zmiana kodu nie przebudowywała tej warstwy
COPY pyproject.toml ./
RUN mkdir -p src/docrag && touch src/docrag/__init__.py \
 && pip install -c /tmp/constraints.txt ".[${EXTRAS}]" \
 && pip uninstall -y docrag \
 && rm -rf src build

# 3. kod
COPY src ./src
RUN pip install --no-deps . && rm -rf build
# config.yaml montuje compose (config.podman-3050.yaml / config.windows-5070.yaml) - obraz jest wspólny

EXPOSE 8008
CMD ["docrag", "serve", "--host", "0.0.0.0", "--port", "8008"]
