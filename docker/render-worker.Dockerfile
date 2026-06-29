FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /src/magenta-realtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      build-essential \
      ca-certificates \
      git \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /src/magenta-realtime

FROM base AS jax-gpu

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e ".[jax]" "jax[cuda13]"

CMD ["python", "-m", "magenta_rt.render_worker"]

FROM base AS mlx-cpu

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e ".[mlx]"

CMD ["python", "-m", "magenta_rt.render_worker"]
