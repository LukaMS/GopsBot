FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY training/requirements.lock training/requirements.lock
RUN python -m pip install --upgrade pip && \
    python -m pip install -r training/requirements.lock
COPY training training

ENTRYPOINT ["python", "-m", "training.gops_rl"]
CMD ["--help"]

