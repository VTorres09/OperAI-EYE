FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY deploy/docker/app-requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-compile -r /tmp/requirements.txt \
    && groupadd --gid 10001 operai \
    && useradd --uid 10001 --gid operai --no-create-home operai \
    && mkdir -p /data \
    && chown operai:operai /data

COPY --chown=operai:operai operai_eye /app/operai_eye
COPY --chown=operai:operai deploy/docker/config.toml /app/config.toml

USER operai

EXPOSE 8000

CMD ["uvicorn", "operai_eye.edge.compose_api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
