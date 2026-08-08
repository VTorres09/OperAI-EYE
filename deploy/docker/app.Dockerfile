FROM python:3.11-slim-bookworm

ARG TARGETARCH

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/usr/local/lib/python3.11/site-packages:/usr/lib/python3/dist-packages

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
    && if [ "$TARGETARCH" = "arm64" ]; then \
        curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key \
          | gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg; \
        echo "deb [arch=arm64 signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] https://archive.raspberrypi.com/debian/ bookworm main" \
          > /etc/apt/sources.list.d/raspberrypi.list; \
        apt-get update; \
        apt-get install -y --no-install-recommends python3-picamera2; \
      fi \
    && rm -rf /var/lib/apt/lists/*

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
