# Multimodal Market Signal — serving image
#
# Two-stage-ish build: a slim base, deps installed first (cached across
# rebuilds when only source changes), then the project code. This image
# runs the FastAPI serving app (serving/app.py); training and the
# retrieval-index build are one-off steps run against a mounted volume
# (see docker-compose.yml), not baked into the image.

# Multimodal Market Signal — serving image
#
# Two-stage-ish build: a slim base, deps installed first (cached across
# rebuilds when only source changes), then the project code. This image
# runs the FastAPI serving app (serving/app.py) via start.sh, which trains
# the model and builds the retrieval index on first boot if they aren't
# already present (see start.sh) — needed for hosts without a persisted
# volume, like Hugging Face Spaces. Defaults to port 7860 (what Hugging
# Face Spaces expects); set $PORT to override elsewhere.

FROM python:3.11-slim

WORKDIR /app

# System deps for pyarrow / tensorflow wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x start.sh

# Hugging Face Spaces runs containers as a non-root user; make sure that
# user can write the trained-model / retrieval-index files start.sh
# generates on boot, regardless of who ends up running the container.
RUN chmod -R 777 /app

ENV PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.environ.get(\"PORT\", 7860)}/health')" || exit 1

CMD ["./start.sh"]
