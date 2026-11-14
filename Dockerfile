ARG PYTHON_IMAGE=python:3.14-slim
FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
COPY frontend/vendor/ ./vendor/
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM ${PYTHON_IMAGE} AS application
ARG REQUIREMENTS_FILE=requirements-web.txt
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SATMOD_SERVE_FRONTEND=1 \
    SATMOD_DATA_DIR=/data \
    SATMOD_PUBLIC=1 \
    SATMOD_REPLICAS=1 \
    WEB_CONCURRENCY=1 \
    PORT=8000 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1
COPY requirements-web.txt requirements-neural.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip==26.3 \
    && python -m pip install --no-cache-dir -r "${REQUIREMENTS_FILE}" \
    && useradd --create-home --uid 10001 satmod \
    && mkdir /data && chown satmod:satmod /data
COPY backend/ ./backend/
COPY utils/ ./utils/
COPY training/ ./training/
COPY scripts/workspace_backup.py ./scripts/workspace_backup.py
COPY models/ ./models/
COPY experiments/ ./experiments/
COPY classifier_samples.npz normalization_stats_all_snr.npz satellite_modulation_cnn_all_snr.keras evaluation_results.json snr_accuracy_results.csv ./
COPY --from=frontend /build/frontend/dist/ ./frontend/dist/
# Default images start unprivileged. A platform with a root-owned mount can
# build with RUNTIME_USER=root; the entrypoint initializes /data then drops UID.
ARG RUNTIME_USER=satmod
USER ${RUNTIME_USER}
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/api/ready',timeout=4).read()" || exit 1
CMD ["python", "-m", "backend.services.deployment_entrypoint"]
