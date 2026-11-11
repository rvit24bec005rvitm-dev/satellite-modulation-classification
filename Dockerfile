FROM node:22-bookworm-slim AS frontend
WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
COPY frontend/vendor/ ./vendor/
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.14-slim AS application
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    SATMOD_SERVE_FRONTEND=1 \
    SATMOD_DATA_DIR=/data \
    SATMOD_PUBLIC=1 \
    PORT=8000 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1
COPY requirements-web.txt ./
RUN python -m pip install --no-cache-dir -r requirements-web.txt \
    && useradd --create-home --uid 10001 satmod \
    && mkdir /data && chown satmod:satmod /data
COPY backend/ ./backend/
COPY utils/ ./utils/
COPY training/ ./training/
COPY models/ ./models/
COPY experiments/ ./experiments/
COPY classifier_samples.npz normalization_stats_all_snr.npz satellite_modulation_cnn_all_snr.keras evaluation_results.json snr_accuracy_results.csv ./
COPY --from=frontend /build/frontend/dist/ ./frontend/dist/
USER satmod
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/api/health',timeout=4).read()" || exit 1
CMD ["sh", "-c", "exec python -m uvicorn backend.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --proxy-headers"]
