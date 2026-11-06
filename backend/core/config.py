"""Environment-based service configuration; paths resolve from the repository."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
except ImportError:
    pass
MODEL_PATH = Path(os.getenv('SATMOD_MODEL_PATH', 'models/cnn_weights.npz'))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH
MAX_UPLOAD_BYTES = int(os.getenv('SATMOD_MAX_UPLOAD_BYTES', '4194304'))
MAX_SAMPLES = 65536
CORS_ORIGINS = [value.strip() for value in os.getenv('SATMOD_CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',') if value.strip()]
VERSION = '1.0.0'
