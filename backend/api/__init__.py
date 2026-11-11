"""Shared HTTP boundary for production-model initialization failures."""
import json
import logging
from fastapi import HTTPException


def require_production_model():
    from backend.services import classifier
    try:
        classifier.model()
        classifier.stats()
    except Exception as exc:
        logging.getLogger('satmod').exception(json.dumps({'event':'model_unavailable','error':type(exc).__name__}))
        raise HTTPException(503,'Production model or training statistics unavailable; inspect backend logs.') from exc
