"""Trusted model evaluation and validated offline dataset exports; never web training."""
import json
from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse, Response
from pydantic import Field

from backend.api.schemas import StrictModel
from backend.services import operations
from backend.services.research_evaluation import (
    ModelUnavailable, compare_evaluations, completed_evaluations, model_catalog,
)
from backend.services.training_exports import ExportCapacityReached, export_training_dataset

router = APIRouter(prefix='/api/research')


def validated_training_export(dataset_id,model,seed,allow_small):
    try:return export_training_dataset(dataset_id,model,seed,allow_small)
    except ExportCapacityReached as exc:raise HTTPException(429,str(exc),headers={'Retry-After':'1'}) from exc


class ExternalEvaluation(StrictModel):
    datasetId: str = Field(min_length=1, max_length=64)
    modelId: str = Field(default='1d-cnn', min_length=1, max_length=100)
    modelVersion: str | None = Field(default=None, max_length=100)


class EvaluationComparison(StrictModel):
    jobIds: list[str] = Field(min_length=2, max_length=8)


@router.get('/models')
def models():
    return model_catalog()


@router.post('/evaluations')
def evaluate(body: ExternalEvaluation):
    try:
        job = operations.submit('evaluation', body.model_dump())
    except ModelUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        status = 429 if 'capacity' in str(exc) else 409
        raise HTTPException(status, str(exc)) from exc
    return JSONResponse(job, status_code=202)


@router.get('/evaluations')
def evaluations(datasetId: str | None = None, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=200)):
    operations.runner()
    return completed_evaluations(datasetId, offset, limit)


@router.post('/evaluations/compare')
def compare(body: EvaluationComparison):
    if len(set(body.jobIds)) != len(body.jobIds):
        raise ValueError('Choose distinct completed evaluation jobs.')
    return compare_evaluations(body.jobIds)


@router.get('/datasets/{dataset_id}/training-manifest')
def training_manifest(dataset_id: str, model: Literal['random-forest', 'cnn', 'resnet', 'cldnn'] = 'random-forest',
                      seed: int = Query(42, ge=0, le=4294967295), allowSmall: bool = False):
    manifest, _ = validated_training_export(dataset_id, model, seed, allowSmall)
    return manifest


@router.get('/datasets/{dataset_id}/training.npz')
def training_npz(dataset_id: str, model: Literal['random-forest', 'cnn', 'resnet', 'cldnn'] = 'random-forest',
                 seed: int = Query(42, ge=0, le=4294967295), allowSmall: bool = False):
    manifest, artifact = validated_training_export(dataset_id, model, seed, allowSmall)
    return Response(artifact, media_type='application/octet-stream', headers={
        'Content-Disposition': f'attachment; filename="{manifest["artifact"]["filename"]}"',
        'X-Artifact-SHA256': manifest['artifact']['sha256'],
    })


@router.get('/datasets/{dataset_id}/experiment-config')
def experiment_config(dataset_id: str, model: Literal['random-forest', 'cnn', 'resnet', 'cldnn'] = 'random-forest',
                      seed: int = Query(42, ge=0, le=4294967295), allowSmall: bool = False):
    manifest, _ = validated_training_export(dataset_id, model, seed, allowSmall)
    return Response(json.dumps(manifest['experimentConfig'], indent=2), media_type='application/json',
                    headers={'Content-Disposition': 'attachment; filename="experiment-config.json"'})
