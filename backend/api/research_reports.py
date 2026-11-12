"""Persistent, identity-checked explanation and managed comparison workflows."""
import json
from typing import Literal
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import Field
from backend.api import require_production_model
from backend.api.schemas import StrictModel, IQSample
from backend.services import workspace_store as store, saved_explanations, managed_comparisons

router=APIRouter(prefix='/api/research')


class PersistentExplanation(StrictModel):
    method: Literal['both','integrated-gradients','saliency']='both'
    targetClass: Literal['BPSK','QPSK','16-QAM'] | None=None
    steps: int=Field(default=64,ge=16,le=256)
    samples: list[IQSample] | None=Field(default=None,min_length=128,max_length=65536)


class ManagedWindow(StrictModel):
    captureId: str
    start: int=Field(default=0,ge=0)
    length: int=Field(default=128,ge=16,le=65536)


class ManagedComparison(StrictModel):
    first: ManagedWindow
    second: ManagedWindow
    firstAnalysisId: str | None=None
    secondAnalysisId: str | None=None


@router.post('/analyses/{analysis_id}/explanations')
def explain_analysis(analysis_id: str,body: PersistentExplanation):
    require_production_model()
    try:return saved_explanations.create(analysis_id,body.method,body.targetClass,body.steps,saved_explanations.supplied_rows(body.samples))
    except saved_explanations.ExplanationConflict as exc:raise HTTPException(409,str(exc)) from exc
    except BlockingIOError as exc:raise HTTPException(429,str(exc)) from exc


@router.get('/analyses/{analysis_id}/explanations')
def stored_explanations(analysis_id: str):
    return saved_explanations.response(saved_explanations.load_analysis(analysis_id))


@router.post('/comparisons')
def compare_recordings(body: ManagedComparison):
    try:return managed_comparisons.create(body.first.model_dump(),body.second.model_dump(),body.firstAnalysisId,body.secondAnalysisId)
    except saved_explanations.ExplanationConflict as exc:raise HTTPException(409,str(exc)) from exc


@router.get('/comparisons')
def comparisons(limit: int=Query(50,ge=1,le=200),offset: int=Query(0,ge=0),search: str=Query('',max_length=200)):
    return store.listing('comparison',limit,offset,search)


@router.get('/comparisons/{record_id}')
def comparison(record_id: str):return managed_comparisons.load(record_id)


@router.get('/comparisons/{record_id}/export')
def export_comparison(record_id: str,format: Literal['json','pdf']='json'):
    report=managed_comparisons.load(record_id)
    if format=='json':data=json.dumps(report,indent=2,allow_nan=False).encode();mime='application/json'
    else:
        from backend.services.research_pdf import comparison_pdf
        data=comparison_pdf(report);mime='application/pdf'
    return Response(data,media_type=mime,headers={'Content-Disposition':f'attachment; filename="satmod-comparison.{format}"'})


@router.delete('/comparisons/{record_id}')
def delete_comparison(record_id: str):
    store.delete('comparison',record_id)
    return {'deleted':True}
