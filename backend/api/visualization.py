"""Bounded research visualization endpoints; the main application applies access controls."""
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import Field
from backend.api.schemas import SignalRequest
from backend.services.signals import from_rows
from backend.services.time_frequency import analyze

router=APIRouter(prefix='/api/research')


class TimeFrequencyRequest(SignalRequest):
    fftSize: int=Field(default=64,strict=True,ge=16,le=1024)
    overlap: float=Field(default=.5,strict=True,ge=0,lt=1)
    window: Literal['hann','boxcar']='hann'
    sampleRateKnown: bool | None=Field(default=None,strict=True)


@router.post('/time-frequency')
def time_frequency(body: TimeFrequencyRequest):
    try:return analyze(from_rows(body.samples),body.sampleRate,body.fftSize,body.overlap,body.window,body.sampleRateKnown)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
