"""Bounded wire contracts. All probabilities use fractions, SNR is in dB."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class IQSample(StrictModel):
    i: float = Field(ge=-1e6, le=1e6)
    q: float = Field(ge=-1e6, le=1e6)

class SignalRequest(StrictModel):
    samples: list[IQSample] = Field(min_length=16, max_length=65536)
    sampleRate: float = Field(default=1, ge=1, le=1e9)
    source: str = Field(default='I/Q capture', max_length=200)

class ClassifyRequest(SignalRequest):
    windowStart: int = Field(default=0, ge=0, le=65408)

class GenerateRequest(StrictModel):
    modulation: Literal['BPSK','QPSK','16-QAM','8-PSK','16-APSK','32-APSK','GMSK','BFSK'] = 'BPSK'
    snrDb: float = Field(default=12, ge=-30, le=60)
    numSamples: int = Field(default=128, ge=128, le=65536)
    sps: int = Field(default=8, ge=1, le=32)
    sampleRate: float = Field(default=1, ge=1, le=1e9)
    rrcAlpha: float = Field(default=.35, gt=0, le=1)
    rrcEnabled: bool = True
    dopplerHz: float = Field(default=0, ge=-5e8, le=5e8)
    cfoHz: float = Field(default=0, ge=-5e8, le=5e8)
    phaseNoiseStd: float = Field(default=0, ge=0, le=2)
    ricianEnabled: bool = False
    ricianK: float = Field(default=0, ge=0, le=100)
    twtaDrive: float = Field(default=0, ge=0, le=4)
    amplitudeRms: float = Field(default=.008584335, gt=0, le=100)
    seed: int = Field(default=42, ge=0, le=4294967295)

class CompareRequest(StrictModel):
    first: SignalRequest
    second: SignalRequest

class BenchmarkRequest(StrictModel):
    modelId: str = '1d-cnn'
    sampleId: int = Field(default=42, ge=0, le=299)
    iterations: int = Field(default=30, ge=5, le=100)

class ClassificationResult(StrictModel):
    id: str
    createdAt: str
    source: str
    predictedModulation: str
    confidence: float
    probabilities: dict[str, float]
    entropyBits: float
    processingMs: float
    sampleRate: float
    modelId: str
    preprocessing: dict
    warnings: list[str]
    analysis: dict
    samples: list[IQSample]

class HealthStatus(StrictModel):
    connected: bool
    modelId: str
    backend: str | None = None
    error: str | None = None
    version: str
