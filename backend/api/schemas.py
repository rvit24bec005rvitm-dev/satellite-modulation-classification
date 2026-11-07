"""Bounded wire contracts. All probabilities use fractions, SNR is in dB."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class IQSample(StrictModel):
    i: float = Field(strict=True, ge=-1e6, le=1e6)
    q: float = Field(strict=True, ge=-1e6, le=1e6)

class SignalRequest(StrictModel):
    samples: list[IQSample] = Field(min_length=16, max_length=65536)
    sampleRate: float = Field(default=1, ge=1, le=1e9)
    source: str = Field(default='I/Q capture', max_length=200)

class ClassifyRequest(SignalRequest):
    windowStart: int = Field(default=0, ge=0, le=65408)

class GenerateRequest(StrictModel):
    modulation: Literal['BPSK','QPSK','16-QAM','32-QAM','8-PSK','16-APSK','32-APSK','GMSK','BFSK'] = 'BPSK'
    snrDb: float = Field(default=12, ge=-30, le=60)
    numSamples: int = Field(default=128, ge=128, le=65536)
    sps: int = Field(default=8, ge=1, le=32)
    sampleRate: float = Field(default=1, ge=1, le=1e9)
    gaussianBt: float = Field(default=.3, ge=.05, le=1)
    bfskDeviationHz: float | None = Field(default=None, gt=0, le=5e8)
    continuousPhase: bool = True
    apskRadii: list[float] | None = Field(default=None, min_length=2, max_length=3)
    apskOffsets: list[float] | None = Field(default=None, min_length=2, max_length=3)
    phaseNoiseMode: Literal['iid', 'wiener'] = 'iid'
    fadingBlockSamples: int = Field(default=1, ge=1, le=65536)
    normalizeOutput: bool = True
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

class ExplainRequest(ClassifyRequest):
    targetClass: Literal['BPSK', 'QPSK', '16-QAM'] | None = None
    method: Literal['integrated-gradients', 'saliency'] = 'integrated-gradients'
    steps: int = Field(default=64, ge=16, le=256)

class CompareRequest(StrictModel):
    first: SignalRequest
    second: SignalRequest

class BenchmarkRequest(StrictModel):
    modelId: str = '1d-cnn'
    sampleId: int = Field(default=42, ge=0, le=299)
    iterations: int = Field(default=30, ge=5, le=100)

class InputValidity(StrictModel):
    status: Literal['valid']
    checks: list[str]
    channelOrder: str

class CalibrationStatus(StrictModel):
    status: Literal['not-calibrated']
    temperature: float
    reason: str

class OODStatus(StrictModel):
    status: Literal['not-validated']
    detected: bool | None
    reason: str

class TrustStatus(StrictModel):
    inputValidity: InputValidity
    calibration: CalibrationStatus
    ood: OODStatus
    interpretation: str

class AttributionSample(StrictModel):
    sample: int
    i: float
    q: float
    importance: float
    rawI: float
    rawQ: float

class ExplanationResult(StrictModel):
    method: Literal['integrated-gradients','saliency']
    targetClass: Literal['BPSK','QPSK','16-QAM']
    targetQuantity: str
    modelId: str
    windowStart: int
    windowLength: int
    integrationSteps: int
    baseline: dict
    inputLogit: float
    baselineLogit: float
    logitDifference: float
    attributionSum: float | None
    completenessError: float | None
    processingMs: float
    samples: list[AttributionSample] = Field(min_length=128,max_length=128)
    limitations: list[str]

class ClassificationResult(StrictModel):
    id: str
    createdAt: str
    source: str
    predictedModulation: Literal['BPSK','QPSK','16-QAM']
    confidence: float = Field(ge=0,le=1)
    probabilities: dict[str, float]
    entropyBits: float
    processingMs: float
    preprocessingMs: float
    trust: TrustStatus
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
