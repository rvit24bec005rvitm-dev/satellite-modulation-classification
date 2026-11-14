"""Anonymous readiness status; never returns credentials, paths or exceptions."""
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from backend.services import deployment_readiness

router = APIRouter()


@router.get('/api/ready')
def ready():
    result = deployment_readiness.cached_report()
    return JSONResponse(result, status_code=200 if result['ready'] else 503,
                        headers={'Cache-Control': 'no-store'})
