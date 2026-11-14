"""Discover shared-workspace access without exposing or accepting credentials in URLs."""
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.services.deployment_readiness import access_ready

router = APIRouter()


@router.get('/api/access')
def access():
    return JSONResponse({
        'tokenRequired': bool(os.getenv('SATMOD_ACCESS_TOKEN', '')),
        'configured': access_ready(),
        'publicMode': os.getenv('SATMOD_PUBLIC') == '1',
        'sharedWorkspace': True,
    }, headers={'Cache-Control': 'no-store'})


@router.get('/api/session')
def session():
    # WorkspaceAccess authenticates this route before it reaches the handler.
    return JSONResponse({'authorized': True, 'sharedWorkspace': True},
                        headers={'Cache-Control': 'no-store'})
