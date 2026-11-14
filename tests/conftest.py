"""Never let a test's default managed workspace resolve to an operator's data."""
import os

import pytest


@pytest.fixture(scope='session', autouse=True)
def isolated_default_workspace(tmp_path_factory):
    from backend.services import workspace_store

    original = os.environ.get('SATMOD_DATA_DIR')
    os.environ['SATMOD_DATA_DIR'] = str(tmp_path_factory.mktemp('default-workspace'))
    workspace_store.close_connections()
    try:
        yield
    finally:
        workspace_store.close_connections()
        if original is None:
            os.environ.pop('SATMOD_DATA_DIR', None)
        else:
            os.environ['SATMOD_DATA_DIR'] = original
