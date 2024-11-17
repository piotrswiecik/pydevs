import pytest

from pydevs.services.fuse import LangfuseService


@pytest.fixture
def fuse_client():
    return LangfuseService()