import pytest

from dotenv import load_dotenv
from pydevs.services.fuse import LangfuseService
from pydevs.services.vector import VectorDBService
from pydevs.services.openai import OpenAIService


@pytest.fixture
def fuse_client():
    load_dotenv()
    return LangfuseService()


@pytest.fixture
def openai_client():
    load_dotenv()
    return OpenAIService()


@pytest.fixture
def vector_db_client(openai_client):
    load_dotenv()
    return VectorDBService(embedding_provider=OpenAIService())
