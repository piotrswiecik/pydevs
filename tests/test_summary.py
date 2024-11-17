import logging

from dotenv import load_dotenv

from pydevs.services.ollama import OllamaService
from pydevs.services.openai import OpenAIService

load_dotenv()


def test_openai_extract_information():
    service = OpenAIService(default_model="gpt-4o-mini")

    text = """
    Tom, James, Kate live in New York. Tom is a software engineer. James is a doctor. Kate is a teacher.
    Tom likes to use Python. James likes to use Java. Kate likes to use JavaScript.
    Kate uses google.com and facebook.com. Tom uses github.com. James uses youtube.com.
    """

    from apps.summary.main import extract_information

    summary = extract_information(
        service,
        "Test",
        text,
        "entities",
        "Mentioned people, places, or things mentioned in the article. Skip the links and images.",
    )

    assert isinstance(summary, str)


def test_gemma_extract_information():
    service = OllamaService(default_model="gemma2:2b")

    text = """
    Tom, James, Kate live in New York. Tom is a software engineer. James is a doctor. Kate is a teacher.
    Tom likes to use Python. James likes to use Java. Kate likes to use JavaScript.
    Kate uses google.com and facebook.com. Tom uses github.com. James uses youtube.com.
    """

    from apps.summary.main import extract_information

    summary = extract_information(
        service,
        "Test",
        text,
        "entities",
        "Mentioned people, places, or things mentioned in the article. Skip the links and images.",
    )

    assert isinstance(summary, str)
    logging.info(summary)


def test_get_result():
    from apps.summary.main import get_result

    content = "<tag>content</tag>"
    tag_name = "tag"
    result = get_result(content, tag_name)
    assert result == "content"
    logging.info(result)


def test_get_result_not_found():
    from apps.summary.main import get_result

    content = "<tag>content</tag>"
    tag_name = "not_found"
    result = get_result(content, tag_name)
    assert result is None
    logging.info(result)