import logging
from dotenv import load_dotenv

from pydevs.services.openai import OpenAIService

load_dotenv()


def test_openai_extract_information():
    service = OpenAIService()

    text = """
    Tom, James, Kate live in New York. Tom is a software engineer. James is a doctor. Kate is a teacher.
    Tom likes to use Python. James likes to use Java. Kate likes to use JavaScript.
    Kate uses google.com and facebook.com. Tom uses github.com. James uses youtube.com.
    """

    from apps.summary.main import extract_information

    summary = extract_information(
        service, "Test", text, "entities", "Mentioned people, places, or things mentioned in the article. Skip the links and images.")

    logging.info(summary)
    # TODO: assertion
    
    