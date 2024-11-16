import os
import sys

from dotenv import load_dotenv

from pydevs.services.base import AIServiceBase
from pydevs.services.openai import OpenAIService
from pydevs.types.completion import TextCompletionConfig, TextCompletionPayload

extraction_types = [
    {
        "key": "topics",
        "description": "Main subjects covered in the article. Focus here on the headers and all specific topics discussed in the article.",
    },
    {
        "key": "entities",
        "description": "Mentioned people, places, or things mentioned in the article. Skip the links and images.",
    },
    {
        "key": "keywords",
        "description": "Key terms and phrases from the content. You can think of them as hashtags that increase searchability of the content for the reader. Example of keyword: OpenAI, Large Language Model, API, Agent, etc.",
    },
    {
        "key": "links",
        "description": "Complete list of the links and images mentioned with their 1-sentence description.",
    },
    {
        "key": "resources",
        "description": "Tools, platforms, resources mentioned in the article. Include context of how the resource can be used, what the problem it solves or any note that helps the reader to understand the context of the resource.",
    },
    {
        "key": "takeaways",
        "description": "Main points and valuable lessons learned. Focus here on the key takeaways from the article that by themselves provide value to the reader (avoid vague and general statements like 'it's really important' but provide specific examples and context). You may also present the takeaway in broader context of the article.",
    },
    {
        "key": "context",
        "description": "Background information and setting. Focus here on the general context of the article as if you were explaining it to someone who didn't read the article.",
    },
]


def extract_information(
    ai: AIServiceBase, title: str, text: str, extr_type: str, extr_des: str
):
    extraction_message = TextCompletionPayload(
        role="system",
        content=f"""
        Extract only {extr_type}:{extr_des} from user message under the content of the article titled "{title}".
        Transform the content into clear, structured, simple bullet points without formatting except links and images.
        
        Format link like so: - name: brief description with images and links if the original message contains them

        Keep full accuracy of the original message.
        """,
    )

    user_message = TextCompletionPayload(
        role="user",
        content=f"Here's the article titled '{title}' with the content to extract information from: {text}",
    )

    response = ai.text_completion(
        payload=[extraction_message, user_message],
    )
    return str(response.choices[0])


def detailed_summary():
    pass


if __name__ == "__main__":
    load_dotenv()

    path = None
    try:
        path = sys.argv[1]
    except IndexError:
        print("Usage: python -m apps.summary.main <path>")
        sys.exit(1)

    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        sys.exit(1)

    openai = OpenAIService()

    with open(path, "r") as f:
        content = f.read()
