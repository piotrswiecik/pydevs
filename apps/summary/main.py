import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from pydevs.services.base import AIServiceBase
from pydevs.services.ollama import OllamaService
from pydevs.services.openai import OpenAIService

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
) -> str:
    extraction_message = {
        "role": "system",
        "content": f"""
        Extract only {extr_type}:{extr_des} from user message under the content of the article titled "{title}".
        Transform the content into clear, structured, simple bullet points without formatting except links and images.
        
        Format link like so: - name: brief description with images and links if the original message contains them

        Keep full accuracy of the original message.
        """,
    }

    user_message = {
        "role": "user",
        "content": f"Here's the article titled '{title}' with the content to extract information from: {text}",
    }

    response = ai.text_completion(
        messages=[user_message],
        **{"temperature": 0.6}
    )

    try:
        return response[0]["content"]
    except (KeyError, IndexError):
        logging.error("Invalid response from AI service.")


def draft_summary(ai: AIServiceBase, title: str, article: str, context: str, entities: str, links: str, topics: str, takeaways: str) -> str:
    prompt = f"""
    As a copywriter, create a standalone, fully detailed article based on "${title}" that can be understood without reading the original. Write in markdown format, incorporating all images within the content. The article must:

    Write in Polish, ensuring every crucial element from the original is included while:
    - Stay driven and motivated, ensuring you never miss the details needed to understand the article
    - NEVER reference to the original article
    - Always preserve original headers and subheaders
    - Mimic the original author's writing style, tone, expressions and voice
    - Presenting ALL main points with complete context and explanation
    - Following the original structure and flow without omitting any details
    - Including every topic, subtopic, and insight comprehensively
    - Preserving the author's writing characteristics and perspective
    - Ensuring readers can fully grasp the subject matter without prior knowledge
    - Use title: "${title}" as the title of the article you create. Follow all other headers and subheaders from the original article
    - Include cover image

    Before writing, examine the original to capture:
    * Writing style elements
    * All images, links and vimeo videos from the original article
    * Include examples, quotes and keypoints from the original article
    * Language patterns and tone
    * Rhetorical approaches
    * Argument presentation methods

    Note: You're forbidden to use high-emotional language such as "revolutionary", "innovative", "powerful", "amazing", "game-changer", "breakthrough", "dive in", "delve in", "dive deeper" etc.

    Reference and integrate ALL of the following elements in markdown format:

    <context>${context}</context>
    <entities>${entities}</entities>
    <links>${links}</links>
    <topics>${topics}</topics>
    <key_insights>${takeaways}</key_insights>

    <original_article>${article}</original_article>

    Create the new article within <final_answer> tags. The final text must stand alone as a complete work, containing all necessary information, context, and explanations from the original article. No detail should be left unexplained or assumed as prior knowledge.
    """
    response = ai.text_completion(
        messages=[{"role": "user", "content": prompt}],
    )
    return response[0]["content"]

def get_result(content: str, tag_name: str) -> Optional[str]:
    regex = re.compile(rf"<{tag_name}>(.*?)</{tag_name}>", re.S)
    match = regex.match(content)
    if match is None:
        return match
    return match.group(1)

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

    ai = OpenAIService(default_model="gpt-4o-mini")

    with open(path, "r") as f:
        content = f.read()

    # feature extraction from the document
    extracted_results = dict()
    for et in extraction_types:
        extract = extract_information(
            ai,
            "Generatywne AI w praktyce",
            content,
            et["key"],
            et["description"],
        )
        extracted_results.update({
            et["key"]: extract
        })

    # save extracted features
    for er_key, er_val in extracted_results.items():
        _pth = Path(path).parent / f"{Path(path).stem}_{er_key}.txt"
        with open(_pth, "w") as f:
            f.write(er_val)

    draft = draft_summary(
        ai,
        "Generatywne AI w praktyce",
        content,
        extracted_results["context"],
        extracted_results["entities"],
        extracted_results["links"],
        extracted_results["topics"],
        extracted_results["takeaways"],
    )

    _pth = Path(path).parent / f"{Path(path).stem}_draft.md"
    with open(_pth, "w") as f:
        f.write(draft)

    draft_content = get_result(draft, "final_answer")