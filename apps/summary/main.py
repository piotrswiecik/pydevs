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

def critique_summary(ai: AIServiceBase, summary: str, article: str, context: str) -> str:
    prompt = f"""
    Analyze the provided compressed version of the article critically, focusing solely on its factual accuracy, structure and comprehensiveness in relation to the given context.

    <analysis_parameters>
    PRIMARY OBJECTIVE: Compare compressed version against original content with 100% precision requirement.

    VERIFICATION PROTOCOL:
    - Each statement must match source material precisely
    - Every concept requires direct source validation
    - No interpretations or assumptions permitted
    - Markdown formatting must be exactly preserved
    - All technical information must maintain complete accuracy

    CRITICAL EVALUATION POINTS:
    1. Statement-level verification against source
    2. Technical accuracy assessment
    3. Format compliance check
    4. Link and reference validation
    5. Image placement verification
    6. Conceptual completeness check

    <original_article>${article}</original_article>

    <context desc="It may help you to understand the article better.">${context}</context>

    <compressed_version>${summary}</compressed_version>

    RESPONSE REQUIREMENTS:
    - Identify ALL deviations, regardless of scale
    - Report exact location of each discrepancy
    - Provide specific correction requirements
    - Document missing elements precisely
    - Mark any unauthorized additions

    Your task: Execute comprehensive analysis of compressed version against source material. Document every deviation. No exceptions permitted.
    """
    response = ai.text_completion(
        messages=[{"role": "system", "content": prompt}],
    )
    return response[0]["content"]


def final_summary(ai: AIServiceBase, refined_draft: str, topics: str, takeaways: str, critique: str, context: str):
    prompt = f"""
    Create a final compressed version of the article that starts with an initial concise overview, then covers all the key topics using available knowledge in a condensed manner, and concludes with essential insights and final remarks. 
        Consider the critique provided and address any issues it raises. 

    Important: Include relevant links and images from the context in markdown format. Do NOT include any links or images that are not explicitly mentioned in the context.
    Note: You're forbidden to use high-emotional language such as "revolutionary", "innovative", "powerful", "amazing", "game-changer", "breakthrough", "dive in", "delve in", "dive deeper" etc.

    Requirement: Use Polish language.

    Guidelines for compression:
    - Maintain the core message and key points of the original article
    - Always preserve original headers and subheaders
    - Ensure that images, links and videos are present in your response
    - Eliminate redundancies and non-essential details
    - Use concise language and sentence structures
    - Preserve the original article's tone and style in a condensed form

    Provide the final compressed version within <final_answer> tags.

    <refined_draft>${refined_draft}</refined_draft>
    <topics>${topics}</topics>
    <key_insights>${takeaways}</key_insights>
    <critique note="This is important, as it was created based on the initial draft of the compressed version. Consider it before you start writing the final compressed version">${critique}</critique>
    <context>${context}</context>

    Let's start.
    """
    response = ai.text_completion(
        messages=[{"role": "user", "content": prompt}],
    )
    return response[0]["content"]


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

    # note that this won't run on gemma2 - system prompts required
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

    critique = critique_summary(
        ai,
        draft_content,
        content,
        "\n\n".join([result for result in extracted_results.values()]),
    )

    _pth = Path(path).parent / f"{Path(path).stem}_critique.md"
    with open(_pth, "w") as f:
        f.write(critique)

    final_summary = final_summary(
        ai,
        draft_content,
        extracted_results["topics"],
        extracted_results["takeaways"],
        critique,
        extracted_results["context"],
    )
    result = get_result(final_summary, "final_answer")

    _pth = Path(path).parent / f"{Path(path).stem}_final_summary.md"
    with open(_pth, "w") as f:
        f.write(final_summary)

    print("DONE!")