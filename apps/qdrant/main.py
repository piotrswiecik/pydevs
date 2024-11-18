import os
import uuid
from dotenv import load_dotenv
import datetime
import sys

from pydevs.services.fuse import LangfuseService
from pydevs.services.openai import OpenAIService
from pydevs.services.vector import VectorDBService

PS1 = f"\U0001f916 $ "
COLLECTION_NAME = "qdrant-cli"


def create_system_prompt(context):
    return rf"""
    You are a helpful assistant.
    You are given a conversation history (context) and you will receive a user message.
    Your task is to respond to the user message based on the conversation history.
    <context>
    {context}
    </context>
    """


if __name__ == "__main__":
    load_dotenv()
    print("Welcome to Qdrant! Type to chat or /bye to exit.")

    openai = OpenAIService(api_key=os.getenv("OPENAI_API_KEY"))
    langfuse = LangfuseService(
        host=os.getenv("LANGFUSE_HOST"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    )
    qdrant = VectorDBService(
        embedding_provider=openai,
        db_url=os.getenv("VECTOR_DB_URL"),
        db_api_key=os.getenv("VECTOR_DB_API_KEY"),
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M")
    trace = langfuse.create_trace(
        id=f"qdrant-{timestamp}",
        name=f"Qdrant-CLI-{timestamp}",
        session_id=f"qdrant-{timestamp}",
        user_id="pydevs",
    )
    qdrant.create_collection(COLLECTION_NAME)

    context = []

    while True:
        user_input = input(PS1)
        if user_input == "/bye":
            break

        similar_messages = qdrant.search(
            collection_name=COLLECTION_NAME, query=user_input
        )

        if similar_messages is not None:
            print(f"--> Vector DB query found {len(similar_messages)} similar messages.")
        else:
            print("--> No similar messages found.")
            similar_messages = []

        context = [
            {"role": "user", "content": user_input},
            *[
                {"role": message.payload["role"], "content": message.payload["text"]}
                for message in similar_messages
            ],
            *context,
        ]

        response = openai.text_completion(
            messages=[
                {"role": "system", "content": create_system_prompt(context)},
                {"role": "user", "content": user_input},
            ],
            **{"model": "gpt-4o-mini"},
        )

        langfuse.create_generation(
            trace,
            name=f"user-msg-{datetime.datetime.now().strftime("%Y%m%d%H%M")}",
            input=user_input,
        )

        answer = response[0]["content"]
        print(f"Assistant: {answer}")

        qdrant.add_points(
            collection_name=COLLECTION_NAME,
            points=[{"id": str(uuid.uuid4()), "text": user_input, "role": "user"}],
        )
        qdrant.add_points(
            collection_name=COLLECTION_NAME,
            points=[{"id": str(uuid.uuid4()), "text": answer, "role": "assistant"}],
        )

    # clean up
    langfuse.finalize_trace(trace, input=None, output=None)
    print("Goodbye!")
