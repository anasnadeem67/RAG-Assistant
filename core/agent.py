import os
from dotenv import load_dotenv
from agents import Agent, ModelSettings
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI
from core.tools import upload_document, add_text_directly, search_documents, list_indexed_documents, clear_vector_store

load_dotenv()

def build_agent() -> Agent:
    client = AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )
    model = OpenAIChatCompletionsModel(
        model=os.getenv("MODEL"),
        openai_client=client,
    )
    return Agent(
        name="RAGAssistant",
        instructions="""You are a RAG assistant.
1. User uploads a document → call upload_document or add_text_directly.
2. User asks a question → ALWAYS call search_documents first, then answer.
3. Cite source: "Based on [filename]..."
4. Keep answers concise.""",
        tools=[upload_document, add_text_directly, search_documents, list_indexed_documents, clear_vector_store],
        model=model,
        model_settings=ModelSettings(max_tokens=1024),
    )
