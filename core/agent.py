import os
from dotenv import load_dotenv
from agents import Agent, ModelSettings
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

load_dotenv()

def build_agent() -> Agent:
    client = AsyncOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        default_headers={
            "HTTP-Referer": "https://rag-assistant.local",
            "X-Title": "RAG Assistant",
        }
    )
    model = OpenAIChatCompletionsModel(
        model=os.getenv("MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        openai_client=client,
    )
    return Agent(
        name="RAGAssistant",
        instructions="""You are a helpful RAG assistant.
The user's question and relevant document context will be provided to you together.
Answer the question based on the provided context.
Always cite the source document name when answering from documents.
Be concise and accurate.""",
        tools=[],
        model=model,
        model_settings=ModelSettings(max_tokens=1024),
    )
