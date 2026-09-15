import os
from langchain_ollama import ChatOllama
from config import LLM_MODEL


def get_llm():
    ollama_base_url = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )
    return ChatOllama(
        model=LLM_MODEL,
        base_url=ollama_base_url,
        temperature=0,
        num_predict=180,
        num_ctx=2048,
    )
