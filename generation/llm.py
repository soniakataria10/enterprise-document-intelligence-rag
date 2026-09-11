from langchain_ollama import ChatOllama
from config import LLM_MODEL


def get_llm():
    return ChatOllama(
        model=LLM_MODEL,
        temperature=0,
        num_predict=180,
        num_ctx=2048,
    )
