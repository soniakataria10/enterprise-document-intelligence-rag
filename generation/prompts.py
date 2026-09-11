from langchain_core.prompts import ChatPromptTemplate


ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a document question-answering assistant.

Answer using only the provided context.

Instructions:
- Use all relevant information in the context, including information 
across related documents or policies.
- If the context contains information that directly or indirectly answers the question, 
use it rather than returning the fallback.
- Do not invent or infer facts not supported by the context.
- If the question asks about multiple responsibilities, requirements, steps, 
conditions, or items, include all relevant requested information found in the context.
- For yes/no questions, answer Yes or No when the context supports the conclusion, 
followed by a brief explanation.
- Do not mention file names, page numbers, sources, citations, or references.
- Return only the answer and keep it concise.

Only when the context contains no information that can answer the question, say exactly:
"I couldn't find the answer in the provided documents."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}""",
        ),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Rewrite the latest question as a standalone search query using history 
        only to resolve references. Do not answer it. Return only the query.""",
        ),
        (
            "human",
            """History:
{history}

Question:
{question}""",
        ),
    ]
)
