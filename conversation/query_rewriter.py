from generation.prompts import REWRITE_PROMPT
from langchain_core.messages import HumanMessage, SystemMessage


def rewrite_query(llm, question, history):
    if not history:
        return question

    history_text = "\n".join(
        f"{message['role']}: "
        f"{message['content']}"
        for message in history[-6:]
    )
    messages = REWRITE_PROMPT.format_messages(
            history=history_text,
            question=question,
        )

    response = llm.invoke(messages)

    rewritten = response.content.strip()
    rewritten = rewritten.strip('"').strip("'")

    return rewritten or question


def needs_rewrite(question, history):
    if not history:
        return False

    question_lower = question.lower().strip()

    follow_up_phrases = (
        "what about",
        "how about",
        "and what",
        "and how",
        "what else",
        "does it",
        "do they",
        "is it",
        "are they",
        "can it",
        "can they",
        "why is that",
        "what does that",
        "what is that",
    )

    if question_lower.startswith(follow_up_phrases):
        return True

    reference_words = {
        "it",
        "its",
        "they",
        "them",
        "their",
        "this",
        "that",
        "these",
        "those",
        "he",
        "she",
    }

    words = {
        word.strip(".,?!:;").lower()
        for word in question.split()
    }

    if words & reference_words:
        return True

    return False