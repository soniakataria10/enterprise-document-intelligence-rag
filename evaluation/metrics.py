import re

def normalize_text(text):
    if not text:
        return ""

    text = text.lower()
    text = re.sub(r"[^\w\s%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def evaluate_required_facts(actual_answer, required_facts):
    """
    Check whether every required fact appears in the generated answer.

    Returns:
        passed, score, matched_facts, missing_facts
    """
    if not required_facts:
        return None, None, [], []

    normalized_answer = normalize_text(actual_answer)

    matched_facts = []
    missing_facts = []

    for fact in required_facts:
        normalized_fact = normalize_text(fact)

        if normalized_fact in normalized_answer:
            matched_facts.append(fact)
        else:
            missing_facts.append(fact)

    score = len(matched_facts) / len(required_facts)
    passed = len(missing_facts) == 0

    return passed, round(score, 4), matched_facts, missing_facts


def evaluate_out_of_scope(answer):
    """Check whether RAG correctly refuses an unsupported question."""
    answer_lower = answer.lower()

    fallback_phrases = [
        "couldn't find",
        "could not find",
        "cannot find",
        "can't find",
        "not found",
        "not available",
        "not provided",
        "not mentioned",
        "does not contain",
        "don't have enough information",
        "do not have enough information",
        "insufficient information",
        "available documents do not",
        "provided documents do not",
        "context does not contain",
    ]

    return any(
        phrase in answer_lower
        for phrase in fallback_phrases
    )

