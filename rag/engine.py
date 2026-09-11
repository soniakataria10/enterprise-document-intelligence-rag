from generation.prompts import ANSWER_PROMPT
from config import MIN_RERANK_SCORE, SOURCE_MAX_COUNT, CONTEXT_MAX_CHUNKS, RERANK_SCORE_GAP


def format_context(docs):
    parts = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown")

        page = doc.metadata.get("page")

        if isinstance(page, int):
            page += 1
        else:
            page = "N/A"

        parts.append(f"Document: {source}\nPage: {page}\nContent:\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


def build_sources(docs, max_sources):

    if not docs:
        return []

    sources = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")

        if isinstance(page, int):
            page += 1
        else:
            page = "N/A"

        score = doc.metadata.get("rerank_score")

        key = (source, page)
        if key in seen:
            continue

        seen.add(key)

        sources.append(
            {
                "source": source,
                "page": page,
                "score": score,
            }
        )
        if len(sources) >= max_sources:
            break

    return sources


def filter_relevant_docs(docs, max_score_gap, max_docs):

    if not docs:
        return []

    best_score = docs[0].metadata.get("rerank_score")

    if best_score is None:
        return docs[:max_docs]

    relevant = []

    for doc in docs:
        score = doc.metadata.get("rerank_score")

        if score is None:
            continue

        # COMPARE DIFFERENCE BETWEEN BEST RESULTS
        if best_score - score > max_score_gap:
            continue

        relevant.append(doc)

        if len(relevant) >= max_docs:
            break

    return relevant


class RAGEngine:
    def __init__(self, llm, retriever, reranker):
        self.llm = llm
        self.retriever = retriever
        self.reranker = reranker

    def answer(self, question):

        candidates = self.retriever.retrieve(question)

        if not candidates:
            return {
                "answer": "I couldn't find the answer in the provided documents.",
                "sources": [],
                "retrieved": [],
                "reranked": [],
            }

        reranked = self.reranker.rerank(question, candidates)

        if not reranked:
            return {
                "answer": "I couldn't find the answer in the provided documents.",
                "sources": [],
                "retrieved": [],
                "reranked": [],
            }

        best_score = reranked[0].metadata.get("rerank_score", float("-inf"))

        if best_score < MIN_RERANK_SCORE:
            return {
                "answer": "I couldn't find the answer in the provided documents.",
                "sources": [],
                "retrieved": [],
                "reranked": reranked,
            }

        relevant_docs = filter_relevant_docs(
            reranked,
            max_score_gap=RERANK_SCORE_GAP,
            max_docs=CONTEXT_MAX_CHUNKS,
        )

        if not relevant_docs:
            return {
                "answer": "I couldn't find the answer in the provided documents.",
                "sources": [],
                "retrieved": [],
                "reranked": reranked,
            }

        context = format_context(relevant_docs)

        messages = ANSWER_PROMPT.format_messages(
            context=context,
            question=question,
        )

        response = self.llm.invoke(messages)

        answer = response.content.strip()

        # 6. Handle LLM fallback
        fallback = "I couldn't find the answer in the provided documents."

        if fallback.lower() in answer.lower():
            return {"answer": fallback, "sources": [], "retrieved": [], "reranked": reranked}

        sources = build_sources(relevant_docs, max_sources=SOURCE_MAX_COUNT)

        return {
            "answer": response.content,
            "sources": sources,
            "retrieved": relevant_docs,
            "reranked": reranked,
        }
