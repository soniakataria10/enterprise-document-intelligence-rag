from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL, RERANK_K


class Reranker:
    def __init__(self):
        self.model = CrossEncoder(RERANKER_MODEL)

    def rerank(self, query, documents, top_k=RERANK_K):
        if not documents:
            return []

        pairs = []
        for doc in documents:
            source = doc.metadata.get("source", "unknown")

            document_text = f"Document: {source}\nContent: {doc.page_content}"
            pairs.append((query, document_text))

        scores = self.model.predict(pairs)
        ranked = sorted(
            zip(documents, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        results = []
        for doc, score in ranked[:top_k]:
            doc.metadata = {**doc.metadata, "rerank_score": float(score)}
            results.append(doc)
        return results
