import re
from rank_bm25 import BM25Okapi
from config import DENSE_K, BM25_K, RERANK_CANDIDATES, RRF_K


def tokenize(text):
    return re.findall(r"(?u)\b\w+\b", text.lower())

def document_key(doc):

    return (
        doc.metadata.get("document_id", ""),
        doc.metadata.get("page", ""),
        doc.metadata.get("chunk_id", ""),
        doc.page_content[:100],
    )

class HybridRetriever:
    def __init__(self, vector_store, documents):
        self.vector_store = vector_store
        self.documents = documents

        if documents:
            tokenized_documents = []
            for doc in documents:
                source = doc.metadata.get("source","")
                searchable_text = (
                    f"{source} "
                    f"{doc.page_content}"
                )
                tokenized_documents.append(tokenize(searchable_text))

            self.bm25 = BM25Okapi(tokenized_documents)
        else:
            self.bm25 = None

    def retrieve(self, query, dense_k=DENSE_K, bm25_k=BM25_K, candidate_k=RERANK_CANDIDATES, rrf_k=RRF_K):

        if not self.documents: return[]

        dense = self.vector_store.similarity_search_with_score(
            query, k=dense_k
        )
        dense_docs = []
        for doc, score in dense:
            doc.metadata = {
                **doc.metadata, 
                "dense_distance": float(score)
                }
            dense_docs.append(doc)

        bm25_docs = []
        if self.bm25 is not None:
            scores = self.bm25.get_scores(tokenize(query))

            ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

            for idx, score in ranked[:bm25_k]:
                doc = self.documents[idx]
                doc.metadata = {
                    **doc.metadata, 
                    "bm25_score": float(score)
                    }
                bm25_docs.append(doc)

        fused = {}

        def add_results(documents, retrieval_type):

            for rank, doc in enumerate(documents, start=1):

                key = document_key(doc)
                if key not in fused:

                    fused[key] = {
                        "doc": doc,
                        "score": 0.0,
                    }

                fused[key]["score"] += (
                    1.0 /
                    (rrf_k + rank)
                )

                doc.metadata[
                    f"{retrieval_type}_rank"
                ] = rank

        add_results(dense_docs, "dense")

        add_results(bm25_docs, "bm25")

        # Sort by RRF score
        ranked_fused = sorted(
            fused.values(),
            key=lambda item: item["score"],
            reverse=True,
        )
        results = []

        for item in ranked_fused[:candidate_k]:

            doc = item["doc"]
            doc.metadata["rrf_score"] = (
                item["score"]
            )
            results.append(doc)

        return results
    