import json
import math
import re
import time

from config import EVAL_FOLDER
from retrieval.embeddings import get_embeddings
from retrieval.vector_store import load_store
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker
from generation.llm import get_llm
from rag.engine import RAGEngine
from langchain_core.documents import Document


# Semantic similarity is diagnostic only.
ANSWER_SIMILARITY_THRESHOLD = 0.75


def cosine_similarity(vector1, vector2):
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a * a for a in vector1))
    magnitude2 = math.sqrt(sum(b * b for b in vector2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def evaluate_answer_similarity(actual_answer, expected_answer, embeddings):
    """Return semantic similarity score only. This is not final PASS/FAIL."""
    if not expected_answer:
        return None

    expected_vector = embeddings.embed_query(expected_answer)
    actual_vector = embeddings.embed_query(actual_answer)

    similarity = cosine_similarity(expected_vector, actual_vector)
    return round(similarity, 4)


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


def evaluate_with_llm_judge(llm, question, expected_answer, actual_answer):
    """
    Use the existing local Ollama LLM as a correctness judge.
    Accept paraphrases, but reject contradictions, missing important facts,
    incorrect numbers, and incorrect negation.
    """
    if not expected_answer:
        return None

    prompt = f"""
You are evaluating the answer produced by a Retrieval-Augmented
Generation (RAG) system.

Decide whether the ACTUAL ANSWER correctly answers the QUESTION
according to the EXPECTED ANSWER.

Rules:
- Accept correct paraphrases.
- Do not require identical wording.
- Important facts, numbers, names, policy requirements and negations
  must be correct.
- If important information is missing, return FAIL.
- If the actual answer contradicts the expected answer, return FAIL.
- Extra information is acceptable only when it does not change or
  contradict the expected meaning.
- A refusal such as "I couldn't find the answer" must FAIL when an
  expected answer is provided.

QUESTION:
{question}

EXPECTED ANSWER:
{expected_answer}

ACTUAL ANSWER:
{actual_answer}

Return exactly one word:
PASS
or
FAIL
"""

    try:
        response = llm.invoke(prompt)

        judge_text = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        judge_text = judge_text.strip().upper()

        if judge_text.startswith("PASS"):
            return True

        if judge_text.startswith("FAIL"):
            return False

        print(f"WARNING: Unexpected LLM judge response: {judge_text}")
        return None

    except Exception as e:
        print(f"WARNING: LLM judge failed: {e}")
        return None


def calculate_answer_pass(required_facts_passed, llm_judge_passed, answer_similarity):
    """
    Final answer decision:
    - Prefer LLM judge.
    - If required facts exist, they must also pass.
    - Similarity is only a fallback if the judge cannot run.
    """

    if required_facts_passed is True:
        return True

    if required_facts_passed is False:
        return False

    if llm_judge_passed is not None:
        return llm_judge_passed

    if answer_similarity is not None:
        return (
            answer_similarity
            >= ANSWER_SIMILARITY_THRESHOLD
        )

    return False


def extract_sources(result):
    """Extract unique source filenames from result['sources'].""" 
    sources = set()

    for source in result.get("sources", []):
        if isinstance(source, str):
            sources.add(source)

        elif isinstance(source, dict):
            filename = (
                source.get("source")
                or source.get("document_id")
                or source.get("file")
                or source.get("filename")
            )

            if filename:
                sources.add(str(filename))

        elif hasattr(source, "metadata"):
            filename = (
                source.metadata.get("source")
                or source.metadata.get("document_id")
            )

            if filename:
                sources.add(str(filename))

    return sorted(sources)


def calculate_source_recall(actual_sources, expected_sources):
    """Of all expected sources, how many were returned?"""
    if not expected_sources:
        return None

    expected = set(expected_sources)
    actual = set(actual_sources)

    return round(len(expected & actual) / len(expected), 4)


def calculate_source_precision(actual_sources, expected_sources):
    """Of all returned sources, how many were expected?"""
    if not actual_sources:
        return 0.0

    expected = set(expected_sources)
    actual = set(actual_sources)

    return round(len(expected & actual) / len(actual), 4)


def extract_reranked_candidates(result):
    """
    Reads optional result['reranked'] returned by engine.answer().
    If engine.py does not return it yet, this safely returns [].
    """
    candidates = []

    for doc in result.get("reranked", []):
        if not hasattr(doc, "metadata"):
            continue

        page = doc.metadata.get("page")

        if isinstance(page, int):
            display_page = page + 1
        else:
            display_page = "N/A"

        candidates.append({
            "source": doc.metadata.get("source", "unknown"),
            "page": display_page,
            "rerank_score": doc.metadata.get("rerank_score"),
            "rrf_score": doc.metadata.get("rrf_score"),
            "dense_rank": doc.metadata.get("dense_rank"),
            "bm25_rank": doc.metadata.get("bm25_rank"),
        })

    return candidates


def calculate_retrieval_recall(reranked_candidates, expected_sources):
    """
    Checks whether expected source documents appeared in the reranked list.
    Helps separate retrieval failures from answer-generation failures.
    """
    if not expected_sources:
        return None

    retrieved_sources = {
        item.get("source")
        for item in reranked_candidates
        if item.get("source")
    }

    expected = set(expected_sources)

    return round(
        len(expected & retrieved_sources) / len(expected),
        4,
    )


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


def main():
    dataset_path = EVAL_FOLDER / "questions.json"

    if not dataset_path.exists():
        raise SystemExit(
            "Create evaluation_data/questions.json first."
        )

    data = json.loads(
        dataset_path.read_text(encoding="utf-8")
    )

    print("Initializing RAG system...")

    embeddings = get_embeddings()
    store = load_store(embeddings)
    raw = store.get(include=["documents", "metadatas"])

    documents = [
        Document(
            page_content=text,
            metadata=metadata or {},
        )
        for text, metadata in zip(
            raw.get("documents", []),
            raw.get("metadatas", []),
        )
    ]

    if not documents:
        raise SystemExit(
            "No indexed chunks were found. "
            "Upload and ingest documents first."
        )

    retriever = HybridRetriever(store, documents)
    reranker = Reranker()

    # One local LLM instance is reused for answer generation and evaluation.
    llm = get_llm()

    engine = RAGEngine(
        llm,
        retriever,
        reranker,
    )

    print(f"Loaded {len(documents)} chunks.")
    print(f"Running {len(data)} evaluation questions...")

    results = []

    answer_tests = 0
    answer_passes = 0

    source_tests = 0
    source_passes = 0

    source_recall_scores = []
    source_precision_scores = []
    retrieval_recall_scores = []

    out_of_scope_tests = 0
    out_of_scope_passes = 0

    out_of_scope_clean_source_tests = 0
    out_of_scope_clean_source_passes = 0

    total_latency = 0.0

    for item in data:
        question_id = item.get("id", "")
        question = item["question"]
        expected_answer = item.get("expected_answer")
        expected_sources = item.get("expected_sources", [])
        required_facts = item.get("required_facts", [])
        category = item.get("category", "unknown")

        print("\n" + "=" * 70)
        print(f"{question_id} | {category}")
        print(f"QUESTION: {question}")

        start_time = time.perf_counter()
        result = engine.answer(question)
        latency = time.perf_counter() - start_time
        total_latency += latency

        actual_answer = result.get("answer", "")
        actual_sources = extract_sources(result)
        reranked_candidates = extract_reranked_candidates(result)

        answer_similarity = None
        required_facts_passed = None
        required_facts_score = None
        matched_facts = []
        missing_facts = []
        llm_judge_passed = None
        answer_passed = None

        source_recall = None
        source_precision = None
        source_passed = None
        retrieval_recall = None

        out_of_scope_passed = None
        out_of_scope_sources_clean = None

        if category != "out_of_scope":
            answer_tests += 1

            answer_similarity = evaluate_answer_similarity(
                actual_answer,
                expected_answer,
                embeddings,
            )

            (
                required_facts_passed,
                required_facts_score,
                matched_facts,
                missing_facts,
            ) = evaluate_required_facts(
                actual_answer,
                required_facts,
            )

            llm_judge_passed = evaluate_with_llm_judge(
                llm,
                question,
                expected_answer,
                actual_answer,
            )

            answer_passed = calculate_answer_pass(
                required_facts_passed,
                llm_judge_passed,
                answer_similarity,
            )

            if answer_passed:
                answer_passes += 1

            if expected_sources:
                source_tests += 1

                source_recall = calculate_source_recall(
                    actual_sources,
                    expected_sources,
                )

                source_precision = calculate_source_precision(
                    actual_sources,
                    expected_sources,
                )

                source_passed = source_recall == 1.0

                if source_passed:
                    source_passes += 1

                source_recall_scores.append(source_recall)
                source_precision_scores.append(source_precision)

                retrieval_recall = calculate_retrieval_recall(
                    reranked_candidates,
                    expected_sources,
                )

                if retrieval_recall is not None:
                    retrieval_recall_scores.append(
                        retrieval_recall
                    )

        else:
            out_of_scope_tests += 1

            out_of_scope_passed = evaluate_out_of_scope(
                actual_answer
            )

            if out_of_scope_passed:
                out_of_scope_passes += 1

            out_of_scope_clean_source_tests += 1

            out_of_scope_sources_clean = (
                len(actual_sources) == 0
            )

            if out_of_scope_sources_clean:
                out_of_scope_clean_source_passes += 1

        if category == "out_of_scope":
            overall_pass = (
                bool(out_of_scope_passed)
                and bool(out_of_scope_sources_clean)
            )
        else:
            checks = []

            if answer_passed is not None:
                checks.append(answer_passed)

            if source_passed is not None:
                checks.append(source_passed)

            overall_pass = all(checks) if checks else False

        evaluation_result = {
            "id": question_id,
            "question": question,
            "category": category,
            "expected_answer": expected_answer,
            "actual_answer": actual_answer,
            "required_facts": required_facts,
            "matched_facts": matched_facts,
            "missing_facts": missing_facts,
            "required_facts_score": required_facts_score,
            "required_facts_passed": required_facts_passed,
            "answer_similarity": answer_similarity,
            "llm_judge_passed": llm_judge_passed,
            "answer_passed": answer_passed,
            "expected_sources": expected_sources,
            "actual_sources": actual_sources,
            "source_recall": source_recall,
            "source_precision": source_precision,
            "source_passed": source_passed,
            "retrieval_recall": retrieval_recall,
            "reranked_candidates": reranked_candidates,
            "out_of_scope_passed": out_of_scope_passed,
            "out_of_scope_sources_clean": out_of_scope_sources_clean,
            "response_time_seconds": round(latency, 2),
            "overall_pass": overall_pass,
        }

        results.append(evaluation_result)

        print(f"\nANSWER:\n{actual_answer}")

        if expected_answer:
            print(f"\nEXPECTED:\n{expected_answer}")
            print("\nSemantic similarity:", answer_similarity)

            if required_facts:
                print(
                    "Required facts score:",
                    required_facts_score,
                )
                print(
                    "Required facts:",
                    "PASS"
                    if required_facts_passed
                    else "FAIL",
                )

                if missing_facts:
                    print("Missing facts:", missing_facts)

            print(
                "LLM judge:",
                "PASS"
                if llm_judge_passed
                else "FAIL",
            )

            print(
                "Answer:",
                "PASS"
                if answer_passed
                else "FAIL",
            )

        if expected_sources:
            print(
                "\nExpected sources:",
                expected_sources,
            )
            print(
                "Returned sources:",
                actual_sources,
            )
            print(
                "Source recall:",
                source_recall,
            )
            print(
                "Source precision:",
                source_precision,
            )
            print(
                "Retrieval recall:",
                retrieval_recall,
            )
            print(
                "Source:",
                "PASS"
                if source_passed
                else "FAIL",
            )

        if category == "out_of_scope":
            print(
                "\nOut-of-scope handling:",
                "PASS"
                if out_of_scope_passed
                else "FAIL",
            )
            print(
                "No sources returned:",
                "PASS"
                if out_of_scope_sources_clean
                else "FAIL",
            )

        print(
            f"\nResponse time: {latency:.2f}s"
        )
        print(
            "OVERALL:",
            "PASS"
            if overall_pass
            else "FAIL",
        )

    answer_accuracy = (
        answer_passes / answer_tests * 100
        if answer_tests
        else 0
    )

    source_accuracy = (
        source_passes / source_tests * 100
        if source_tests
        else 0
    )

    average_source_recall = (
        sum(source_recall_scores)
        / len(source_recall_scores)
        * 100
        if source_recall_scores
        else 0
    )

    average_source_precision = (
        sum(source_precision_scores)
        / len(source_precision_scores)
        * 100
        if source_precision_scores
        else 0
    )

    average_retrieval_recall = (
        sum(retrieval_recall_scores)
        / len(retrieval_recall_scores)
        * 100
        if retrieval_recall_scores
        else 0
    )

    out_of_scope_accuracy = (
        out_of_scope_passes
        / out_of_scope_tests
        * 100
        if out_of_scope_tests
        else 0
    )

    out_of_scope_source_cleanliness = (
        out_of_scope_clean_source_passes
        / out_of_scope_clean_source_tests
        * 100
        if out_of_scope_clean_source_tests
        else 0
    )

    overall_passes = sum(
        1
        for result in results
        if result["overall_pass"]
    )

    overall_accuracy = (
        overall_passes / len(results) * 100
        if results
        else 0
    )

    average_latency = (
        total_latency / len(results)
        if results
        else 0
    )

    summary = {
        "total_questions": len(results),
        "passed_questions": overall_passes,
        "failed_questions": len(results) - overall_passes,
        "overall_accuracy": round(overall_accuracy, 2),
        "answer_accuracy": round(answer_accuracy, 2),
        "source_accuracy": round(source_accuracy, 2),
        "average_source_recall": round(
            average_source_recall,
            2,
        ),
        "average_source_precision": round(
            average_source_precision,
            2,
        ),
        "average_retrieval_recall": round(
            average_retrieval_recall,
            2,
        ),
        "out_of_scope_accuracy": round(
            out_of_scope_accuracy,
            2,
        ),
        "out_of_scope_source_cleanliness": round(
            out_of_scope_source_cleanliness,
            2,
        ),
        "average_response_time_seconds": round(
            average_latency,
            2,
        ),
    }

    output = {
        "summary": summary,
        "results": results,
    }

    out = EVAL_FOLDER / "results.json"

    out.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n")
    print("=" * 70)
    print("             RAG EVALUATION REPORT")
    print("=" * 70)

    print(f"Total Questions:              {len(results)}")
    print(f"Passed:                       {overall_passes}")
    print(
        f"Failed:                       "
        f"{len(results) - overall_passes}"
    )
    print(f"Overall Accuracy:             {overall_accuracy:.2f}%")

    print("-" * 70)

    print(f"Answer Accuracy:              {answer_accuracy:.2f}%")
    print(f"Source Accuracy:              {source_accuracy:.2f}%")
    print(
        f"Average Source Recall:        "
        f"{average_source_recall:.2f}%"
    )
    print(
        f"Average Source Precision:     "
        f"{average_source_precision:.2f}%"
    )
    print(
        f"Average Retrieval Recall:     "
        f"{average_retrieval_recall:.2f}%"
    )
    print(
        f"Out-of-Scope Accuracy:        "
        f"{out_of_scope_accuracy:.2f}%"
    )
    print(
        f"Out-of-Scope Source Clean:    "
        f"{out_of_scope_source_cleanliness:.2f}%"
    )
    print(
        f"Average Response Time:        "
        f"{average_latency:.2f}s"
    )

    print("=" * 70)

    failed_results = [
        result
        for result in results
        if not result["overall_pass"]
    ]

    if failed_results:
        print("\nFAILED QUESTIONS")
        print("-" * 70)

        for result in failed_results:
            print(
                f"\n{result['id']}: "
                f"{result['question']}"
            )

            if result["answer_passed"] is False:
                if result["required_facts_passed"] is False:
                    print(
                        "Reason: Required facts were missing."
                    )
                    print(
                        "Missing facts:",
                        result["missing_facts"],
                    )

                if result["llm_judge_passed"] is False:
                    print(
                        "Reason: Local LLM judge marked "
                        "the answer incorrect."
                    )

                print(
                    "Semantic similarity (diagnostic):",
                    result["answer_similarity"],
                )

            if result["source_passed"] is False:
                print(
                    "Reason: Not all expected sources "
                    "were returned."
                )
                print(
                    "Source recall:",
                    result["source_recall"],
                )
                print(
                    "Source precision:",
                    result["source_precision"],
                )

            if result["out_of_scope_passed"] is False:
                print(
                    "Reason: RAG did not correctly reject "
                    "the out-of-scope question."
                )

            if result["out_of_scope_sources_clean"] is False:
                print(
                    "Reason: Fallback answer still "
                    "returned sources."
                )

            if result["retrieval_recall"] is not None:
                print(
                    "Retrieval recall:",
                    result["retrieval_recall"],
                )

    print(
        f"\nDetailed results saved to:\n{out}"
    )


if __name__ == "__main__":
    main()
