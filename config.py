from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_FOLDER = BASE_DIR / "data" / "documents"
CHROMA_PATH = BASE_DIR / "chroma_db"
LOG_FOLDER = BASE_DIR / "logs"
EVAL_FOLDER = BASE_DIR / "evaluation_data"

COLLECTION_NAME = "enterprise_rag"

LLM_MODEL = "llama3.2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

DENSE_K = 10
BM25_K = 10
RERANK_K = 6
RERANK_CANDIDATES = 15
RRF_K = 60

MAX_HISTORY_MESSAGES = 8

MIN_RERANK_SCORE = -10.0
RERANK_SCORE_GAP = 3.0
CONTEXT_MAX_CHUNKS = 3
SOURCE_MAX_COUNT = 3
