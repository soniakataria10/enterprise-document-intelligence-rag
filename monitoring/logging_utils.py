import json
import time
from config import LOG_FOLDER


def log_event(event):
    LOG_FOLDER.mkdir(parents=True, exist_ok=True)
    path = LOG_FOLDER / "rag_events.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def timer():
    return time.perf_counter()
