from config import MAX_HISTORY_MESSAGES


def add_message(history, role, content):
    history.append({"role": role, "content": content})
    return history[-MAX_HISTORY_MESSAGES:]


def format_history(history):
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history[-MAX_HISTORY_MESSAGES:]
    )
