from langchain_chroma import Chroma
from config import CHROMA_PATH, COLLECTION_NAME 

def create_store(documents, embeddings):
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )
    if documents:
        store.add_documents(documents)
    return store

def load_store(embeddings):
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )

def reset_store():
    # Import lazily so the app can start without an initialized DB.
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
