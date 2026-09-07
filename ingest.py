import os
from pathlib import Path
import hashlib
from config import DOCUMENT_FOLDER
from ingestion.loaders import load_file
from ingestion.splitter import split_documents
from retrieval.embeddings import get_embeddings
from retrieval.vector_store import create_store, load_store

def calculate_bytes_hash(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()

def document_hash_exists(document_hash: str) -> bool:

    embeddings = get_embeddings()
    vectorstore = load_store(embeddings)
    result = vectorstore.get(
        where={"document_hash": document_hash},
        limit=1
    )
    return bool(result["ids"])

def ingest_single_pdf(file_path, document_hash):

    file_name = os.path.basename(file_path)

    # Get existing Chroma
    embeddings = get_embeddings()

    # Load PDF
    documents = load_file(Path(file_path), document_hash)
    if not documents:
        raise ValueError(f"No text found in {file_path}.")

    # Split into chunks
    chunks = split_documents(documents)

    # Add metadata
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = (
            f"{file_name}__chunk_{index}"
            )
        chunk.metadata["chunk_index"] = index

    # Add ONLY new documents chunks
    create_store(chunks, embeddings)
    
    return {
        "status": "added",
        "file_name": file_name,
        "chunk_count": len(chunks)
    }

def get_chunk_count():
    embeddings = get_embeddings()
    vectorstore = load_store(embeddings)
    return vectorstore._collection.count()

def delete_document(file_name):

    embeddings = get_embeddings()
    vectorstore = load_store(embeddings)

    # Find all chunks belonging to this document
    result = vectorstore.get(
        where={
            "document_id": file_name
        }
    )
    ids = result.get("ids", [])

    # Delete chunks from Chroma
    if ids:
        vectorstore.delete(ids=ids)

    file_path = os.path.join(DOCUMENT_FOLDER, file_name)

    if os.path.exists(file_path):
        os.remove(file_path)

    return len(ids)

def delete_all_documents():

    embeddings = get_embeddings()
    vectorstore = load_store(embeddings)
    result = vectorstore.get()
    ids = result.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)
    deleted_files = []

    if os.path.exists(DOCUMENT_FOLDER):
        for file_name in os.listdir(DOCUMENT_FOLDER):
            file_path = os.path.join(DOCUMENT_FOLDER, file_name)
            if (
                os.path.isfile(file_path)
                and file_name.lower().endswith(".pdf")
            ):
                os.remove(file_path)
                deleted_files.append(file_name)
    return {
        "documents": len(deleted_files),
        "chunks": len(ids),
        "files": deleted_files
    }

def get_indexed_document_ids():

    embeddings = get_embeddings()

    vectorstore = load_store(embeddings)

    result = vectorstore.get(include=["metadatas"])

    metadatas = result.get("metadatas", [])
    return {
        metadata.get("document_id")
        for metadata in metadatas
        if metadata
        and metadata.get("document_id")
    }


