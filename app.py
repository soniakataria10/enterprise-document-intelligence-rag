import streamlit as st
import time
import re
import os
from ingest import ingest_single_pdf, get_chunk_count, delete_document, delete_all_documents, get_indexed_document_ids, calculate_bytes_hash, document_hash_exists
from config import CHROMA_PATH, DOCUMENT_FOLDER
from retrieval.embeddings import get_embeddings
from retrieval.vector_store import load_store
from retrieval.hybrid import HybridRetriever
from retrieval.reranker import Reranker
from generation.llm import get_llm
from rag.engine import RAGEngine
from conversation.query_rewriter import rewrite_query, needs_rewrite
from conversation.memory import add_message
from monitoring.logging_utils import log_event, timer

st.set_page_config(page_title="Enterprise Document Intelligence RAG", page_icon="🤖", layout="wide")

st.markdown(
    """
    <style>
        .stAppDeployButton {
            display: none;
        }
        #MainMenu {
            display: none;
        }
        header[data-testid="stHeader"] {
            background: transparent;
        }
        .block-container {
            padding-top: 1rem;
        }
        .st-emotion-cache-10p9htt{
            margin-bottom: 0rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🤖 Enterprise Document Intelligence RAG")

@st.dialog("Delete Document")
def delete_document_dialog(file_name):
    st.warning(
        f"⚠️  Are you sure you want to delete "
        f"**{file_name}** and all of its chunks from the Knowledge Base?"
    )
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Delete", use_container_width=True):
            try:
                deleted_chunks = delete_document(file_name)

                st.cache_resource.clear()

                message = st.success(f"Deleted Successfully")
                time.sleep(1)
                message.empty()

                # Close dialog and refresh app
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

@st.dialog("Delete All Documents")
def delete_all_documents_dialog():
    st.warning("⚠️ This action will permanently remove all documents and their indexed chunks")
    st.write("This action cannot be undone.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Yes, Delete All", use_container_width=True):
            try:
                result = delete_all_documents()
                st.cache_resource.clear()
                st.session_state.history = []
                st.success(f"Deleted Successfully")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Delete failed: {e}")
    with col2:
        if st.button("Cancel", use_container_width=True):
            st.rerun()

def clean_llm_response(response):

    # Remove [Source: anything]
    response = re.sub(
        r"\[Source:.*?\]",
        "",
        response,
        flags=re.IGNORECASE
    )
    # Remove Source: lines
    response = re.sub(
        r"(?im)^Source:\s*.*$",
        "",
        response
    )
    # Remove References: lines
    response = re.sub(
        r"(?im)^References?:\s*.*$",
        "",
        response
    )
    # Remove Citations: lines
    response = re.sub(
        r"(?im)^Citations?:\s*.*$",
        "",
        response
    )
    # Remove excessive blank lines
    response = re.sub(
        r"\n{3,}",
        "\n\n",
        response
    )
    return response.strip()

@st.cache_resource(
    show_spinner="Processing..."
)

def initialize():
    embeddings = get_embeddings()
    store = load_store(embeddings)

    raw = store.get(
        include=["documents", "metadatas"]
    )

    raw_documents = raw.get("documents", [])
    raw_metadatas = raw.get("metadatas", [])

    # ---------------------------------
    # Knowledge Base is empty
    # ---------------------------------

    if not raw_documents:
        return None, None
    
    from langchain_core.documents import Document

    documents = [
        Document(
            page_content=t, 
            metadata=m or {}
        )
        for t, m in zip(
            raw_documents, 
            raw_metadatas
        )
    ]
    retriever = HybridRetriever(store, documents)
    reranker = Reranker()
    llm = get_llm()
    
    engine = RAGEngine(llm, retriever, reranker)
    return engine, llm

# Initialize RAG ---------------

engine = None
llm = None
if CHROMA_PATH.exists():
    try:
        engine, llm = initialize()
        if engine is None:

            st.info(
                "📚 No PDF documents in the Knowledge Base yet. "
                "Upload a PDF to get started."
            )
    except Exception as e:
        st.error(
            f"Initialization failed: {e}"
        )
else:
    st.info(
        "📚 No PDF documents in the Knowledge Base yet. "
        "Upload a PDF to get started."
    )

# Chat History -------------

if "history" not in st.session_state:
    st.session_state.history = []


# Uploader key

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

st.markdown("""
<style>
    .document-name {
        font-size: 0.82rem;
        line-height: 1.35;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
        padding-top: 0.45rem;
    }
</style>
""", unsafe_allow_html=True)

#---------------------------
# Sidebar Knowledge Base
#---------------------------
  
with st.sidebar:
    st.caption("📚 KNOWLEDGE BASE")

    # Create folder if it doesn't exist
    os.makedirs(DOCUMENT_FOLDER, exist_ok=True)

    # Count PDF documents
    pdf_files = [
        file for file in os.listdir(DOCUMENT_FOLDER)
        if file.lower().endswith(".pdf")
    ]
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Documents:** {len(pdf_files)}")
    with col2:
        # Count chunks
        chunk_count = get_chunk_count()
        st.write(f"**Chunks:** {chunk_count}")

    #---------------------------
    # PDF(s) Uploader
    #---------------------------
   
    uploaded_files = st.file_uploader(
        "📤 Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"pdf_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_files:
        if st.button("➕ Upload File(s)"):
            progress = st.progress(0)
            status_placeholder = st.empty()
            total_files = len(uploaded_files)
            successful = 0
            failed = 0
            skipped = 0

            indexed_documents = get_indexed_document_ids()

            for index, uploaded_file in enumerate(uploaded_files):
                file_name = uploaded_file.name
                file_path = os.path.join(DOCUMENT_FOLDER, file_name)

                # check duplicate file name  
                if file_name in indexed_documents or os.path.exists(file_path):

                    skipped += 1
                    status_placeholder.warning(
                        f"⏭️ {file_name} already exists. Skipped."
                    )
                    progress.progress(
                        (index + 1)
                        / total_files
                    )
                    continue
                
                try:
                    status_placeholder.info(f"Processing {file_name}...")

                    # Get uploaded PDF bytes once
                    file_bytes = uploaded_file.getvalue()

                    # Calculate content hash before saving
                    document_hash = calculate_bytes_hash(file_bytes)

                    # Same content, different filename check
                    if document_hash_exists(document_hash):
                        skipped += 1

                        status_placeholder.warning(
                            f"⏭️ {file_name} is a duplicate document. "
                            f"Skipped."
                        )
                        progress.progress(
                            (index + 1) / total_files
                        )
                        continue

                    # Save PDF
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    result  = ingest_single_pdf(file_path, document_hash)

                    if result["status"] == "added":
                        successful += 1
                        indexed_documents.add(file_name)

                        status_placeholder.success(
                            f"✅ {file_name}"
                            f"({result['chunk_count']} chunks)"
                        )
                except Exception as e:
                    failed += 1
                    try:
                        delete_document(file_name)
                    except Exception:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                    status_placeholder.error(f"❌ {uploaded_file}: {e}")

                progress.progress((index + 1) / total_files)

            #Summary

            st.session_state.upload_summary = (
                f"Upload Complete - "
                f"Added: {successful} | "
                f"Skipped: {skipped} | "
                f"Failed: {failed}"
             )
            
            # Clear uploader
            st.session_state.uploader_key += 1

            # Rebuild retriever with newely uploaded documents
            st.cache_resource.clear()
            st.rerun()

    # Summary for uploaded files

    if "upload_summary" in st.session_state:
        st.success(
            st.session_state.pop(
                "upload_summary"
            )
        )

    #---------------------------
    # List of Documents
    #---------------------------

    st.caption("DOCUMENTS")

    if pdf_files:

        document_container = st.container(height=400)

        with document_container:
            for file_name in sorted(pdf_files):
                col1, col2 = st.columns([7.5, 1.5])
                with col1:
                    st.markdown(
                        f"""<div class="document-name" title="{file_name}">
                            📄 {file_name}
                        </div>
                        """, unsafe_allow_html=True
                    )
                with col2:
                    if st.button(
                        #"🗑️",
                        "🗑", 
                        key=f"delete_{file_name}", 
                        help=f"Delete{file_name}"
                        ):
                        delete_document_dialog(file_name)
    else:
        st.info("No PDF documents found.")

    # ---------------------------
    # Button Delete All Documents 
    # ---------------------------

    if pdf_files:
        if st.button("🗑 Delete All Documents", use_container_width=True):
            delete_all_documents_dialog()
    
    #---------------------------
    # Clear Chat button
    #---------------------------
    
    st.caption("CHAT")
    if st.button("Clear chat", use_container_width=True, disabled=(engine is None)):
        st.session_state.history = []
        st.rerun()


for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

question = st.chat_input("Ask about your documents...", disabled=(engine is None))

if question:
    if engine is None or llm is None:
        st.warning("Please upload documents to the Knowledge Base first.")
        st.stop()

    with st.chat_message("user"):
        st.write(question)

    started = timer()
    history_before = list(st.session_state.history)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            if needs_rewrite(question, history_before):
                standalone = rewrite_query(llm, question, history_before)
            else:
                standalone = question

            result = engine.answer(standalone)

            answer_according_prompt = result["answer"]
            answer = clean_llm_response(answer_according_prompt)
        st.write(answer)
        
        if result["sources"]:
            st.markdown("**Sources**")
            seen = set()
            for s in result["sources"]:
                key = (s["source"], s["page"])
                if key not in seen:
                    st.write(f"- {s['source']} — page {s['page']}")
                    seen.add(key)

    st.session_state.history = add_message(
        st.session_state.history, "user", question
    )
    st.session_state.history = add_message(
        st.session_state.history, "assistant", answer
    )

    log_event({
        "question": question,
        "rewritten_question": standalone,
        "sources": result["sources"],
        "latency_seconds": timer() - started,
    })

