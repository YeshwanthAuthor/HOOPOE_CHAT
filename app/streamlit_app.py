import base64
import copy
import hashlib
import sys
import uuid
from pathlib import Path

import streamlit as st


# Make project modules importable when Streamlit runs this file from the app folder.
PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

PAGE_TITLE = "Hoopoe | RAG Chat"
APP_ICON = PROJECT_DIR / "app" / "assets" / "Hoopoe_display.png"
PAGE_ICON = PROJECT_DIR / "app" / "assets" / "Hoopoe_favicon.png"
UPLOAD_FOLDER = PROJECT_DIR / "data" / "uploads"
GREETING = "Hi, I am Hoopoe. How can I help you today?"


st.set_page_config(page_title=PAGE_TITLE, page_icon=str(PAGE_ICON), layout="wide")

from modules.bm25_retriever import build_bm25_retriever
from modules.embedder import create_retriever
from modules.loader import load_document
from modules.memory import clear_session_memory, record_turn
from modules.rag_chain import RetrievalPipeline, build_chat_chain, build_rag_chain
from modules.reranker import CrossEncoderReranker
from modules.retriever import get_hybrid_retriever
from modules.splitter import split_doc
from modules.utils import get_logger, load_config

logger = get_logger(__name__)


@st.cache_data
def read_config():
    return load_config()


def start_new_chat():
    chat_id = str(uuid.uuid4())
    chat_name = f"Chat {st.session_state.latest_chat_number}"

    st.session_state.chat_names[chat_id] = chat_name
    st.session_state.messages[chat_id] = [{"role": "assistant", "content": GREETING}]
    st.session_state.documents[chat_id] = {
        "indexed_files": {},
        "chunks": [],
        "pipeline": None,
    }
    st.session_state.active_chat_id = chat_id


def prepare_app_state(config):
    old_state_exists = (
        "existing_chat_list" in st.session_state
        and "chat_names" not in st.session_state
    )
    if old_state_exists:
        for key in ["latest_chat_number", "active_chat_id", "llm_provider", "memory_enabled"]:
            st.session_state.pop(key, None)

    if "latest_chat_number" not in st.session_state:
        st.session_state.latest_chat_number = 1

    if "chat_names" not in st.session_state:
        st.session_state.chat_names = {}

    if "messages" not in st.session_state:
        st.session_state.messages = {}

    if "documents" not in st.session_state:
        st.session_state.documents = {}

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = None

    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = config["llm"]["provider"]

    if "memory_enabled" not in st.session_state:
        st.session_state.memory_enabled = config["memory"]["enabled_default"]

    if "waiting_for_response" not in st.session_state:
        st.session_state.waiting_for_response = False

    if "pending_user_message" not in st.session_state:
        st.session_state.pending_user_message = None

    if not st.session_state.chat_names:
        start_new_chat()


def current_chat_id():
    if st.session_state.active_chat_id in st.session_state.chat_names:
        return st.session_state.active_chat_id

    first_chat_id = next(iter(st.session_state.chat_names))
    st.session_state.active_chat_id = first_chat_id
    return first_chat_id


def current_config(config):
    """A per-request copy of the base config with the sidebar's live toggles
    (LLM provider, memory on/off) applied on top. The embedding provider is
    deliberately NOT overridden here - it stays whatever config.yaml says,
    since switching it mid-chat would point at a FAISS index built with a
    different embedding model.
    """
    updated_config = copy.deepcopy(config)
    updated_config["llm"]["provider"] = st.session_state.llm_provider
    updated_config["memory"]["enabled_default"] = st.session_state.memory_enabled
    return updated_config


def save_file(uploaded_file, chat_id):
    folder = UPLOAD_FOLDER / chat_id
    folder.mkdir(parents=True, exist_ok=True)

    saved_file = folder / Path(uploaded_file.name).name
    saved_file.write_bytes(uploaded_file.getbuffer())
    return saved_file


def get_file_hash(uploaded_file):
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def index_document(uploaded_file, chat_id, config):
    """Load, chunk, and index one uploaded file, then rebuild this chat's
    full retrieval pipeline (FAISS + BM25 + reranker) from EVERY chunk
    indexed so far in this chat - not just the new file's chunks.

    This calls modules.embedder.create_retriever directly instead of the
    self-healing get_retriever from Module 3: get_retriever only builds a
    new FAISS index when none exists yet on disk, and simply loads the old
    one otherwise - so if a second, different file were uploaded to the
    same chat, its chunks would silently never make it into the index.
    Always rebuilding from the chat's full accumulated chunk list keeps
    every uploaded file searchable, at the cost of re-embedding earlier
    files again on each new upload (fine at this app's document sizes/count).
    """
    chat_documents = st.session_state.documents[chat_id]
    file_hash = get_file_hash(uploaded_file)

    if file_hash in chat_documents["indexed_files"]:
        return chat_documents["indexed_files"][file_hash], 0

    saved_file = save_file(uploaded_file, chat_id)
    loaded_docs = load_document(str(saved_file))

    chunk_settings = config["chunking"]
    new_chunks = split_doc(
        loaded_docs,
        chunk_size=chunk_settings["chunk_size"],
        chunk_overlap=chunk_settings["chunk_overlap"],
    )

    if not new_chunks:
        raise ValueError("No readable text was found in the uploaded document.")

    chat_documents["chunks"].extend(new_chunks)
    provider = config["embedding"]["provider"]

    faiss_retriever = create_retriever(chat_documents["chunks"], chat_id, provider, config)
    bm25_retriever = build_bm25_retriever(chat_documents["chunks"], config)
    hybrid_retriever = get_hybrid_retriever(faiss_retriever, bm25_retriever, config)

    reranker = CrossEncoderReranker(config) if config["reranking"]["enabled"] else None
    chat_documents["pipeline"] = RetrievalPipeline(hybrid_retriever, reranker, config)

    file_info = {
        "name": uploaded_file.name,
        "chunks": len(new_chunks),
    }
    chat_documents["indexed_files"][file_hash] = file_info
    return file_info, len(new_chunks)


def get_bot_reply(user_message, chat_id, config):
    pipeline = st.session_state.documents[chat_id]["pipeline"]

    if pipeline:
        retrieved = pipeline.invoke(user_message)
        get_answer = build_rag_chain(retrieved, config)
    else:
        get_answer = build_chat_chain(config)

    result = get_answer(user_message, chat_id)
    bot_reply = result["answer"]
    citations = result["citations"]

    if config["memory"]["enabled_default"]:
        record_turn(chat_id, user_message, bot_reply, config)

    return bot_reply, citations


def format_citation_caption(citations):
    """'Sources: leave_policy.docx, hr_handbook.docx' - deduplicated,
    in bracket-number order."""
    seen = set()
    filenames = []
    for citation in sorted(citations, key=lambda c: c["n"]):
        filename = citation["filename"]
        if filename not in seen:
            seen.add(filename)
            filenames.append(filename)
    return "Sources: " + ", ".join(filenames)


def show_chat_messages(chat_id):
    for message in st.session_state.messages[chat_id]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            citations = message.get("citations")
            if citations:
                st.caption(format_citation_caption(citations))


def add_message(chat_id, role, content, citations=None):
    message = {"role": role, "content": content}
    if citations:
        message["citations"] = citations
    st.session_state.messages[chat_id].append(message)


def show_app_title():
    icon_data = base64.b64encode(APP_ICON.read_bytes()).decode("utf-8")
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:14px; margin: 0 0 1.5rem 0;">
            <img src="data:image/png;base64,{icon_data}" width="76" height="76"
                 style="object-fit:contain; display:block;" />
            <h1 style="margin:0; padding:0 0 4px 0; line-height:1; font-size:3.2rem; font-weight:700;">
                Hoopoe
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


def apply_sidebar_style():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarContent"] {
                padding-top: 1.25rem;
            }
            [data-testid="stSidebar"] h3 {
                margin: 0 0 0.35rem 0;
                padding: 0;
            }
            [data-testid="stSidebar"] hr {
                margin: 1rem 0 0.85rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_footer_note():
    st.markdown(
        """
        <style>
            [data-testid="stBottom"]::after {
                content: "Made hand-in-hand by a human and an LLM. One had coffee.";
                display: block;
                width: 100%;
                margin: 0.35rem 0 0.15rem 0;
                color: rgba(128, 128, 128, 0.85);
                font-size: 0.82rem;
                line-height: 1.25;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


config = read_config()
prepare_app_state(config)

active_chat_id = current_chat_id()
runtime_config = current_config(config)

apply_sidebar_style()
show_app_title()

with st.sidebar:
    st.markdown("### ⚙️ Chat Settings")

    new_chat_col, clear_chat_col = st.columns(2)
    with new_chat_col:
        if st.button("New Chat", use_container_width=True):
            st.session_state.latest_chat_number += 1
            start_new_chat()
            active_chat_id = current_chat_id()
    with clear_chat_col:
        if st.button("Clear Conversation", use_container_width=True):
            clear_session_memory(active_chat_id)
            st.session_state.messages[active_chat_id] = [{"role": "assistant", "content": GREETING}]
            st.rerun()

    chat_ids = list(st.session_state.chat_names.keys())
    selected_chat_id = st.selectbox(
        "Active Chat Session",
        options=chat_ids,
        index=chat_ids.index(active_chat_id),
        format_func=lambda chat_id: st.session_state.chat_names[chat_id],
    )
    st.session_state.active_chat_id = selected_chat_id
    active_chat_id = selected_chat_id

    st.session_state.llm_provider = st.selectbox(
        "LLM Provider",
        options=["openai", "gemini"],
        index=["openai", "gemini"].index(st.session_state.llm_provider),
    )
    st.session_state.memory_enabled = st.toggle("Enable Memory", value=st.session_state.memory_enabled)
    runtime_config = current_config(config)

    st.divider()
    st.subheader(
        "Document",
        help="Upload a file and click Index Document to ask questions about the file.",
    )

    allowed_extensions = [ext.lstrip(".") for ext in config["upload"]["allowed_extensions"]]
    max_upload_size_mb = config["upload"]["max_size_mb"]
    max_upload_size_bytes = max_upload_size_mb * 1024 * 1024

    uploaded_file = st.file_uploader(
        f"Upload {'/'.join(ext.upper() for ext in allowed_extensions)} (max {max_upload_size_mb} MB)",
        type=allowed_extensions,
        accept_multiple_files=False,
    )

    upload_too_large = bool(uploaded_file and uploaded_file.size > max_upload_size_bytes)
    if upload_too_large:
        st.error(f"{uploaded_file.name} is too large. Maximum upload size is {max_upload_size_mb} MB.")

    if uploaded_file and st.button(
        "Index Document",
        type="primary",
        use_container_width=True,
        disabled=upload_too_large,
    ):
        with st.spinner("Indexing document..."):
            try:
                file_info, new_chunks = index_document(uploaded_file, active_chat_id, runtime_config)
                if new_chunks:
                    st.success(f"Indexed {file_info['name']} into {new_chunks} chunks.")
                else:
                    st.info(f"{file_info['name']} is already indexed.")
            except Exception as error:
                logger.error("Failed to index %s for chat %s: %s", uploaded_file.name, active_chat_id, error)
                st.error(str(error))

    indexed_files = st.session_state.documents[active_chat_id]["indexed_files"].values()
    if indexed_files:
        st.caption("Indexed files")
        for file_info in indexed_files:
            st.write(f"- {file_info['name']} ({file_info['chunks']} chunks)")
    else:
        st.caption("No document indexed for this chat.")


show_chat_messages(active_chat_id)

show_footer_note()

user_message = st.chat_input(
    "Waiting for Hoopoe..." if st.session_state.waiting_for_response else "Ask your question",
    disabled=st.session_state.waiting_for_response,
)

if user_message and not st.session_state.waiting_for_response:
    add_message(active_chat_id, "user", user_message)
    st.session_state.pending_user_message = user_message
    st.session_state.waiting_for_response = True
    st.rerun()

if st.session_state.waiting_for_response and st.session_state.pending_user_message:
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            bot_reply = None
            citations = []
            try:
                bot_reply, citations = get_bot_reply(
                    st.session_state.pending_user_message,
                    active_chat_id,
                    runtime_config,
                )
                st.markdown(bot_reply)
                if citations:
                    st.caption(format_citation_caption(citations))
            except Exception as error:
                logger.error("Failed to get a response for chat %s: %s", active_chat_id, error)
                st.error(str(error))
                bot_reply = "Sorry, something went wrong while generating a response. Please try again."

    add_message(active_chat_id, "assistant", bot_reply, citations)
    st.session_state.pending_user_message = None
    st.session_state.waiting_for_response = False
    st.rerun()
