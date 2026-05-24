import copy
import base64
import hashlib
import json
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
CONFIG_FILE = PROJECT_DIR / "config" / "config.json"
UPLOAD_FOLDER = PROJECT_DIR / "data" / "uploads"
VECTORSTORE_FOLDER = PROJECT_DIR / "vectorstores"
GREETING = "Hi, I am Hoopoe. How can I help you today?"
MAX_UPLOAD_SIZE_BYTES = 2 * 1024 * 1024
MAX_UPLOAD_SIZE_LABEL = "2 MB"


st.set_page_config(page_title=PAGE_TITLE, page_icon=str(PAGE_ICON), layout="wide")

from modules.embedder import build_or_update_vectorstore
from modules.loader import load_document
from modules.memory import update_recent_memory
from modules.rag_chain import build_chat_chain, build_rag_chain
from modules.retriever import get_retriever
from modules.splitter import split_doc


@st.cache_data
def read_config():
    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def start_new_chat():
    chat_id = str(uuid.uuid4())
    chat_name = f"Chat {st.session_state.latest_chat_number}"

    st.session_state.chat_names[chat_id] = chat_name
    st.session_state.messages[chat_id] = [{"role": "assistant", "content": GREETING}]
    st.session_state.documents[chat_id] = {
        "indexed_files": {},
        "retriever": None,
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
        st.session_state.llm_provider = config.get("llm_provider", "openai")

    if "memory_enabled" not in st.session_state:
        st.session_state.memory_enabled = config.get("memory", {}).get("enabled_default", True)

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
    updated_config = copy.deepcopy(config)
    updated_config["llm_provider"] = st.session_state.llm_provider
    updated_config.setdefault("memory", {})
    updated_config["memory"]["enabled"] = st.session_state.memory_enabled
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
    chat_documents = st.session_state.documents[chat_id]
    file_hash = get_file_hash(uploaded_file)

    if file_hash in chat_documents["indexed_files"]:
        return chat_documents["indexed_files"][file_hash], 0

    saved_file = save_file(uploaded_file, chat_id)
    loaded_docs = load_document(str(saved_file))

    chunk_settings = config.get("chunking", {})
    chunks = split_doc(
        loaded_docs,
        chunk_size=chunk_settings.get("chunk_size", 500),
        chunk_overlap=chunk_settings.get("chunk_overlap", 100),
    )

    if not chunks:
        raise ValueError("No readable text was found in the uploaded document.")

    index_folder = VECTORSTORE_FOLDER / chat_id / config["embedding_provider"]
    vectorstore = build_or_update_vectorstore(chunks, config, persist_dir=str(index_folder))
    chat_documents["retriever"] = get_retriever(vectorstore, config)

    file_info = {
        "name": uploaded_file.name,
        "chunks": len(chunks),
    }
    chat_documents["indexed_files"][file_hash] = file_info
    return file_info, len(chunks)


def get_bot_reply(user_message, chat_id, config):
    retriever = st.session_state.documents[chat_id]["retriever"]

    if retriever:
        chain = build_rag_chain(retriever, config)
    else:
        chain = build_chat_chain(config)

    response = chain.invoke({"question": user_message, "session_id": chat_id})
    bot_reply = getattr(response, "content", str(response))

    if config.get("memory", {}).get("enabled", True):
        update_recent_memory(
            chat_id,
            user_input=user_message,
            ai_output=bot_reply,
            n=config.get("memory", {}).get("max_memory_window", 5),
        )

    return bot_reply


def show_chat_messages(chat_id):
    for message in st.session_state.messages[chat_id]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def add_message(chat_id, role, content):
    st.session_state.messages[chat_id].append({"role": role, "content": content})


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

    if st.button("New Chat", use_container_width=True):
        st.session_state.latest_chat_number += 1
        start_new_chat()
        active_chat_id = current_chat_id()

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

    uploaded_file = st.file_uploader(
        f"Upload PDF/DOCX/TXT (max {MAX_UPLOAD_SIZE_LABEL})",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
    )

    upload_too_large = bool(uploaded_file and uploaded_file.size > MAX_UPLOAD_SIZE_BYTES)
    if upload_too_large:
        st.error(f"{uploaded_file.name} is too large. Maximum upload size is {MAX_UPLOAD_SIZE_LABEL}.")

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
            try:
                bot_reply = get_bot_reply(
                    st.session_state.pending_user_message,
                    active_chat_id,
                    runtime_config,
                )
            except Exception as error:
                bot_reply = f"Error: {error}"
            st.markdown(bot_reply)

    add_message(active_chat_id, "assistant", bot_reply)
    st.session_state.pending_user_message = None
    st.session_state.waiting_for_response = False
    st.rerun()
