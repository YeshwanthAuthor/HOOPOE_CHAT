import base64
import contextlib
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
APP_ICON_B64 = base64.b64encode(APP_ICON.read_bytes()).decode("utf-8")
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

    if "indexing_in_progress" not in st.session_state:
        st.session_state.indexing_in_progress = False

    if "pending_upload_files" not in st.session_state:
        st.session_state.pending_upload_files = None

    if "last_indexing_result" not in st.session_state:
        st.session_state.last_indexing_result = None

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
    """Load, chunk, and save one uploaded file into this chat's accumulated
    chunk list. Does NOT build the retrieval pipeline itself - when
    uploading several files at once, call this once per file first, then
    call rebuild_pipeline() a single time for the whole batch. That way the
    FAISS index, BM25 index, and cross-encoder model are each built once per
    upload batch instead of once per file.
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

    file_info = {
        "name": uploaded_file.name,
        "chunks": len(new_chunks),
    }
    chat_documents["indexed_files"][file_hash] = file_info
    return file_info, len(new_chunks)


def rebuild_pipeline(chat_id, config):
    """Rebuild this chat's full retrieval pipeline (FAISS + BM25 + hybrid +
    reranker) from EVERY chunk indexed so far in this chat - not just the
    most recently uploaded file's chunks. Call this once after indexing a
    batch of one or more files with index_document().

    This calls modules.embedder.create_retriever directly instead of the
    self-healing get_retriever from Module 3: get_retriever only builds a
    new FAISS index when none exists yet on disk, and simply loads the old
    one otherwise - so if a second, different file were uploaded to the
    same chat, its chunks would silently never make it into the index.
    Always rebuilding from the chat's full accumulated chunk list keeps
    every uploaded file searchable, at the cost of re-embedding earlier
    files again on each new upload batch (fine at this app's document
    sizes/count).
    """
    chat_documents = st.session_state.documents[chat_id]
    provider = config["embedding"]["provider"]

    faiss_retriever = create_retriever(chat_documents["chunks"], chat_id, provider, config)
    bm25_retriever = build_bm25_retriever(chat_documents["chunks"], config)
    hybrid_retriever = get_hybrid_retriever(faiss_retriever, bm25_retriever, config)

    reranker = CrossEncoderReranker(config) if config["reranking"]["enabled"] else None
    chat_documents["pipeline"] = RetrievalPipeline(hybrid_retriever, reranker, config)


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
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:14px; margin: 0 0 1.5rem 0;">
            <img src="data:image/png;base64,{APP_ICON_B64}" width="76" height="76"
                 style="object-fit:contain; display:block;" />
            <h1 style="margin:0; padding:0 0 4px 0; line-height:1; font-size:3.2rem; font-weight:700;">
                Hoopoe
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )


HOOPOE_SPINNER_CSS = """
<style>
@keyframes hoopoe-glide {
    0%   { transform: translateX(0px); }
    50%  { transform: translateX(16px); }
    100% { transform: translateX(0px); }
}
@keyframes hoopoe-flutter {
    0%, 100% { transform: translateY(0px) rotate(-6deg); }
    25%      { transform: translateY(-7px) rotate(3deg); }
    50%      { transform: translateY(0px) rotate(6deg); }
    75%      { transform: translateY(-4px) rotate(-3deg); }
}
.hoopoe-spinner-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 4px 0;
}
.hoopoe-spinner-outer {
    display: inline-block;
    animation: hoopoe-glide 1.6s ease-in-out infinite;
}
.hoopoe-spinner-bird {
    display: block;
    width: 30px;
    height: 30px;
    object-fit: contain;
    animation: hoopoe-flutter 0.5s ease-in-out infinite;
}
.hoopoe-spinner-text {
    font-size: 0.95rem;
    color: inherit;
}
</style>
"""


@contextlib.contextmanager
def hoopoe_spinner(text: str):
    """Drop-in replacement for st.spinner: shows the Hoopoe icon animated
    with CSS to look like it's fluttering/flying in place, next to a status
    message, for the two slow actions in this app (indexing documents,
    generating a chat response). Usage is identical to st.spinner:
    `with hoopoe_spinner("Thinking..."):`.
    """
    placeholder = st.empty()
    placeholder.markdown(
        HOOPOE_SPINNER_CSS
        + f"""
        <div class="hoopoe-spinner-wrap">
            <span class="hoopoe-spinner-outer">
                <img class="hoopoe-spinner-bird" src="data:image/png;base64,{APP_ICON_B64}" />
            </span>
            <span class="hoopoe-spinner-text">{text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    try:
        yield
    finally:
        placeholder.empty()


def apply_sidebar_style():
    """Sidebar look-and-feel, in three parts:
      1. Static, no internal scrollbar - the sidebar grows to fit its
         content (New Chat, chat picker, Clear Conversation, provider,
         memory toggle, document uploader, indexed-files list) instead of
         being pinned to the viewport height with its own scroller.
      2. A consistent, compact vertical rhythm between every widget so the
         sidebar reads as one neatly stacked column instead of default
         Streamlit spacing (which varies widget to widget).
      3. A visibly distinct sidebar-collapse toggle - Streamlit's default
         arrow icon is a faint outline that's easy to miss, especially on
         a dark background. Selectors below target both places that
         control appears: attached to the sidebar itself when it's open,
         and floating over the main area when it's collapsed. Streamlit
         doesn't publish these as a stable public API, so if a Streamlit
         upgrade ever renames them and the toggle stops looking different,
         these selectors are the first thing to re-check in the browser's
         dev tools.
    """
    st.markdown(
        """
        <style>
            /* --- 1. Static sidebar: no internal scroller ------------------ */
            [data-testid="stSidebar"] {
                position: relative !important;
                height: auto !important;
                min-height: 100vh;
            }
            [data-testid="stSidebarContent"],
            [data-testid="stSidebarUserContent"] {
                height: auto !important;
                overflow: visible !important;
                padding-top: 1.25rem;
            }

            /* --- 2. Neat, consistent stacking of every sidebar widget ----- */
            [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
                gap: 0.6rem;
            }
            [data-testid="stSidebar"] h3 {
                margin: 0 0 0.5rem 0;
                padding: 0;
            }
            [data-testid="stSidebar"] hr {
                margin: 0.9rem 0 0.7rem 0;
            }
            [data-testid="stSidebar"] .stButton,
            [data-testid="stSidebar"] .stSelectbox,
            [data-testid="stSidebar"] .stToggle,
            [data-testid="stSidebar"] .stFileUploader {
                margin-bottom: 0.15rem;
            }

            /* --- 3. A sidebar-collapse toggle that's actually visible ----- */
            [data-testid="stSidebarCollapseButton"] button,
            [data-testid="collapsedControl"] button,
            [data-testid="stSidebar"] button[kind="header"],
            [data-testid="stSidebar"] button[kind="headerNoPadding"],
            [data-testid="collapsedControl"] button[kind="header"],
            [data-testid="collapsedControl"] button[kind="headerNoPadding"] {
                background-color: #FF4B4B !important;
                border-radius: 8px !important;
                box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
            }
            [data-testid="stSidebarCollapseButton"] button:hover,
            [data-testid="collapsedControl"] button:hover {
                background-color: #e0423f !important;
            }
            [data-testid="stSidebarCollapseButton"] svg,
            [data-testid="collapsedControl"] svg {
                fill: #FFFFFF !important;
                color: #FFFFFF !important;
                width: 1.3rem !important;
                height: 1.3rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hide_theme_switcher():
    """The app is forced into dark mode via .streamlit/config.toml
    ([theme] base="dark"). That alone only sets the default - Streamlit's
    built-in "Settings" menu (the hamburger icon, top-right) still lets a
    user manually switch to Light. Hiding that menu is the only way to
    remove the option entirely, since Streamlit has no config flag to
    disable just the theme picker inside it. #MainMenu is Streamlit's
    long-standing id for this menu; the data-testid selector covers newer
    versions that render it differently. This does not touch the sidebar's
    own collapse arrow, which is a separate element.
    """
    st.markdown(
        """
        <style>
            #MainMenu, [data-testid="stMainMenu"] {
                visibility: hidden;
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

hide_theme_switcher()
apply_sidebar_style()
show_app_title()

with st.sidebar:
    st.markdown("### ⚙️ Chat Settings")

    if st.button("➕ New Chat", use_container_width=True):
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

    # Placed right below the chat selector, not next to New Chat, since it
    # acts on whichever chat is selected above - not on the sidebar as a whole.
    if st.button("🧹 Clear Conversation", use_container_width=True):
        clear_session_memory(active_chat_id)
        st.session_state.messages[active_chat_id] = [{"role": "assistant", "content": GREETING}]
        st.rerun()

    st.session_state.llm_provider = st.selectbox(
        "LLM Provider",
        options=["openai", "gemini"],
        index=["openai", "gemini"].index(st.session_state.llm_provider),
    )
    st.session_state.memory_enabled = st.toggle("Enable Memory", value=st.session_state.memory_enabled)
    runtime_config = current_config(config)

    st.divider()
    st.subheader(
        "Documents",
        help="Upload one or more files and click Index Documents to ask questions about them.",
    )

    allowed_extensions = [ext.lstrip(".") for ext in config["upload"]["allowed_extensions"]]
    max_upload_size_mb = config["upload"]["max_size_mb"]
    max_upload_size_bytes = max_upload_size_mb * 1024 * 1024

    uploaded_files = st.file_uploader(
        f"Upload {'/'.join(ext.upper() for ext in allowed_extensions)} (max {max_upload_size_mb} MB each)",
        type=allowed_extensions,
        accept_multiple_files=True,
    )

    too_large_files = [f for f in uploaded_files if f.size > max_upload_size_bytes]
    if too_large_files:
        too_large_names = ", ".join(f.name for f in too_large_files)
        st.error(f"Too large (max {max_upload_size_mb} MB each): {too_large_names}")

    index_button_disabled = bool(too_large_files) or st.session_state.indexing_in_progress
    if uploaded_files and st.button(
        "Index Documents",
        type="primary",
        use_container_width=True,
        disabled=index_button_disabled,
    ):
        # The actual indexing work happens later in the script, after the
        # main chat_input is created - that way chat_input's disabled=True
        # reaches the browser before the slow indexing work starts, instead
        # of after it finishes. See the "indexing_in_progress" block below.
        st.session_state.pending_upload_files = uploaded_files
        st.session_state.indexing_in_progress = True
        st.session_state.last_indexing_result = None
        st.rerun()

    last_result = st.session_state.last_indexing_result
    if last_result:
        if last_result["newly_indexed"]:
            names = ", ".join(f"{info['name']} ({info['chunks']} chunks)" for info in last_result["newly_indexed"])
            st.success(f"Indexed: {names}")
        if last_result["pipeline_error"]:
            st.error(f"Indexed the file content but failed to build the search index: {last_result['pipeline_error']}")
        if last_result["already_indexed"]:
            names = ", ".join(info["name"] for info in last_result["already_indexed"])
            st.info(f"Already indexed: {names}")
        for filename, error_message in last_result["failed"]:
            st.error(f"{filename}: {error_message}")
        st.session_state.last_indexing_result = None

    indexed_files = st.session_state.documents[active_chat_id]["indexed_files"].values()
    if indexed_files:
        st.caption("Indexed files")
        for file_info in indexed_files:
            st.write(f"- {file_info['name']} ({file_info['chunks']} chunks)")
    else:
        st.caption("No document indexed for this chat.")


show_chat_messages(active_chat_id)

show_footer_note()

chat_input_disabled = st.session_state.waiting_for_response or st.session_state.indexing_in_progress
if st.session_state.waiting_for_response:
    chat_placeholder = "Waiting for Hoopoe..."
elif st.session_state.indexing_in_progress:
    chat_placeholder = "Indexing documents..."
else:
    chat_placeholder = "Ask your question"

user_message = st.chat_input(chat_placeholder, disabled=chat_input_disabled)

if user_message and not chat_input_disabled:
    add_message(active_chat_id, "user", user_message)
    st.session_state.pending_user_message = user_message
    st.session_state.waiting_for_response = True
    st.rerun()

if st.session_state.waiting_for_response and st.session_state.pending_user_message:
    with st.chat_message("assistant"):
        with hoopoe_spinner("Thinking..."):
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

if st.session_state.indexing_in_progress and st.session_state.pending_upload_files:
    files_to_index = st.session_state.pending_upload_files
    newly_indexed = []
    already_indexed = []
    failed = []

    with hoopoe_spinner(f"Indexing {len(files_to_index)} document(s)..."):
        for uploaded_file in files_to_index:
            try:
                file_info, new_chunks = index_document(uploaded_file, active_chat_id, runtime_config)
                if new_chunks:
                    newly_indexed.append(file_info)
                else:
                    already_indexed.append(file_info)
            except Exception as error:
                logger.error("Failed to index %s for chat %s: %s", uploaded_file.name, active_chat_id, error)
                failed.append((uploaded_file.name, str(error)))

        pipeline_error = None
        if newly_indexed:
            try:
                rebuild_pipeline(active_chat_id, runtime_config)
            except Exception as error:
                logger.error("Failed to rebuild retrieval pipeline for chat %s: %s", active_chat_id, error)
                pipeline_error = str(error)

    st.session_state.last_indexing_result = {
        "newly_indexed": newly_indexed,
        "already_indexed": already_indexed,
        "failed": failed,
        "pipeline_error": pipeline_error,
    }
    st.session_state.pending_upload_files = None
    st.session_state.indexing_in_progress = False
    st.rerun()
