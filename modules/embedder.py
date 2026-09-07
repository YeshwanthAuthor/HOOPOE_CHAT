import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from modules.utils import get_embedding_model, get_logger, PROJECT_ROOT

logger = get_logger(__name__)


def get_index_path(config: dict, chat_id: str, provider: str) -> str:
    """Per-chat, per-provider FAISS folder: vectorstores/<chat_id>/<provider>/."""
    base_path = config["vector_store"]["base_path"]
    return os.path.join(PROJECT_ROOT, base_path, chat_id, provider)


def _index_path_exists(index_path: str, index_name: str) -> bool:
    """True if both <index_name>.faiss and <index_name>.pkl exist."""
    faiss_path = os.path.join(index_path, f"{index_name}.faiss")
    pkl_path = os.path.join(index_path, f"{index_name}.pkl")
    return os.path.isfile(faiss_path) and os.path.isfile(pkl_path)


def _search_k(config: dict) -> int:
    """How many candidates the FAISS retriever itself should fetch. Uses
    reranking.initial_k so there's a large-enough pool for Module 6 to rerank."""
    return config["reranking"]["initial_k"]


def create_retriever(chunks: List[Document], chat_id: str, provider: str, config: dict):
    """Embed chunks, build a new FAISS index, save it, and return a retriever."""
    if not chunks:
        raise ValueError("No chunks provided to build the FAISS index from.")

    index_path = get_index_path(config, chat_id, provider)
    index_name = config["vector_store"]["index_name"]
    embedding_model = get_embedding_model(config, provider)

    os.makedirs(index_path, exist_ok=True)
    logger.info("Creating new FAISS index (chat_id=%s, provider=%s) at %s", chat_id, provider, index_path)
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    vectorstore.save_local(index_path, index_name=index_name)

    return vectorstore.as_retriever(search_kwargs={"k": _search_k(config)})


def load_retriever(chat_id: str, provider: str, config: dict):
    """Load an existing FAISS index from disk and return a retriever."""
    index_path = get_index_path(config, chat_id, provider)
    index_name = config["vector_store"]["index_name"]
    embedding_model = get_embedding_model(config, provider)

    vectorstore = FAISS.load_local(
        index_path,
        embedding_model,
        index_name=index_name,
        allow_dangerous_deserialization=True,
    )
    return vectorstore.as_retriever(search_kwargs={"k": _search_k(config)})


def get_retriever(config: dict, chat_id: str, provider: str, chunks_if_needed: Optional[List[Document]] = None):
    """Self-healing entry point: load the index if it exists, build it from
    chunks_if_needed if it doesn't, and rebuild from chunks_if_needed if the
    existing index turns out to be corrupted or incompatible.

    NOTE: as of Module 9, app/streamlit_app.py calls create_retriever directly
    instead of this function - it always has the chat's full chunk list in
    memory and needs every uploaded file folded into a fresh index, which
    this function's "load if it already exists on disk" branch would skip
    (see index_document's docstring). That makes this self-healing path
    currently unreachable from the running app; it's kept because it's
    still correct on its own, is covered by Module 3's own test, and would
    become useful again for a feature like resuming a chat's index without
    re-uploading. Confirm nothing else calls it before deleting.
    """
    index_path = get_index_path(config, chat_id, provider)
    index_name = config["vector_store"]["index_name"]

    if not _index_path_exists(index_path, index_name):
        logger.warning(
            "No FAISS index found (chat_id=%s, provider=%s) at %s. Creating a new one.",
            chat_id, provider, index_path,
        )
        if chunks_if_needed is None:
            raise RuntimeError(
                f"FAISS index not found at {index_path}. Pass chunks_if_needed to build one."
            )
        return create_retriever(chunks_if_needed, chat_id, provider, config)

    try:
        return load_retriever(chat_id, provider, config)
    except Exception as e:
        logger.warning("Failed to load FAISS index at %s: %s", index_path, e)
        if chunks_if_needed is None:
            raise RuntimeError(
                f"Existing FAISS index at {index_path} appears corrupted or incompatible, "
                "and no chunks were provided to rebuild it. Delete that folder and re-index."
            ) from e
        logger.info("Rebuilding FAISS index from chunks (chat_id=%s, provider=%s)", chat_id, provider)
        return create_retriever(chunks_if_needed, chat_id, provider, config)
