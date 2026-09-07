from typing import List

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from modules.utils import get_logger

logger = get_logger(__name__)


def build_bm25_retriever(chunks: List[Document], config: dict) -> BM25Retriever:
    """Build an in-memory BM25 keyword retriever from the same chunk Documents
    used to build the FAISS index (Module 3).

    BM25Retriever has no persistent save/load, so this must be called fresh
    every time a document is (re)indexed for a chat session - the caller
    (Module 9) is responsible for rebuilding it alongside the FAISS retriever.
    """
    if not chunks:
        raise ValueError("No chunks provided to build the BM25 retriever from.")

    k = config["reranking"]["initial_k"]
    logger.info("Building BM25 retriever from %d chunks (k=%d)", len(chunks), k)
    return BM25Retriever.from_documents(chunks, k=k)
