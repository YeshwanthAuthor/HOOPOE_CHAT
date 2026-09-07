from langchain_core.documents import Document
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

from modules.utils import get_logger

logger = get_logger(__name__)


def split_doc(docs: List[Document], chunk_size=500, chunk_overlap=50) -> List[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap <= 0 or chunk_overlap > chunk_size:
        raise ValueError("chunk_overlap must be > 0 and < chunk_size")

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    split_docs = []

    for doc in docs:
        for i, text in enumerate(splitter.split_text(doc.page_content)):
            meta = dict(doc.metadata) if doc.metadata else {}
            meta.update({"chunk_index": i})
            split_docs.append(Document(page_content=text, metadata=meta))

    logger.info(
        "Split %d document(s) into %d chunk(s) (chunk_size=%d, chunk_overlap=%d)",
        len(docs), len(split_docs), chunk_size, chunk_overlap,
    )
    return split_docs
