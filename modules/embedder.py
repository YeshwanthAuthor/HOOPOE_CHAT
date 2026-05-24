import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from modules.utils import get_embedding_model

def build_or_update_vectorstore(
    docs: List[Document],
    config: dict,
    persist_dir: Optional[str] = None
):
    embedding_provider = config.get("embedding_provider")
    if not embedding_provider:
        raise ValueError(f"Provider {embedding_provider} is not configured")
    vector_store_config = config.get("vector_store")
    store_type = vector_store_config.get("type")

    print(f"Loading vector store: {store_type}")
    print(f"LLM Provider: {embedding_provider}")

    embedding_model = get_embedding_model(embedding_provider)

    if store_type == "faiss":
        if not persist_dir:
            persist_dir = os.path.join(os.getcwd(), "vectorstores", f"{embedding_provider}_{store_type}_index")
        os.makedirs(persist_dir, exist_ok=True)

        index_path = os.path.join(persist_dir, "index.faiss")
        #metadata_path = os.path.join(persist_dir, "index.pkl")

        if os.path.exists(index_path):
            print("Updating existing FAISS index")
            vectorstore = FAISS.load_local(persist_dir, embedding_model,  allow_dangerous_deserialization = True)
            if docs:
                vectorstore.add_documents(docs)
                vectorstore.save_local(persist_dir)
            else:
                raise ValueError("Docs not found")
        else:
            print("Creating new FAISS index")
            vectorstore = FAISS.from_documents(docs, embedding_model)
            vectorstore.save_local(persist_dir)

        return vectorstore

    raise ValueError(f"Vector store type {store_type} is not supported")




