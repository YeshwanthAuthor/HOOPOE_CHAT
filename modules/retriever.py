import os

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import MultiQueryRetriever
from langchain.retrievers import EnsembleRetriever

from modules.utils import get_logger

logger = get_logger(__name__)


def get_retriever(vectorstore, config):
    """Base or multiquery retriever built straight off a vectorstore.

    Kept as a fallback path for when hybrid retrieval (get_hybrid_retriever,
    below) is turned off in config, or when multiquery mode is wanted instead
    of hybrid search.

    NOTE: currently unused - app/streamlit_app.py only imports get_hybrid_retriever
    from this module, never this function, and nothing reads config['retrieval']
    ['hybrid_enabled'] to decide between them. So this whole function is
    unreachable from the running app today, same as embedder.get_retriever
    (see that function's note). Kept because it's still correct on its own
    and would become useful again if hybrid_enabled or multiquery mode were
    ever wired up in the UI; confirm nothing references it before deleting.

    Note: config.yaml has no "retriever" section by default, so retriever_type
    always falls back to "base" unless one is added - this keeps the
    multiquery option available without requiring it.
    """
    retriever_config = config.get("retriever", {})
    retriever_type = retriever_config.get("type", "base")
    top_k = config["reranking"]["initial_k"]

    if retriever_type == "base":
        return vectorstore.as_retriever(search_kwargs={"k": top_k})
    elif retriever_type == "multiquery":
        llm_provider = config["llm"]["provider"]
        if llm_provider == "openai":
            llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=config["llm"]["openai_model"],
                temperature=0,
            )
        elif llm_provider in ("gemini", "google"):
            llm = ChatGoogleGenerativeAI(
                model=config["llm"]["gemini_model"],
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0,
            )
        else:
            raise ValueError(f"Provider {llm_provider} is not valid.")
        return MultiQueryRetriever.from_llm(vectorstore.as_retriever(search_kwargs={"k": top_k}), llm)
    else:
        raise ValueError(f"Retriever type {retriever_type} is not valid.")


def get_hybrid_retriever(faiss_retriever, bm25_retriever, config: dict):
    """Combine the FAISS retriever (Module 3) and BM25 retriever (Module 4)
    into one EnsembleRetriever, weighted per config['retrieval'].

    faiss_retriever is what embedder.get_retriever(...) returns - already a
    LangChain retriever, not a raw vectorstore.
    """
    vector_weight = config["retrieval"]["vector_weight"]
    bm25_weight = config["retrieval"]["bm25_weight"]

    logger.info(
        "Building hybrid retriever (vector_weight=%s, bm25_weight=%s)",
        vector_weight, bm25_weight,
    )
    return EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[vector_weight, bm25_weight],
    )
