import os

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.retrievers import MultiQueryRetriever

def get_retriever(vectorstore, config):
    retriever_config = config.get("retriever", {})
    retriever = retriever_config.get("type", "base")
    top_k = retriever_config.get("top_k", 3)
    if retriever == "base":
        return vectorstore.as_retriever(search_kwargs={"k": top_k})
    elif retriever == "multiquery":
        llm_provider = config.get("llm_provider")
        if not llm_provider:
            raise ValueError(f"Provider {llm_provider} is not valid.")
        elif llm_provider == "openai":
            llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o-mini", temperature=0)
        elif llm_provider in ("gemini", "google"):
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        else:
            raise ValueError(f"Provider {llm_provider} is not valid.")
        return MultiQueryRetriever.from_llm(vectorstore.as_retriever(search_kwargs={"k": top_k}), llm)
    else:
        raise ValueError(f"Retriever type {retriever} is not valid.")



