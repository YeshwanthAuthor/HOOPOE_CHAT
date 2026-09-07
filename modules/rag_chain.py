"""Builds the final answer for a user question.

Two chains are exposed:
  - build_chat_chain(config)              -> general chat, no documents involved
  - build_rag_chain(retrieved, config)    -> answer grounded in retrieved chunks,
                                              with numbered [1]..[n] citations

Both return a plain function, get_answer(question, session_id) -> dict, not a
class - there's no state here that needs to outlive one call, so a function
is all that's needed. Retrieval + reranking happen *before* build_rag_chain
is called, through RetrievalPipeline (also defined here). The full sequence
for one question:

  1. retrieved = RetrievalPipeline(...).invoke(query)   -> hybrid search, then rerank
  2. get_answer = build_rag_chain(retrieved, config)     -> returns a function
  3. result = get_answer(question, session_id)           -> {"answer": str, "citations": [...]}
"""

import os

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from modules.memory import get_memory_summary, get_recent_context
from modules.utils import get_logger

logger = get_logger(__name__)

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
load_dotenv(dotenv_path=env_path)

# Shown to the user instead of a guess when there is nothing to answer from.
NOT_FOUND_MESSAGE = "The information was not found in the uploaded documents."

CHAT_PROMPT_TEMPLATE = """
System Prompt:
{system_prompt}

Conversation Summary Memory:
{memory_summary}

Recent Chat History:
{recent_memory}

User Question:
{question}

Respond naturally and helpfully. If the user asks about a document but no document
has been uploaded, say that you can answer generally and that uploading a document
will let you answer from that source.
"""

RAG_PROMPT_TEMPLATE = """
System Prompt:
{system_prompt}

Conversation Summary Memory:
{memory_summary}

Recent Chat History:
{recent_memory}

Retrieved Context:
{context}

User Question:
{question}

Answer only using the Retrieved Context above. Cite the bracket number, like [1],
immediately after any fact you take from the context. Only use numbers that appear
in the Retrieved Context above. If the context does not answer the question, say
the information was not found in the uploaded documents - do not guess.
"""


def get_chat_llm(config: dict):
    """Build the chat LLM from config["llm"] (nested schema from config.yaml)."""
    llm_config = config["llm"]
    llm_provider = llm_config["provider"]
    prompt_config = config["prompt"]

    if llm_provider == "openai":
        return ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=llm_config["openai_model"],
            temperature=prompt_config.get("temperature", 0),
            max_tokens=prompt_config.get("max_tokens"),
        )
    if llm_provider in ("google", "gemini"):
        return ChatGoogleGenerativeAI(
            model=llm_config["gemini_model"],
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=prompt_config.get("temperature", 0),
            max_output_tokens=prompt_config.get("max_tokens"),
        )
    raise ValueError(f"Provider {llm_provider} not recognized")


def get_system_prompt(config: dict) -> str:
    return config["prompt"]["system_prompt"]


def _get_memory_context(session_id, memory_enabled: bool):
    """Shared by both chains below: the two memory-summary lines that go
    into every prompt, or a fixed placeholder when memory is turned off.
    """
    if not memory_enabled:
        return "Memory is disabled.", "Memory is disabled."
    return get_memory_summary(session_id), get_recent_context(session_id)


def _call_llm(llm, prompt_text: str, question: str) -> str:
    """Shared by both chains below: call the LLM and return its text,
    with one consistent, actionable error if the call fails."""
    logger.info("Calling LLM for question: %r", question)
    try:
        response = llm.invoke(prompt_text)
    except Exception as e:
        logger.error("LLM call failed for question %r: %s", question, e)
        raise RuntimeError(f"Failed to get a response from the language model: {e}") from e
    return getattr(response, "content", str(response))


class RetrievalPipeline:
    """Single composition point between retrieval and the RAG chain.

    Stage 1: hybrid search (Module 5's EnsembleRetriever of FAISS + BM25).
    Stage 2: cross-encoder reranking (Module 6), only if enabled in config.

    This one holds state across calls (the retriever and reranker objects,
    kept in Streamlit session state between questions), which is why it's a
    class rather than a function like the two chain builders below.

    Call once per question:
        retrieved = RetrievalPipeline(hybrid_retriever, reranker, config).invoke(query)
        get_answer = build_rag_chain(retrieved, config)
    """

    def __init__(self, hybrid_retriever, reranker, config: dict):
        self.hybrid_retriever = hybrid_retriever
        self.reranker = reranker
        self.reranking_enabled = config["reranking"]["enabled"]

    def invoke(self, query: str):
        """Returns a list of (Document, score) tuples.

        score is a float when reranking ran, or None when reranking is
        disabled (or no reranker was supplied) - callers use this to tell
        the two cases apart.
        """
        try:
            candidates = self.hybrid_retriever.invoke(query)
        except Exception as e:
            logger.error("Hybrid retrieval failed for query %r: %s", query, e)
            raise RuntimeError(f"Failed to retrieve documents for the question: {e}") from e

        logger.info("Retrieved %d candidate chunks for query: %r", len(candidates), query)

        if not self.reranking_enabled or self.reranker is None:
            return [(doc, None) for doc in candidates]

        try:
            return self.reranker.rerank(query, candidates)
        except Exception as e:
            logger.error("Reranking failed for query %r: %s", query, e)
            raise RuntimeError(f"Failed to rerank retrieved documents: {e}") from e


def combine_docs(docs) -> str:
    """Number each chunk [1]..[n] with its source filename, in the same order
    used for build_citations - this is what keeps citation numbers in the
    prompt consistent with the numbers returned to the UI.
    """
    if not docs:
        return "No relevant context found."

    context_parts = []
    for i, doc in enumerate(docs, start=1):
        filename = doc.metadata.get("filename", "unknown source")
        context_parts.append(f"[{i}] Source: {filename}\n{doc.page_content}")
    return "\n\n".join(context_parts)


def build_citations(docs) -> list:
    """One citation entry per retrieved chunk, numbered to match combine_docs.

    Not deduplicated on purpose - the same document can legitimately supply
    several chunks/citations, and collapsing duplicates for display is the
    UI's job (Module 9), not this module's.
    """
    return [
        {"n": i, "filename": doc.metadata.get("filename", "unknown source")}
        for i, doc in enumerate(docs, start=1)
    ]


def build_chat_chain(config: dict):
    """General conversation, no retrieved documents involved.

    Returns get_answer(question, session_id=None) -> {"answer": str,
    "citations": []}, matching build_rag_chain's return shape so the UI can
    treat both chains the same way.

    get_answer only *reads* memory (get_memory_summary/get_recent_context)
    - it never writes to it. Once the caller has the answer, it must call
    modules.memory.record_turn(session_id, question, answer, config) itself
    to persist the turn. Keeping read and write on opposite sides of this
    call is deliberate: mixing them together is how Module 8's summary-
    memory bug happened in the first place (the write step lived inside a
    wrapper nothing ever called).
    """
    memory_enabled = config["memory"]["enabled_default"]
    llm = get_chat_llm(config)
    system_prompt = get_system_prompt(config)
    prompt_template = ChatPromptTemplate.from_template(CHAT_PROMPT_TEMPLATE)

    def get_answer(question: str, session_id=None) -> dict:
        memory_summary, recent_memory = _get_memory_context(session_id, memory_enabled)
        prompt_text = prompt_template.format(
            system_prompt=system_prompt,
            memory_summary=memory_summary,
            recent_memory=recent_memory,
            question=question,
        )
        answer = _call_llm(llm, prompt_text, question)
        return {"answer": answer, "citations": []}

    return get_answer


def build_rag_chain(retrieved: list, config: dict):
    """Answer grounded in already-retrieved-and-reranked chunks.

    retrieved is the exact output of RetrievalPipeline.invoke(query): a list
    of (Document, score-or-None) tuples. Retrieval does not happen here - it
    already happened in RetrievalPipeline, so this function only does
    prompting, the hallucination guard, and the LLM call.

    Returns get_answer(question, session_id=None) -> {"answer": str,
    "citations": [{"n": int, "filename": str}, ...]}.

    Like build_chat_chain above, get_answer only reads memory - the caller
    must call modules.memory.record_turn(session_id, question, answer,
    config) after calling it, to persist the turn.
    """
    memory_enabled = config["memory"]["enabled_default"]
    min_relevance_score = config["reranking"]["min_relevance_score"]
    llm = get_chat_llm(config)
    system_prompt = get_system_prompt(config)
    prompt_template = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    docs = [doc for doc, score in retrieved]
    scores = [score for doc, score in retrieved]

    def get_answer(question: str, session_id=None) -> dict:
        if not docs:
            logger.info("No documents retrieved for question %r.", question)
            return {"answer": NOT_FOUND_MESSAGE, "citations": []}

        # Reranking assigns a real float score to every doc; when it is
        # skipped (disabled, or no reranker given) every score is None, so
        # this guard only fires when reranking actually ran and every
        # candidate scored below the configured threshold.
        reranking_ran = all(score is not None for score in scores)
        if reranking_ran and all(score < min_relevance_score for score in scores):
            logger.info(
                "All %d rerank scores below min_relevance_score=%s for question %r.",
                len(scores), min_relevance_score, question,
            )
            return {"answer": NOT_FOUND_MESSAGE, "citations": []}

        context = combine_docs(docs)
        citations = build_citations(docs)
        memory_summary, recent_memory = _get_memory_context(session_id, memory_enabled)
        prompt_text = prompt_template.format(
            system_prompt=system_prompt,
            memory_summary=memory_summary,
            recent_memory=recent_memory,
            context=context,
            question=question,
        )
        answer = _call_llm(llm, prompt_text, question)
        return {"answer": answer, "citations": citations}

    return get_answer
