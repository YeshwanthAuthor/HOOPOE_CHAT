"""Builds the final answer for a user question.

The app is restricted to document Q&A: build_chat_chain is only used for
short greetings/pleasantries (see is_small_talk below), never as a general
knowledge fallback. Anything else must be grounded in indexed documents.

Two chains are exposed:
  - build_chat_chain(config)              -> small-talk replies only, no documents involved
  - build_rag_chain(retrieved, config)    -> answer grounded in retrieved chunks,
                                              with numbered [1]..[n] citations

Both return a plain function, get_answer(question, session_id) -> dict, not a
class - there's no state here that needs to outlive one call, so a function
is all that's needed. Retrieval + reranking happen *before* build_rag_chain
is called, through RetrievalPipeline (also defined here). The full sequence
for one question:

  1. is_small_talk(question)                              -> route to build_chat_chain if True
  2. retrieved = retrieve_for_question(pipeline, question, config)
                                                            -> splits compound questions into
                                                               sub-questions (split_into_subquestions),
                                                               runs RetrievalPipeline.invoke() per
                                                               sub-question, merges the results
                                                               (merge_retrieved) - see that function's
                                                               docstring for why a single retrieval
                                                               pass isn't enough for a question that
                                                               spans more than one document/topic
  3. get_answer = build_rag_chain(retrieved, config)     -> returns a function
  4. result = get_answer(question, session_id)           -> {"answer": str, "citations": [...]}
"""

import os
import re

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

from modules.memory import get_memory_summary, get_recent_context
from modules.utils import get_logger

logger = get_logger(__name__)

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
load_dotenv(dotenv_path=env_path)

# Shown to the user instead of a guess when a document is indexed but the
# question isn't answered by it - i.e. no retrieved chunk individually
# clears reranking.min_relevance_score. See build_rag_chain's relevance
# filter/hallucination guard.
NOT_FOUND_MESSAGE = "No response found for the asked question in the documentation."

# Shown when the chat has no document indexed at all yet, so there is
# nothing to ground an answer in - used in place of the old general-chat
# fallback (see get_bot_reply in app/streamlit_app.py).
NO_DOCUMENT_MESSAGE = "Please upload a document before asking questions."

# Short greetings/pleasantries that bypass document grounding entirely and
# get a normal friendly reply from build_chat_chain instead of NOT_FOUND_MESSAGE
# or NO_DOCUMENT_MESSAGE. Deliberately a small, exact-match phrase list rather
# than an LLM classification call, so it stays free, instant, and predictable.
SMALL_TALK_PHRASES = {
    "hi", "hello", "hey", "hiya", "yo",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "thanks a lot", "thank you so much", "thx", "ty",
    "bye", "goodbye", "see you", "see ya", "take care",
    "ok", "okay", "k", "cool", "great", "nice", "sure", "sounds good",
    "how are you", "how are you doing", "whats up", "sup",
    "who are you", "what can you do", "what is your name",
}


def is_small_talk(text: str) -> bool:
    """True for a short greeting/pleasantry that doesn't need document
    grounding. Conservative on purpose: only an exact match (after lowercasing
    and stripping punctuation) against SMALL_TALK_PHRASES counts, and anything
    longer than 6 words is never treated as small talk - so "hi, what's the
    leave policy?" still gets routed through document retrieval instead of
    being waved through as a greeting.
    """
    normalized = re.sub(r"[^\w\s]", "", text.strip().lower())
    if not normalized or len(normalized.split()) > 6:
        return False
    return normalized in SMALL_TALK_PHRASES


CHAT_PROMPT_TEMPLATE = """
System Prompt:
{system_prompt}

Conversation Summary Memory:
{memory_summary}

Recent Chat History:
{recent_memory}

User Question:
{question}

This is a short greeting or pleasantry (not a substantive question - the caller
only routes messages here after is_small_talk() matches them). Respond
naturally and briefly. Do not attempt to answer factual questions here, even
if one is implied - this app only answers questions from indexed documents.
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
in the Retrieved Context above. If the question has multiple parts, answer each
part using whichever numbered chunks support it - the context may come from more
than one document. If the context does not answer the question (or a part of it),
reply exactly: "No response found for the asked question in the documentation." -
do not guess, and do not answer from general knowledge.
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


DECOMPOSE_PROMPT_TEMPLATE = """
A user asked the question below. If it is genuinely asking about more than one
distinct topic, split it into separate, standalone questions - one per line,
each fully self-contained (resolve shared context/pronouns from the original
question, e.g. "it"/"that"). If it is already about a single topic, output it
unchanged as one line. Output only the question line(s) - no numbering, no
bullets, no other text, and do not answer the question(s).

Question: {question}
"""


def split_into_subquestions(question: str, config: dict) -> list:
    """Break a compound question into separate standalone sub-questions, one
    per distinct topic - e.g. "what's the USA meal allowance and what are the
    standard working hours?" becomes two questions, one per topic.

    Why this exists: CrossEncoderReranker (Module 6) scores each candidate
    chunk against the *whole* query text. A chunk that would score well
    against "what's the USA meal allowance" gets its score pulled down when
    the same query also carries an unrelated "...and what are the standard
    working hours" clause - the cross-encoder is judging relevance to the
    full two-topic query, and half of that text is noise from that chunk's
    point of view. With enough dilution, chunks that answer either half
    perfectly can drop below reranking.min_relevance_score, and the whole
    question wrongly gets NOT_FOUND_MESSAGE even though each topic is
    covered by a (different) indexed document. Splitting first keeps each
    retrieval/rerank pass focused on one topic, so this can't happen.

    Uses one small LLM call. Falls back to [question] unchanged - no split -
    if the call fails, or returns something unparseable/implausible (more
    than 4 lines), so a decomposition hiccup degrades to a single retrieval
    pass instead of breaking the chat.
    """
    llm = get_chat_llm(config)
    prompt_text = DECOMPOSE_PROMPT_TEMPLATE.format(question=question)

    try:
        response = llm.invoke(prompt_text)
        text = getattr(response, "content", str(response))
    except Exception as e:
        logger.warning("Question decomposition failed for %r, using it as-is: %s", question, e)
        return [question]

    subquestions = []
    for raw_line in text.strip().splitlines():
        cleaned = re.sub(r"^[\s\-*\d.)]+", "", raw_line).strip()
        if cleaned:
            subquestions.append(cleaned)

    if not subquestions or len(subquestions) > 4:
        return [question]

    if len(subquestions) > 1:
        logger.info("Split question %r into %d sub-question(s): %s", question, len(subquestions), subquestions)
    return subquestions


def merge_retrieved(results_lists: list) -> list:
    """Merge several RetrievalPipeline.invoke() results - one list per
    sub-question - into a single list of (Document, score) tuples.

    A chunk retrieved for more than one sub-question is kept once, at its
    best (highest) score. The merged list is sorted by score descending
    (when every entry has a real score) so the strongest matches get the
    lowest, most prominent citation numbers once combine_docs/build_citations
    number them - dedup key is (source, chunk_index), the same fields
    loader.py/splitter.py already stamp onto every chunk's metadata.
    """
    best = {}
    for results in results_lists:
        for doc, score in results:
            key = (doc.metadata.get("source"), doc.metadata.get("chunk_index"))
            current = best.get(key)
            if current is None or (score is not None and (current[1] is None or score > current[1])):
                best[key] = (doc, score)

    merged = list(best.values())
    if merged and all(score is not None for _, score in merged):
        merged.sort(key=lambda pair: pair[1], reverse=True)
    return merged


def retrieve_for_question(pipeline: RetrievalPipeline, question: str, config: dict) -> list:
    """Full retrieval step for one user question, sub-question splitting
    included - this is what app/streamlit_app.py's get_bot_reply calls.
    RetrievalPipeline.invoke() itself stays single-query and unaware of
    decomposition, so it's still simple to call directly on its own (as
    Module 6/7's local test scripts do).

    Controlled by retrieval.decompose_compound_questions in config.yaml
    (default True). When on, every question pays for one extra, small LLM
    call (split_into_subquestions) before retrieval - worth it for
    correctness on compound questions, but set this to False in config.yaml
    to skip decomposition and go back to one retrieval pass per question
    (today's behavior) if that latency/cost isn't wanted. If the question
    comes back unsplit, retrieval happens once, same as before; if it
    splits into more than one sub-question, pipeline.invoke() runs once per
    sub-question and the results are combined with merge_retrieved().
    """
    if not config["retrieval"].get("decompose_compound_questions", True):
        return pipeline.invoke(question)

    subquestions = split_into_subquestions(question, config)
    if len(subquestions) == 1:
        return pipeline.invoke(subquestions[0])

    return merge_retrieved([pipeline.invoke(q) for q in subquestions])


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
    """Small-talk replies only - no retrieved documents involved.

    This is NOT a general-knowledge fallback: the app restricts users to
    document Q&A, so the caller (get_bot_reply in app/streamlit_app.py) only
    reaches this chain when is_small_talk(question) is True. Every other
    question goes through build_rag_chain, or gets NO_DOCUMENT_MESSAGE if no
    document is indexed yet for the chat.

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
        # skipped (disabled, or no reranker given) every score is None and
        # there is no relevance signal to filter on, so every retrieved
        # candidate is kept as-is. When reranking did run, keep only the
        # chunks that individually clear min_relevance_score.
        #
        # This must be a per-chunk filter, not just an all-below-threshold
        # check: CrossEncoderReranker.rerank() always returns exactly
        # final_k chunks (a *count* cap), so if even one of them is a good
        # match the other final_k-1 slots still get filled with whatever
        # scored next best - even chunks from unrelated documents that
        # scored below the relevance threshold. Without filtering those out
        # here, they'd ride along into both the LLM's context and the
        # "Sources" list shown to the user.
        reranking_ran = all(score is not None for score in scores)
        if reranking_ran:
            relevant_docs = [doc for doc, score in zip(docs, scores) if score >= min_relevance_score]
            if not relevant_docs:
                logger.info(
                    "All %d rerank scores below min_relevance_score=%s for question %r.",
                    len(scores), min_relevance_score, question,
                )
                return {"answer": NOT_FOUND_MESSAGE, "citations": []}
            if len(relevant_docs) < len(docs):
                logger.info(
                    "Dropped %d of %d retrieved chunk(s) below min_relevance_score=%s for question %r.",
                    len(docs) - len(relevant_docs), len(docs), min_relevance_score, question,
                )
        else:
            relevant_docs = docs

        context = combine_docs(relevant_docs)
        citations = build_citations(relevant_docs)
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
