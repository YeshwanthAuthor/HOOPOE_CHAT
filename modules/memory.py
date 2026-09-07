"""Two memory layers, both keyed by session_id (the chat_id):

  - Recent-turn memory: the last N raw question/answer pairs (SESSION_RECENT_HISTORY).
  - Summary memory: a running LLM-generated summary of the whole conversation
    (SESSION_MESSAGE_SUMMARY, backed by LangChain's ConversationSummaryMemory).

record_turn() is the single function the app should call once per turn to
keep both layers up to date - see its docstring for the bug this fixes.
"""

import os

from langchain.schema.runnable import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

from modules.utils import get_logger

logger = get_logger(__name__)

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
load_dotenv(dotenv_path=env_path)

SESSION_MESSAGE_HISTORY = {}
SESSION_MESSAGE_SUMMARY = {}
SESSION_RECENT_HISTORY = {}


def get_session_message_history(session_id: str):
    if session_id not in SESSION_MESSAGE_HISTORY:
        SESSION_MESSAGE_HISTORY[session_id] = InMemoryChatMessageHistory()
    return SESSION_MESSAGE_HISTORY[session_id]


def get_session_summary_memory(session_id: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required")
    if session_id not in SESSION_MESSAGE_SUMMARY:
        SESSION_MESSAGE_SUMMARY[session_id] = ConversationSummaryMemory(
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key),
            chat_memory=get_session_message_history(session_id),
            return_messages=True,
        )
    return SESSION_MESSAGE_SUMMARY[session_id]


def get_memory_summary(session_id: str) -> str:
    try:
        memory = get_session_summary_memory(session_id)
    except ValueError:
        return "Summary memory is unavailable because OPENAI_API_KEY is not configured."
    return memory.load_memory_variables({}).get("summary", "")


def get_recent_history(session_id: str, n: int = 10):
    history = SESSION_RECENT_HISTORY.get(session_id, [])
    return history[-n:]


def get_recent_context(session_id: str) -> str:
    history = get_recent_history(session_id, 10)
    if not history:
        return "No recent chat messages"
    return "\n\n".join(f"User: {turn['user']} Assistant: {turn['ai']}" for turn in history)


def update_recent_memory(session_id: str, user_input: str, ai_output: str, n: int = 2) -> None:
    if session_id not in SESSION_RECENT_HISTORY:
        SESSION_RECENT_HISTORY[session_id] = []
    SESSION_RECENT_HISTORY[session_id].append({"user": user_input, "ai": ai_output})
    SESSION_RECENT_HISTORY[session_id] = SESSION_RECENT_HISTORY[session_id][-n:]


def record_turn(session_id: str, user_input: str, ai_output: str, config: dict) -> None:
    """Persist one finished turn into BOTH memory layers. Call this once,
    right after get_bot_reply() has the final answer for a turn.

    Bug this fixes: the summary-refresh steps below (save_context, then
    predict_new_summary to roll the summary forward) used to live only
    inside add_memory_to_chain()'s chain_with_summary closure. Nothing in
    the app ever called add_memory_to_chain() - streamlit_app.py called
    update_recent_memory() directly instead - so recent-turn memory worked
    but the conversation summary stayed frozen/empty. record_turn() does
    both jobs, so calling it once per turn is enough; add_memory_to_chain()
    is no longer needed for correctness (kept below, marked unused).
    """
    max_recent_turns = config["memory"]["max_recent_turns"]
    update_recent_memory(session_id, user_input, ai_output, n=max_recent_turns)

    try:
        summary_memory = get_session_summary_memory(session_id)
    except ValueError as e:
        logger.warning("Skipping summary memory update for session %s: %s", session_id, e)
        return

    try:
        summary_memory.save_context({"input": user_input}, {"output": ai_output})
        messages = summary_memory.chat_memory.messages
        existing_summary = summary_memory.load_memory_variables({}).get("summary", "")
        new_summary = summary_memory.predict_new_summary(messages, existing_summary)
        summary_memory._buffer = new_summary
        logger.info("Updated summary memory for session %s.", session_id)
    except Exception as e:
        logger.error("Failed to update summary memory for session %s: %s", session_id, e)
        raise RuntimeError(f"Failed to update conversation summary memory: {e}") from e


def clear_session_memory(session_id: str) -> None:
    """Wipe both memory layers (and the underlying chat message history) for
    one session - used by the Streamlit 'Clear Conversation' button (Module 9).
    The indexed document/retrieval pipeline for that chat is untouched; only
    conversation memory resets.
    """
    SESSION_MESSAGE_HISTORY.pop(session_id, None)
    SESSION_MESSAGE_SUMMARY.pop(session_id, None)
    SESSION_RECENT_HISTORY.pop(session_id, None)
    logger.info("Cleared conversation memory for session %s.", session_id)


def add_memory_to_chain(rag_chain, session_id: str, enabled: bool = True, recent_n: int = 2):
    """NOTE: currently unused - nothing in the app calls this function.
    record_turn() above is the single entry point the app uses instead.
    Kept only because it wraps a chain in LangChain's RunnableWithMessageHistory,
    which some future LCEL-based chain might still want; confirm nothing
    references it before deleting.
    """
    if not enabled:
        logger.info("Memory disabled for session %s - stateless chat.", session_id)
        return rag_chain

    logger.info("Hybrid memory enabled for session %s.", session_id)
    summary_memory = get_session_summary_memory(session_id)

    def chain_with_summary(input_data, config=None):
        response = rag_chain.invoke(input_data, config=config)
        user_input = input_data["question"]
        ai_output = getattr(response, "content", str(response))

        summary_memory.save_context({"input": user_input}, {"output": ai_output})
        update_recent_memory(session_id, user_input, ai_output, n=recent_n)

        messages = summary_memory.chat_memory.messages
        existing_summary = summary_memory.load_memory_variables({}).get("summary", "")
        new_summary = summary_memory.predict_new_summary(messages, existing_summary)
        summary_memory._buffer = new_summary

        return response

    runnable_with_summary = RunnableLambda(chain_with_summary)

    return RunnableWithMessageHistory(
        runnable_with_summary,
        get_session_history=lambda _: get_session_message_history(session_id),
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key=None,
    )
