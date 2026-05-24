import os

from langchain.schema.runnable import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))
load_dotenv(dotenv_path=env_path)

SESSION_MESSAGE_HISTORY = {}
SESSION_MESSAGE_SUMMARY = {}
SESSION_RECENT_HISTORY = {}

def get_session_message_history(session_id:str):
    if session_id not in SESSION_MESSAGE_HISTORY:
        SESSION_MESSAGE_HISTORY[session_id] = InMemoryChatMessageHistory()
    return SESSION_MESSAGE_HISTORY[session_id]

def get_session_summary_memory(session_id:str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key is required")
    if session_id not in SESSION_MESSAGE_SUMMARY:
        SESSION_MESSAGE_SUMMARY[session_id] = ConversationSummaryMemory(
            llm=ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key),
            chat_memory = get_session_message_history(session_id),
            return_messages=True,
        )
    return SESSION_MESSAGE_SUMMARY[session_id]

def get_memory_summary(session_id:str):
    try:
        memory = get_session_summary_memory(session_id)
    except ValueError:
        return "Summary memory is unavailable because OPENAI_API_KEY is not configured."
    return memory.load_memory_variables({}).get("summary", "")

def get_recent_history(session_id:str, n: int=10):
    history = SESSION_RECENT_HISTORY.get(session_id, [])
    return history[-n:]

def get_recent_history_formatted(session_id:str):
    history = get_recent_history(session_id, 10)
    formatted_history = []
    if not history:
        return "No recent chat messages"
    else:
        for turn in history:
            formatted_history.append(f"User: {turn['user']} Assistant: {turn['ai']}")
        return "\n\n".join(formatted_history)

def get_recent_context(session_id:str):
    return get_recent_history_formatted(session_id)


def update_recent_memory(session_id: str, user_input: str, ai_output: str, n: int = 2):
    if session_id not in SESSION_RECENT_HISTORY:
        SESSION_RECENT_HISTORY[session_id] = []
    SESSION_RECENT_HISTORY[session_id].append({"user": user_input, "ai": ai_output})
    SESSION_RECENT_HISTORY[session_id] =  SESSION_RECENT_HISTORY[session_id][-n:]

def add_memory_to_chain(rag_chain, session_id: str, enabled: bool = True, recent_n: int = 2):
    if not enabled:
        print("⚪ Memory OFF — stateless chat.")
        return rag_chain

    print(f"🟢 Hybrid Memory ON for session: {session_id}")
    summary_memory = get_session_summary_memory(session_id)

    def chain_with_summary(input_data, config=None):
        response = rag_chain.invoke(input_data, config=config)
        user_input = input_data["question"]
        ai_output = getattr(response, "content", str(response))

        # Save to summary memory
        summary_memory.save_context({"input": user_input}, {"output": ai_output})

        # Save to recent short-term memory
        update_recent_memory(session_id, user_input, ai_output, n=recent_n)

        # Refresh summary
        messages = summary_memory.chat_memory.messages
        existing_summary = summary_memory.load_memory_variables({}).get("summary", "")
        new_summary = summary_memory.predict_new_summary(messages, existing_summary)
        summary_memory._buffer = new_summary  # safe internal use

        # print(f"🧠 Updated summary (first 250 chars):\n{new_summary[:250]}...\n")
        # print(f"🕓 Recent {recent_n} Q&A turns stored.")

        return response

    runnable_with_summary = RunnableLambda(chain_with_summary)

    return RunnableWithMessageHistory(
        runnable_with_summary,
        get_session_history=lambda _: get_session_message_history(session_id),
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key=None,
    )
