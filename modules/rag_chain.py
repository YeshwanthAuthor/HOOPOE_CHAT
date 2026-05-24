from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from modules.memory import get_memory_summary, get_recent_context


def get_chat_llm(config):
    llm_provider = config.get("llm_provider")
    prompt_config = config.get("prompt", {})

    if llm_provider == "openai":
        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=prompt_config.get("temperature", 0),
            streaming=True,
        )
    if llm_provider in ("google", "gemini"):
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=prompt_config.get("temperature", 0),
            streaming=True,
        )
    raise ValueError(f"Provider {llm_provider} not recognized")


def get_system_prompt(config):
    return config.get("prompt", {}).get(
        "system_prompt",
        (
            "You are Hoopoe, a professional and thoughtful assistant. Speak in a "
            "natural, human way: clear, concise, and direct. Use uploaded documents "
            "as your primary source of truth, and say what is missing when the "
            "documents do not contain enough information."
        ),
    )


def build_chat_chain(config):
    memory_enabled = config.get("memory", {}).get("enabled", True)
    llm = get_chat_llm(config)
    system_prompt = get_system_prompt(config)

    prompt = ChatPromptTemplate.from_template(
        """
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
    )

    def memory_summary(input_data):
        if not memory_enabled:
            return "Memory is disabled."
        return get_memory_summary(input_data["session_id"])

    def recent_memory(input_data):
        if not memory_enabled:
            return "Memory is disabled."
        return get_recent_context(input_data["session_id"])

    return (
        RunnableParallel(
            {
                "question": lambda x: x["question"],
                "system_prompt": lambda _: system_prompt,
                "memory_summary": RunnableLambda(memory_summary),
                "recent_memory": RunnableLambda(recent_memory),
            }
        )
        | prompt
        | llm
    )


def build_rag_chain(retriever, config):
    memory_enabled = config.get("memory", {}).get("enabled", True)
    llm = get_chat_llm(config)
    system_prompt = get_system_prompt(config)

    prompt = ChatPromptTemplate.from_template(
        """
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

Answer the user using the retrieved context first. If the context is not enough,
say what is missing instead of inventing details.
"""
    )

    def combine_docs(docs):
        if isinstance(docs, dict):
            docs = docs.get("context", [])
        if not docs:
            return "No relevant context found."
        return "\n\n".join(doc.page_content for doc in docs)

    def memory_summary(input_data):
        if not memory_enabled:
            return "Memory is disabled."
        return get_memory_summary(input_data["session_id"])

    def recent_memory(input_data):
        if not memory_enabled:
            return "Memory is disabled."
        return get_recent_context(input_data["session_id"])

    return (
        RunnableParallel(
            {
                "context": (lambda x: x["question"]) | retriever | combine_docs,
                "question": lambda x: x["question"],
                "system_prompt": lambda _: system_prompt,
                "memory_summary": RunnableLambda(memory_summary),
                "recent_memory": RunnableLambda(recent_memory),
            }
        )
        | prompt
        | llm
    )
