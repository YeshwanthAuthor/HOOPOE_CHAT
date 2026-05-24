import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

#load env
env_path = os.path.join(os.path.dirname(__file__), "..", "config", ".env")
env_path = os.path.abspath(env_path)
load_dotenv(dotenv_path=env_path)

#load embeddings
def get_embedding_model(provider: str):
    provider = provider.lower()
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided")
        return OpenAIEmbeddings(api_key=api_key)
    elif provider in ("gemini", "google"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API key must be provided")
        return GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/embedding-001")
    else:
        raise ValueError("Unknown provider {}".format(provider))


