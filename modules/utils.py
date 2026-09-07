import os
import logging
import yaml
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# load env
env_path = os.path.join(os.path.dirname(__file__), "..", "config", ".env")
env_path = os.path.abspath(env_path)
load_dotenv(dotenv_path=env_path)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Tracks which logger names already have handlers attached, so calling
# get_logger() again for the same name (e.g. on every Streamlit rerun)
# never creates duplicate log lines.
_CONFIGURED_LOGGERS = set()


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load the YAML config file, resolved relative to the project root.

    Raises FileNotFoundError if the file is missing, or yaml.YAMLError with a
    clear message if it can't be parsed.
    """
    full_path = os.path.join(PROJECT_ROOT, config_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Config file not found at {full_path}")
    with open(full_path, "r", encoding="utf-8") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse config file at {full_path}: {e}") from e


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to the configured log file and the console.

    Log level and log file path come from config.yaml's `logging` section.
    Handlers are attached only once per logger name so repeated calls (e.g.
    Streamlit reruns) don't duplicate log lines.
    """
    logger = logging.getLogger(name)

    if name in _CONFIGURED_LOGGERS:
        return logger

    config = load_config()
    log_config = config.get("logging", {})
    level = getattr(logging, str(log_config.get("level", "INFO")).upper(), logging.INFO)
    log_file = log_config.get("log_file", "logs/hoopoe.log")
    log_file_path = os.path.join(PROJECT_ROOT, log_file)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.setLevel(level)
    logger.propagate = False

    _CONFIGURED_LOGGERS.add(name)
    return logger


logger = get_logger(__name__)


def get_embedding_model(config: dict, provider: str):
    provider = provider.lower()
    embedding_config = config.get("embedding", {})
    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key must be provided. Set OPENAI_API_KEY in config/.env."
            )
        model_name = embedding_config.get("openai_model", "text-embedding-3-small")
        logger.info("Using OpenAI embedding model: %s", model_name)
        return OpenAIEmbeddings(api_key=api_key, model=model_name)
    elif provider in ("gemini", "google"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key must be provided. Set GEMINI_API_KEY in config/.env."
            )
        model_name = embedding_config.get("gemini_model", "models/embedding-001")
        logger.info("Using Gemini embedding model: %s", model_name)
        return GoogleGenerativeAIEmbeddings(google_api_key=api_key, model=model_name)
    else:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. Expected 'openai' or 'gemini'."
        )
