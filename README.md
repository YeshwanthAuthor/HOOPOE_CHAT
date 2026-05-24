# Hoopoe RAG Chatbot

Hoopoe is a document-aware chatbot built with Streamlit, LangChain, FAISS, and OpenAI/Gemini models. It lets you upload a PDF, DOCX, or TXT file, index the document into a local vector store, and ask questions grounded in the uploaded content.

## Features

- Streamlit chat interface with multiple chat sessions
- PDF, DOCX, and TXT document upload
- Local FAISS vector store per chat session
- OpenAI and Gemini chat model support
- OpenAI and Gemini embedding support
- Configurable retriever, chunking, prompt, and memory settings
- Short-term conversation memory with optional summary memory

## Project Structure

```text
RAG_CHATBOT/
+-- app/
|   +-- streamlit_app.py      # Main Streamlit application
|   +-- gradio_app.py         # Placeholder for a Gradio UI
|   +-- assets/               # App logo and favicon assets
+-- config/
|   +-- config.json           # Runtime configuration
|   +-- .env                  # Local API keys, not for source control
+-- data/
|   +-- uploads/              # Uploaded files, grouped by chat session
+-- modules/
|   +-- embedder.py           # Embedding model and FAISS index handling
|   +-- loader.py             # PDF/DOCX/TXT loaders
|   +-- memory.py             # Conversation memory helpers
|   +-- rag_chain.py          # Chat and RAG chains
|   +-- retriever.py          # Base and multi-query retrievers
|   +-- splitter.py           # Document chunking
|   +-- utils.py              # Provider utilities
+-- vectorstores/             # Persisted FAISS indexes
+-- requirements.txt
+-- README.md
```

## Requirements

- Python 3.11 recommended
- An OpenAI API key for OpenAI chat, OpenAI embeddings, and summary memory
- A Gemini API key if using Gemini chat or Gemini embeddings

## Setup

Create and activate a virtual environment:

```powershell
python -m venv myenv
.\myenv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create `config/.env` with the API keys you plan to use:

```env
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
```

Do not commit real API keys. If this repository has ever been shared with keys inside `config/.env`, rotate those keys.

## Configuration

Main settings live in `config/config.json`.

Important fields:

- `llm_provider`: Chat model provider. Supported values are `openai`, `gemini`, or `google`.
- `embedding_provider`: Embedding provider. Supported values are `openai`, `gemini`, or `google`.
- `vector_store.type`: Currently supports `faiss`.
- `retriever.type`: Use `base` or `multiquery`.
- `retriever.top_k`: Number of chunks returned during retrieval.
- `prompt.system_prompt`: System behavior for Hoopoe.
- `prompt.temperature`: Chat model temperature.
- `memory.enabled_default`: Whether memory starts enabled in the UI.
- `memory.max_memory_window`: Number of recent turns retained.
- `chunking.chunk_size`: Character size for document chunks.
- `chunking.chunk_overlap`: Character overlap between chunks.

Example:

```json
{
  "llm_provider": "openai",
  "embedding_provider": "openai",
  "vector_store": {
    "type": "faiss",
    "local": {
      "faiss_path": "vectorstores/faiss_index"
    }
  },
  "retriever": {
    "type": "base",
    "top_k": 3
  }
}
```

## Run the App

Start the Streamlit app from the project root:

```powershell
streamlit run app/streamlit_app.py
```

Streamlit will print a local URL, usually:

```text
http://localhost:8501
```

## Deploy on Render

This project includes `.python-version` with Python `3.11` for Render. Render's current default Python can be newer than some ML/LangChain packages support, so keep this file in the repository.

Use this Render start command:

```bash
streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

Set these environment variables in Render:

```text
OPENAI_API_KEY
GEMINI_API_KEY
```

The app uses local FAISS indexes. On Render's free/ephemeral filesystem, uploaded files and vector indexes can disappear after restarts unless you attach persistent storage.

## Usage

1. Open the Streamlit app in your browser.
2. Choose the LLM provider from the sidebar.
3. Toggle memory on or off.
4. Upload a PDF, DOCX, or TXT file.
5. Click `Index Document`.
6. Ask questions in the chat input.

If no document is indexed, Hoopoe behaves like a regular chat assistant. Once a document is indexed, responses are generated from retrieved document chunks first.

## Data and Indexes

Uploaded documents are saved under:

```text
data/uploads/<chat_id>/
```

FAISS indexes are saved under:

```text
vectorstores/<chat_id>/<embedding_provider>/
```

These folders can become large and may contain private documents. Keep them out of source control unless you intentionally want to version them.

## Troubleshooting

### `JSONDecodeError` while loading config

Validate the config file:

```powershell
python -m json.tool .\config\config.json
```

JSON requires double-quoted property names, no comments, and no trailing commas.

### `OpenAI API key must be provided`

Add `OPENAI_API_KEY` to `config/.env`, then restart Streamlit.

### `Gemini API key must be provided`

Add `GEMINI_API_KEY` to `config/.env`, then restart Streamlit.

### `No readable text was found in the uploaded document`

The file may be scanned, image-only, encrypted, or otherwise not readable by the current loaders. Try a text-based PDF, DOCX, or TXT file.

## Notes

- The current production entrypoint is `app/streamlit_app.py`.
- `app/gradio_app.py` exists but does not currently implement a Gradio UI.
- Summary memory currently uses OpenAI, so memory summaries require `OPENAI_API_KEY` even when Gemini is selected for chat.
