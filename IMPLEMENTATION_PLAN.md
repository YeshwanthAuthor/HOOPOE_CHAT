# Hoopoe RAG Chatbot — Module-by-Module Implementation Plan

This plan exists because the real implementation work will happen in separate,
token-limited Claude sessions — **one module per session**. Give a fresh
Claude session this whole file plus the module number you want, e.g.:

> "Implement Module 4 from IMPLEMENTATION_PLAN.md. Here is the current state
> of modules/retriever.py: ..."

Each module below is self-contained, touches a small number of files, and
ends with a "Test locally" step you can run before moving to the next module.
Do not start a module until the previous one passes its local test.

---

## 0. Ground truth: what already exists (read this first)

The repo is `RAG_CHATBOT` (product name "Hoopoe"). A working pilot already
exists — **reuse it, do not rewrite from scratch.**

```
RAG_CHATBOT/
├── app/
│   ├── streamlit_app.py     # Full chat UI, multi-session, upload+index flow
│   ├── gradio_app.py        # Empty placeholder — ignore, not in scope
│   └── assets/              # Logo/favicon images, already wired into the UI
├── config/
│   ├── config.json          # Runtime config (will become config.yaml — Module 0)
│   └── .env                 # OPENAI_API_KEY, GEMINI_API_KEY
├── data/
│   ├── sample_docs/         # ✅ 8 sample HR docs already added (see Module 1)
│   └── uploads/<chat_id>/   # Per-chat uploaded files, created at runtime
├── modules/
│   ├── __init__.py          # empty
│   ├── utils.py             # get_embedding_model(provider) — OpenAI/Gemini
│   ├── loader.py            # load_document(path) — .txt/.docx/.pdf → List[Document]
│   ├── splitter.py          # split_doc(docs, chunk_size, chunk_overlap)
│   ├── embedder.py          # build_or_update_vectorstore(docs, config, persist_dir)
│   ├── retriever.py         # get_retriever(vectorstore, config) — "base" or "multiquery"
│   ├── memory.py            # recent-turn memory + LangChain ConversationSummaryMemory
│   └── rag_chain.py         # build_chat_chain / build_rag_chain (LCEL chains)
├── vectorstores/<chat_id>/<provider>/   # FAISS index.faiss + index.pkl, per chat
└── requirements.txt
```

**Existing functional coverage vs. the assignment:**

| Requirement | Status |
|---|---|
| 1. Document ingestion (pdf/docx, chunking, embeddings, FAISS) | ✅ Done — reuse as-is |
| 2. Basic vector retrieval | ✅ Done — reuse as-is |
| 3. Hybrid search (vector + BM25) | ❌ Not implemented — **Modules 4–5** |
| 4. Cross-encoder reranking | ❌ Not implemented — **Module 6** |
| 5. Conversational memory (recent-n + summary) | ⚠️ Half-wired — **bug found, fixed in Module 8** |
| 6. Numbered `[1]…[n]` citations tied to source docs | ⚠️ Wrong format today — **Module 7** |
| 7. Streamlit UI (chat, history, clear, sources, errors) | ⚠️ Missing sources display + clear button — **Module 9** |
| 8. Hallucination handling | ⚠️ Prompt-only today, no grounding check — **Module 7** |
| No hardcoded values / config-driven | ⚠️ Several hardcoded model names — **Module 0** |
| Logging with meaningful exceptions | ⚠️ Uses `print()`, no logger — **Module 0 + Module 10** |

**A second reference project was reviewed:** `ARAG_CROSS_ENCODER_RR` (a separate small course-material project on the same machine, not part of Hoopoe). It has no UI, no memory, and no hybrid search — just a console `app.py` demonstrating FAISS retrieval → cross-encoder reranking → LLM answer. But its code style is exactly the bar Hoopoe's new modules should match: plain functions over classes wherever possible, a class only when something needs to hold state across calls (a loaded model, an open index), `config.yaml` read with a 3-line `load_config()`, one clear function per responsibility, and a single `.invoke(query)` entry point that hides the multi-stage pipeline from the caller. Modules 3, 5, 6, and 7 below are written to mirror this project's actual function names and structure (`get_index_path`, `_index_path_exists`, `create_retriever`, `load_retriever`, `get_retriever(config, ..., chunks_if_needed=None)`, a `CrossEncoderReranker` class, a two-stage `.invoke()` wrapper) rather than reinventing an equivalent shape from scratch. The one deliberate difference: Hoopoe is a multi-chat-session app, so every retriever/index function additionally takes a `chat_id` — the reference project only ever has one global index because it has no concept of separate chats.

**Known bug to fix (found during review, fix in Module 8):** `memory.py`
defines `add_memory_to_chain()`, which is the *only* place that actually
calls `summary_memory.save_context()` and refreshes the running summary.
`streamlit_app.py` never calls `add_memory_to_chain()` — it calls
`build_rag_chain`/`build_chat_chain` directly and only calls
`update_recent_memory()` itself. **Net effect: the "last n messages" memory
works, but the conversation *summary* memory is currently frozen/empty in
the running app.** This must be fixed for requirement 5 to actually work.

---

## Decisions locked in (do not re-litigate these)

1. **Restore/base code**: the pilot app above is the base. Enhance it in place; do not restructure the folder layout.
2. **Sample docs**: 8 one-page docs already created in `data/sample_docs/` — 3 as `.docx` (`leave_policy.docx`, `hr_handbook.docx`, `code_of_conduct.docx`), 5 as `.pdf` (`employee_policies.pdf`, `it_policies.pdf`, `travel_policy.pdf`, `benefits_documentation.pdf`, `company_faqs.pdf`). Fictional company: "K International Pvt. Ltd." Module 1 is **done** — no action needed, just point ingestion at this folder when testing.
3. **LLM/embedding scope**: keep **both** OpenAI and Gemini fully supported everywhere (provider stays a config switch, same as today).
4. **Reranking**: local, free **cross-encoder** via `sentence-transformers` (no hosted API, no new API key). Config:
   ```yaml
   reranking:
     enabled: true
     initial_k: 30
     final_k: 5
     method: "cross-encoder"
     cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L6-v2"
     max_chunk_chars: 1200
     min_relevance_score: -2.0
   ```
   Note: verified against the Hugging Face model card — the id has **no dash before "6"** (`MiniLM-L6-v2`, not `L-6-v2`). An earlier draft of this plan had that backwards; this is the corrected, confirmed id, and it matches the `ARAG_CROSS_ENCODER_RR` reference project's `config.yaml` exactly.
5. **Hybrid search**: use LangChain's own `BM25Retriever` (`langchain_community.retrievers`) combined with the FAISS retriever via LangChain's `EnsembleRetriever` — no hand-written score fusion. New dependency: `rank-bm25` (PyPI package name uses a hyphen; the reference project's `requirements.txt` confirms this).
6. **Config format**: migrate `config/config.json` → `config/config.yaml` (new dependency: `PyYAML`, already pulled in transitively by other packages but pin it explicitly). All model names, thresholds, and paths move into it — no hardcoded values in `.py` files.

---

## Module 0 — Config Migration, Dependencies & Logging

**Goal:** one place for every tunable value, and a real logger everywhere `print()` is used today.

**Files:**
- `config/config.yaml` (new — replaces `config/config.json`)
- `modules/utils.py` (edit — add `load_config()` and `get_logger()`)
- `requirements.txt` (edit — add new deps)

**Full `config/config.yaml` to create:**
```yaml
company_name: "K International Pvt. Ltd."

llm:
  provider: "openai"                 # "openai" or "gemini"
  openai_model: "gpt-4o-mini"
  gemini_model: "gemini-2.5-flash"

embedding:
  provider: "openai"                 # "openai" or "gemini"
  openai_model: "text-embedding-3-small"
  gemini_model: "models/embedding-001"

vector_store:
  type: "faiss"
  base_path: "vectorstores"
  index_name: "index"

chunking:
  chunk_size: 1000
  chunk_overlap: 200

retrieval:
  hybrid_enabled: true
  vector_weight: 0.6
  bm25_weight: 0.4

reranking:
  enabled: true
  initial_k: 30
  final_k: 5
  method: "cross-encoder"
  cross_encoder_model: "cross-encoder/ms-marco-MiniLM-L6-v2"
  max_chunk_chars: 1200
  min_relevance_score: -2.0

prompt:
  system_prompt: >
    You are Hoopoe, a professional and thoughtful assistant. Use the uploaded
    documents as your primary source of truth. Cite every fact you use with
    the bracket number of the chunk it came from, like [1] or [2]. If the
    documents do not contain enough information, say so clearly instead of
    guessing.
  temperature: 0.1
  max_tokens: 900

memory:
  enabled_default: true
  max_recent_turns: 5

upload:
  allowed_extensions: [".pdf", ".docx", ".txt"]
  max_size_mb: 2

logging:
  level: "INFO"
  log_file: "logs/hoopoe.log"
```
Drop the old `vector_store.cloud` (Pinecone) block from `config.json` — nothing reads it, it's dead config.

**Add to `modules/utils.py`** — keep this as simple as the reference project's own `load_config()` (it opens the file and calls `yaml.safe_load`, nothing more); the only reason it lives once in `utils.py` instead of being copy-pasted into every module (which is what `ARAG_CROSS_ENCODER_RR` does) is so five copies can't drift out of sync — everything else about it stays just as small:
```python
def load_config(config_path: str = "config/config.yaml") -> dict:
    """Resolve config_path relative to the project root (same pattern as
    ARAG_CROSS_ENCODER_RR's utils/*.py: os.path.dirname(__file__) + '..'),
    then yaml.safe_load it. Raises FileNotFoundError / yaml.YAMLError with a clear message."""

def get_logger(name: str) -> logging.Logger:
    """Return a logger writing to logging.log_file (from config) and the console,
    configured once per process (avoid duplicate handlers on Streamlit reruns)."""
```

**`requirements.txt` additions:**
```
rank-bm25==0.2.2
sentence-transformers==3.1.1
PyYAML==6.0.2
```

**Test locally:**
```bash
python -c "from modules.utils import load_config, get_logger; c = load_config(); log = get_logger('test'); log.info('config keys: %s', list(c.keys()))"
```
Confirm it prints all top-level config sections and a line appears in `logs/hoopoe.log`.

**Definition of done:** `config.yaml` exists with the schema above, `config.json` is removed, `load_config()`/`get_logger()` work, no other module has changed yet.

---

## Module 1 — Sample HR Documents ✅ Already done

8 one-page documents already exist in `data/sample_docs/`. Nothing to implement. Use these files for manual testing of every later module (leave policy carry-forward, IT password rules, etc. are good test questions).

---

## Module 2 — Document Ingestion Verification

**Goal:** confirm `loader.py` + `splitter.py` work correctly against the real sample docs and clean up citation metadata for later modules.

**Files:** `modules/loader.py` (small edit)

**Change:** `text_to_doc()` currently stores the full absolute file path (plus `:page:N` for PDFs) in `metadata["source"]`. Add a second metadata field `metadata["filename"]` holding just the human-readable base filename (e.g. `leave_policy.docx`), stripped of any `:page:N` suffix. Keep `source` as-is (still useful for debugging) — citations in Module 7 will use `filename`.

```python
def text_to_doc(text: str, source: str, filename: str) -> Document:
    return Document(page_content=text, metadata={"source": source, "filename": filename})
```
Update `load_txt`, `load_docx`, `load_pdf` to pass `filename=os.path.basename(path)`.

**Test locally:**
```python
from modules.loader import load_document
docs = load_document("data/sample_docs/leave_policy.docx")
print(docs[0].metadata)   # expect {'source': '...', 'filename': 'leave_policy.docx'}

from modules.splitter import split_doc
chunks = split_doc(docs, chunk_size=1000, chunk_overlap=200)
print(len(chunks), chunks[0].metadata)
```
Repeat for one `.pdf` in the folder. Confirm `filename` is clean and chunk count looks reasonable (a 1-page doc should be 1–3 chunks at chunk_size=1000).

**Definition of done:** every loaded chunk has a clean `filename` field; no other behavior changed.

---

## Module 3 — Embeddings & FAISS Vector Store (rebuilt to mirror `retriever_faiss.py`)

**Goal:** replace `embedder.py`'s current single `build_or_update_vectorstore` function with the same small function set as `ARAG_CROSS_ENCODER_RR/utils/retriever_faiss.py` — `get_embedding_model`, `_index_path_exists`, `create_retriever`, `load_retriever`, `get_retriever` — each doing exactly one thing, with `get_retriever` as the one function everything else calls. The only change from that reference: every path/index function also takes `chat_id`, because Hoopoe keeps one FAISS index per chat session, not one global index per provider.

> A single global index per provider (exactly like the reference project) was considered and **rejected** for Hoopoe: it would break the existing multi-chat-session UI, where each chat has its own uploaded document and its own index. The per-chat, per-provider layout (`vectorstores/<chat_id>/<provider>/`) stays — only the *internal* function structure is being aligned to the reference.

**Files:** `modules/embedder.py` (rewrite), `modules/utils.py` (edit)

**Changes to `utils.get_embedding_model`:** take the model name from config instead of hardcoding `"models/embedding-001"` (same signature shape as the reference's `get_embedding_model(config, provider)`):
```python
def get_embedding_model(config: dict, provider: str):
    # config["embedding"]["openai_model"] / config["embedding"]["gemini_model"]
```

**Rewrite `modules/embedder.py`** as this function set (`INDEX_NAME = config["vector_store"]["index_name"]`, e.g. `"index"`):
```python
def get_index_path(config: dict, chat_id: str, provider: str) -> str:
    """os.path.join(config['vector_store']['base_path'], chat_id, provider)"""

def _index_path_exists(index_path: str, index_name: str) -> bool:
    """True if both <index_name>.faiss and <index_name>.pkl exist — identical logic
    to the reference project's function of the same name."""

def create_retriever(chunks, chat_id: str, provider: str, config: dict):
    """Embed chunks, FAISS.from_documents(...), save_local(index_path, index_name=...),
    return vectorstore.as_retriever(...). Logs via get_logger instead of print()."""

def load_retriever(chat_id: str, provider: str, config: dict):
    """FAISS.load_local(index_path, model, index_name=..., allow_dangerous_deserialization=True)
    -> .as_retriever(...)."""

def get_retriever(config: dict, chat_id: str, provider: str, chunks_if_needed=None):
    """Same self-healing logic as the reference's get_retriever(): if no index exists,
    build one from chunks_if_needed (raise a clear error if none given); if loading an
    existing index fails (corrupted/incompatible), log a warning and rebuild from
    chunks_if_needed if given, else raise a clear RuntimeError telling the user to
    delete the folder and re-index."""
```
`streamlit_app.py`'s `index_document()` (Module 9) calls `get_retriever(config, chat_id, provider, chunks_if_needed=chunks)` — one call replaces what used to be a manual `build_or_update_vectorstore` call.

**Test locally:**
```python
from modules.utils import load_config
from modules.loader import load_document
from modules.splitter import split_doc
from modules.embedder import get_retriever

config = load_config()
docs = split_doc(load_document("data/sample_docs/leave_policy.docx"), 1000, 200)
retriever = get_retriever(config, chat_id="test_chat", provider="openai", chunks_if_needed=docs)
print(retriever.invoke("carry forward"))
```
Confirm `vectorstores/test_chat/openai/` now has `index.faiss`/`index.pkl`, and the search returns the carry-forward bullet point. Re-run the same script a second time with `chunks_if_needed=None` — it should load the existing index instead of failing.

**Definition of done:** indexing/loading works exactly as before functionally, model name and index name come from config, a corrupted index no longer crashes the app with a raw stack trace, and the function names/shapes match the reference project's pattern (adjusted for `chat_id`).

---

## Module 4 — BM25 Keyword Retriever

**Goal:** add the keyword half of hybrid search, working on the *same chunks* used for the FAISS index.

**Files:** new `modules/bm25_retriever.py`

```python
def build_bm25_retriever(chunks: list[Document], config: dict):
    """Build a langchain_community.retrievers.BM25Retriever from the same
    chunk Documents that went into FAISS. k comes from config["reranking"]["initial_k"]
    if hybrid retrieval feeds the reranker, else a sane default."""
```
Since `BM25Retriever` has no persistent save/load (it's in-memory only), it must be rebuilt from the same chunk list every time a document is (re)indexed — store it in Streamlit session state alongside the FAISS retriever (wiring happens in Module 9).

**Test locally:**
```python
from modules.loader import load_document
from modules.splitter import split_doc
from modules.bm25_retriever import build_bm25_retriever

chunks = split_doc(load_document("data/sample_docs/it_policies.pdf"), 1000, 200)
bm25 = build_bm25_retriever(chunks, config)
results = bm25.get_relevant_documents("password length")
print([d.page_content[:80] for d in results])
```
Confirm the password-length bullet comes back near the top (BM25 is strong on exact keyword matches like "12 characters").

**Definition of done:** BM25 retriever returns sensible keyword-matched results on its own, independent of FAISS.

---

## Module 5 — Hybrid Retrieval (Ensemble)

**Goal:** combine vector search (Module 3) and BM25 (Module 4) into one retriever.

**Files:** `modules/retriever.py` (edit)

**Change:** add a new function alongside the existing `get_retriever` (the "multiquery" one, unchanged — not to be confused with `embedder.get_retriever` from Module 3, which builds the FAISS half):
```python
def get_hybrid_retriever(faiss_retriever, bm25_retriever, config: dict):
    """EnsembleRetriever([faiss_retriever, bm25_retriever],
    weights=[config['retrieval']['vector_weight'], config['retrieval']['bm25_weight']]).
    Both underlying retrievers should use k = config['reranking']['initial_k']
    when reranking is enabled, else config's own top_k."""
```
`faiss_retriever` here is what `embedder.get_retriever(...)` from Module 3 returns (already a LangChain retriever, not a raw vectorstore). Keep the existing `retriever.get_retriever` function untouched (it's still used for the "multiquery" retriever type and as a fallback if `retrieval.hybrid_enabled: false`).

**Test locally:**
```python
hybrid = get_hybrid_retriever(faiss_retriever, bm25, config)
for d in hybrid.get_relevant_documents("what happens if I lose my laptop"):
    print(d.metadata["filename"], "-", d.page_content[:80])
```
Ask one question that's better for keyword search (exact term, e.g. "MFA") and one better for semantic search (paraphrased, e.g. "how do I keep my account safe") — confirm both return relevant chunks from `it_policies.pdf`.

**Definition of done:** hybrid retriever returns a merged, deduplicated candidate list from both retrievers.

---

## Module 6 — Cross-Encoder Reranking

**Goal:** re-score the hybrid retriever's candidates with a cross-encoder and keep only the best `final_k`. This mirrors `ARAG_CROSS_ENCODER_RR/utils/reranker_cross_encoder.py` almost line for line — it's already exactly the right shape and simplicity for this job, so there's no reason to redesign it.

**Files:** new `modules/reranker.py`

```python
class CrossEncoderReranker:
    """Loads the cross-encoder model once (in __init__), reuses it for every call —
    this is why it's a class and not a plain function: constructing a CrossEncoder
    per query would reload the model from disk every time."""

    def __init__(self, config: dict):
        self.final_k = config["reranking"]["final_k"]
        self.max_chars = config["reranking"]["max_chunk_chars"]
        self.model = CrossEncoder(config["reranking"]["cross_encoder_model"])

    def rerank(self, query: str, docs: list[Document], top_k: int | None = None) -> list[tuple[Document, float]]:
        """
        1. Truncate each candidate's page_content to self.max_chars before scoring
           (cross-encoders are slow on long text).
        2. Score all (query, chunk_text) pairs with self.model.predict(...).
        3. Sort descending by score, return the top (top_k or self.final_k) as
           (Document, score) tuples — caller decides whether it wants the scores.
        """
```
If `config["reranking"]["enabled"]` is `false`, the caller (Module 7) should skip this class entirely and just take the top `final_k` from the hybrid retriever directly — keep that branching in `rag_chain.py`, not inside `reranker.py`.

**Test locally:**
```python
from modules.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker(config)
candidates = hybrid.get_relevant_documents("carry forward leave")  # from Module 5, initial_k=30
ranked = reranker.rerank("carry forward leave", candidates)
for doc, score in ranked:
    print(round(score, 3), doc.metadata["filename"], doc.page_content[:60])
```
Confirm the leave-policy carry-forward chunk is now ranked #1, and scores are descending.

**Definition of done:** reranker measurably reorders candidates toward the actually-relevant chunk; first run downloads the cross-encoder model (~90MB) once.

---

## Module 7 — Citation-Aware RAG Chain & Hallucination Guard

**Goal:** this is the module that satisfies requirement 6 (numbered citations) and requirement 8 (hallucination handling) properly. It replaces the fragile regex-based `format_sources_on_separate_line` approach in `streamlit_app.py` with a structured return value.

**Files:** `modules/rag_chain.py` (edit — the core rewrite)

**Composition point:** this is also where hybrid retrieval (Module 5) and reranking (Module 6) meet, the same way `ARAG_CROSS_ENCODER_RR/utils/two_stage_retriever.py`'s `TwoStageRetriever` class wraps its FAISS retriever + `CrossEncoderReranker` behind one `.invoke(query)` call. Add a small equivalent here so `streamlit_app.py` never has to orchestrate multiple retrieval calls itself:
```python
class RetrievalPipeline:
    """Stage 1: hybrid retriever (Module 5) → Stage 2: CrossEncoderReranker (Module 6).
    One entry point: invoke(query) -> list[(Document, score)], honoring
    config['reranking']['enabled'] (skip stage 2 and return hybrid results, score=None,
    when reranking is off)."""

    def __init__(self, hybrid_retriever, reranker: CrossEncoderReranker | None, config: dict): ...

    def invoke(self, query: str) -> list[tuple[Document, float | None]]: ...
```

**Signature change:** `build_rag_chain` currently takes a `retriever` and calls it *inside* the LCEL chain (`(lambda x: x["question"]) | retriever | combine_docs`). Reranking doesn't fit that shape — it needs the retrieved docs *and* their scores, not just a `Document` list — so retrieval+reranking now happens *before* the chain is built, via `RetrievalPipeline.invoke(query)` (see the composition point above). Change the signature to `build_rag_chain(retrieved: list[tuple[Document, float | None]], config: dict)`, called fresh per question from `get_bot_reply` (Module 9) with that question's own `RetrievalPipeline.invoke(...)` result. `combine_docs` now just formats the `docs` half of `retrieved` — it no longer needs to be piped after a retriever.

**Change `combine_docs`:** number the *final* reranked/retrieved chunks `[1]…[n]` (n = however many chunks made it into context, typically `final_k`) and show the filename per number:
```python
def combine_docs(docs):
    # for i, doc in enumerate(docs, start=1):
    #     f"[{i}] Source: {doc.metadata['filename']}\n{doc.page_content}"
```

**Change the prompt** to explicitly instruct: *"Cite the bracket number, like [1], immediately after any fact you take from the context. Only use numbers that appear in the Retrieved Context above. If the context does not answer the question, say the information was not found in the uploaded documents — do not guess."*

**Change the return contract.** Instead of returning a raw LangChain message, have `build_rag_chain` (and `build_chat_chain`, with an empty citations list) return a small dict the UI can render deterministically:
```python
{
    "answer": "<LLM text, containing inline [n] citations>",
    "citations": [{"n": 1, "filename": "leave_policy.docx"}, {"n": 2, "filename": "hr_handbook.docx"}],
}
```
Build `citations` in code from the same `docs` list used for `combine_docs` — do **not** rely on parsing the LLM's text to find sources. This is what makes the citation grounding reliable: the numbers the LLM sees and the numbers in the Sources list are guaranteed to refer to the same chunks, because both come from the same enumeration in `combine_docs`.

**Hallucination guard:** `RetrievalPipeline.invoke(query)` returns `[(doc, score), ...]`. If reranking is enabled and every score in that list is below `config["reranking"]["min_relevance_score"]`, skip calling the LLM with that context entirely and return the standard "not found in the uploaded documents" message directly (saves an LLM call and guarantees the exact wording requirement 8 asks for). `combine_docs` and the citation numbering both work off `[doc for doc, score in results]` — the scores exist for this guard, not for anything shown to the LLM.

**Test locally:**
```python
retrieved = pipeline.invoke("What is the carry-forward limit for leave?")  # [(doc, score), ...] from Module 7's RetrievalPipeline
rag_chain = build_rag_chain(retrieved, config)
result = rag_chain.invoke({"question": "What is the carry-forward limit for leave?", "session_id": "test"})
print(result["answer"])
print(result["citations"])
```
Confirm the answer contains `[1]` (or whichever number matches `leave_policy.docx`), and `citations` lists that filename. Then ask an off-topic question ("what's the weather today") and confirm you get the "not found" message, not a guess.

**Definition of done:** every RAG answer that uses document content contains at least one `[n]`, every `[n]` in the answer has a matching entry in `citations`, and a no-context question produces the standard "not found" response instead of a hallucination.

---

## Module 8 — Memory: Fix the Summary-Update Bug

**Goal:** actually fix the bug found in the review (Section 0) — summary memory currently never updates in the live app.

**Files:** `modules/memory.py` (edit), `modules/rag_chain.py` (small edit)

**Change:** move the "save to summary memory + refresh summary" logic that already exists inside `add_memory_to_chain`'s `chain_with_summary` closure into a small standalone function that the caller can invoke explicitly after getting a response, instead of only inside the unused `RunnableWithMessageHistory` wrapper:
```python
def record_turn(session_id: str, user_input: str, ai_output: str, config: dict) -> None:
    """Update BOTH recent-turn memory and summary memory for this turn.
    Reads config['memory']['max_recent_turns']. Replaces the app's separate
    update_recent_memory() call — this becomes the single entry point."""
```
Have `streamlit_app.py`'s `get_bot_reply` (Module 9 will touch this file) call `record_turn(...)` once per turn instead of calling `update_recent_memory` directly. Keep `add_memory_to_chain` if you like, but it's no longer required for correctness once `record_turn` exists — note in a comment that it's currently unused, don't delete it without checking nothing else references it.

**Test locally:**
```python
from modules.memory import record_turn, get_memory_summary, get_recent_context
record_turn("t1", "What is the leave policy?", "Employees get 8 CL, 8 SL, 15 EL... [1]", config)
record_turn("t1", "What about carry-forward?", "Up to 10 days of EL carry forward... [1]", config)
print(get_memory_summary("t1"))
print(get_recent_context("t1"))
```
Confirm `get_memory_summary("t1")` returns a *non-empty, updated* summary after two turns (this is the part that's broken today) and `get_recent_context` shows both turns.

**Definition of done:** summary memory visibly changes after each turn; a follow-up question like "What about carry-forward?" resolves correctly using context from the previous turn (test this end-to-end once Module 9's UI is wired up).

---

## Module 9 — Streamlit UI Integration

**Goal:** wire Modules 3–8 into the running app, and close the two UI gaps from the requirements: displaying sources, and a real "clear conversation" control.

**Files:** `app/streamlit_app.py` (edit)

**Wiring changes:**
- `read_config()` now loads `config/config.yaml` via `modules.utils.load_config()` instead of `json.load` on `config.json`.
- `index_document()`: after `get_retriever(...)` (Module 3) builds the FAISS retriever, also call `build_bm25_retriever` (Module 4) on the same `chunks`, combine them with `get_hybrid_retriever` (Module 5), and wrap that plus a `CrossEncoderReranker` in one `RetrievalPipeline` (Module 7). Store just that single pipeline object in `st.session_state.documents[chat_id]["pipeline"]` — this is the one piece of state `get_bot_reply` needs, instead of juggling three separate retriever objects.
- `get_bot_reply()`: call `pipeline.invoke(user_message)` to get `[(doc, score), ...]`, pass it into `build_rag_chain` (Module 7's rewritten version), and switch to the new dict return contract (`result["answer"]`, `result["citations"]`). Replace the `update_recent_memory` call with `record_turn` (Module 8).
- Remove `format_sources_on_separate_line` — no longer needed, citations are structured now.

**New UI pieces:**
- **Sources display:** under each assistant message, if `citations` is non-empty, render a small `st.caption` or expander: `"Sources: leave_policy.docx, hr_handbook.docx"` (deduplicate filenames, keep bracket-number order). Store `citations` alongside each message in `st.session_state.messages[chat_id]` so history re-renders correctly on rerun.
- **Clear/reset button:** add a "Clear Conversation" button next to "New Chat" in the sidebar. It resets `st.session_state.messages[chat_id]` to just the greeting and clears that chat's memory (`SESSION_MESSAGE_HISTORY`, `SESSION_MESSAGE_SUMMARY`, `SESSION_RECENT_HISTORY` entries for that `chat_id` in `memory.py` — add a small `clear_session_memory(session_id)` helper to `memory.py` for this), but **keeps the indexed document** (unlike "New Chat", which starts a document-less session too).
- **Error messages:** wrap `index_document` and `get_bot_reply` calls in try/except that log via `get_logger` and show `st.error(str(error))` with a user-friendly message (already partly done for indexing — extend the same pattern to `get_bot_reply`, which currently only does `bot_reply = f"Error: {error}"` inline).

**Test locally:**
```bash
streamlit run app/streamlit_app.py
```
Manual test script:
1. Upload `data/sample_docs/leave_policy.docx`, click Index Document.
2. Ask "What is the leave policy?" — confirm the answer cites `[1]` and a Sources caption shows `leave_policy.docx`.
3. Ask "What about carry-forward?" — confirm it correctly resolves "it" to leave policy (memory working).
4. Click "Clear Conversation" — confirm chat history resets but the document is still indexed (ask a question again without re-uploading).
5. Ask something unrelated (e.g. "what's the capital of France") — confirm you get the "not found in the uploaded documents" message, not a guess or a general-knowledge answer, since a document is indexed.
6. Try uploading a `.png` — confirm Streamlit's file picker rejects it (type filter) and, if you bypass that, the app shows a clear error rather than crashing.

**Definition of done:** every manual test above passes without a raw Python traceback appearing in the UI.

---

## Module 10 — Logging & Exception-Handling Audit

**Goal:** a final pass to make sure every module fails loudly but cleanly, per the assignment's explicit requirement.

**Files:** all of `modules/*.py`, `app/streamlit_app.py` (review pass, small edits only)

**Checklist per file:**
- No bare `print()` calls remain — replaced with `get_logger(__name__)` calls at appropriate levels (`info` for normal flow, `warning` for recoverable issues like index rebuild, `error` for caught exceptions before re-raising or returning a user message).
- Every function that can fail on bad input (missing file, bad file type, missing API key, empty document, corrupted index) raises an exception with a **specific, actionable message** — not a generic "something went wrong."
- `streamlit_app.py` never lets an unhandled exception reach the user as a raw traceback — every entry point from a button/input triggers a try/except that logs the full exception and shows a short, clear `st.error(...)`.

**Test locally:** deliberately trigger each of these and confirm a clean log line + a clean UI message (not a crash):
1. Remove `OPENAI_API_KEY` temporarily from `.env`, restart app, try to index a document.
2. Upload a `.txt` file that's empty (0 readable text) — should hit the existing "No readable text was found" path.
3. Delete `index.pkl` from an already-indexed chat's vectorstore folder, then ask a question — should trigger the Module 3 rebuild-or-clear-error path.

**Definition of done:** all three trigger conditions produce a log entry in `logs/hoopoe.log` and a readable error in the UI, never a stack trace.

---

## Module 11 — Final README.md Rewrite

**Goal:** the deliverable README covering everything the assignment asks for.

**Files:** `README.md`

**Required sections** (in this order): Project title; Problem statement; Solution overview; Architecture diagram (ASCII or Mermaid, matching the assignment's Documents → Loader → Chunking → Embeddings → Vector Store → {Vector Search, BM25} → Hybrid → Reranking → LLM+Memory → Grounded Answer+Sources → Streamlit UI flow); Technology stack; Project structure (tree, updated with `bm25_retriever.py`, `reranker.py`, `config.yaml`, `logs/`); Setup instructions; Environment variable requirements (`OPENAI_API_KEY`, `GEMINI_API_KEY`); How to run the app; Sample inputs (2–3 example questions against the sample docs, including one follow-up question testing memory); Sample outputs (the actual answer + citations you got when you tested Module 9); Key design decisions (per-chat FAISS isolation, hybrid weights, reranking thresholds, why summary-memory bug was fixed this way); Limitations (in-memory BM25 rebuilt per session — not persisted; local FAISS + ephemeral hosting caveat already noted in the old README; cross-encoder adds latency; summary memory requires `OPENAI_API_KEY` even when chatting with Gemini).

**Test locally:** have someone unfamiliar with the repo follow the Setup + Run sections from a clean checkout and confirm they reach a working chat with no undocumented steps.

**Definition of done:** README is accurate to the final code (write this module last, after Module 10 passes).

---

## Suggested order

Modules 0 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11, testing locally after each one before starting the next. Module 1 is already done.
