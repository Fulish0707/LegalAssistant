# LangGraph RAG Agent Migration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the chat agent from the deprecated `langchain_classic.initialize_agent` onto `langchain.agents.create_agent`, with a swappable model connector so Qwen3-4B on Ollama is the primary model and Gemini is the comparison baseline, and LangSmith tracing plus a runnable eval harness.

**Architecture:** `llm_model.py` becomes two env-driven factories (`get_chat_model`, `get_embedding_model`) that no provider name escapes. `langchain_config.py` loses all import-time I/O and exposes `build_agent(user_id)` returning a compiled LangGraph agent with a typed `context_schema`, a `SqliteSaver` checkpointer for memory, and one deterministic `search_documents` tool. `app.py` swaps its callback-driven SSE producer for `agent.stream()` while keeping the wire contract identical.

**Tech Stack:** Python, Flask, LangChain 1.x, LangGraph 1.2.11, LangSmith 0.11.0, `langchain-ollama` 1.1.0 (qwen3:4b, nomic-embed-text), `langchain-google-genai` (Gemini, text-embedding-004), Chroma, SQLite.

**Design:** `docs/superpowers/specs/2026-09-06-langgraph-rag-agent-design.md`

---

## Base branch — read before Task 1

This plan builds on `feat/google-oidc-user-foundation` (worktree at `.worktrees/google-oidc-user-foundation`, HEAD `ce75d49`), **not `main`**. That branch already replaced the process-global retriever with `build_agent(user_id)` / `build_rag_tool(user_id)` factories, and this plan extends those factories rather than recreating them.

If that branch is not going to merge soon, stop and rebase this plan onto `main` first — Tasks 5 and 6 assume the factory boundaries exist.

That branch's `requirements.txt` is unpinned and stale against main's pinning work. **Resolve every `requirements.txt` conflict toward main's pinned versions.**

## Baseline — verify before starting

```bash
python -m pytest tests/ -q
```

Expected on `main` today: `5 failed, 5 passed`. All five failures come from `load_dotenv()` running at import in `llm_model.py`, which restores `GOOGLE_API_KEY` after the tests deliberately remove it. **Task 2 is what turns them green.** Do not treat them as pre-existing noise to ignore — they are the acceptance signal.

---

## Task 1: Pin the new dependencies

**Files:**
- Modify: `requirements.txt`

**Step 1: Confirm what is already installed**

```bash
pip show langgraph langsmith langchain-ollama langgraph-checkpoint-sqlite 2>&1 | grep -E "^(Name|Version)|WARNING"
```

Expected: `langgraph 1.2.11`, `langsmith 0.11.0`, `langchain-ollama 1.1.0` present; `langgraph-checkpoint-sqlite` **not found**.

**Step 2: Install the missing checkpointer**

```bash
pip install langgraph-checkpoint-sqlite
python -c "from langgraph.checkpoint.sqlite import SqliteSaver; print('ok', SqliteSaver)"
```

Expected: `ok <class ...SqliteSaver>`

**Step 3: Add the pins**

Append to `requirements.txt`, using the exact versions `pip show` reported:

```
# LangGraph agent runtime
langgraph==1.2.11
langsmith==0.11.0
langgraph-checkpoint-sqlite==<version from step 2>
langchain-ollama==1.1.0
```

Do **not** add `langchain-huggingface`. Qwen3-4B is served through Ollama; see the design's runtime section.

**Step 4: Note the Python drift**

`requirements.txt` header claims Python 3.10; the active interpreter is 3.13.9. Update the header comment to state the interpreter actually in use, or reconcile the environment. Do not silently leave both claims in the file.

**Step 5: Commit**

```bash
git add requirements.txt
git commit -m "chore: pin langgraph, langsmith, and ollama connector deps"
```

---

## Task 2: Model connectors with no import-time side effects

This is the highest-value task: it fixes the five failing tests and creates the seam every later task depends on.

**Files:**
- Modify: `llm_model.py` (full rewrite)
- Test: `tests/test_model_connectors.py` (create)

**Step 1: Write the failing test**

```python
import importlib
import os
import sys
import unittest


class ModelConnectorTests(unittest.TestCase):
    def setUp(self):
        sys.modules.pop("llm_model", None)
        self._saved = dict(os.environ)

    def tearDown(self):
        os.environ.clear()
        os.environ.update(self._saved)
        sys.modules.pop("llm_model", None)

    def test_import_does_not_build_clients(self):
        """Importing must be inert: no I/O, no client construction."""
        os.environ.pop("GOOGLE_API_KEY", None)
        module = importlib.import_module("llm_model")

        self.assertFalse(hasattr(module, "llm"))
        self.assertFalse(hasattr(module, "embedding_model"))

    # Do NOT assert os.environ["GOOGLE_API_KEY"] is None here. pytest runs the whole
    # suite in one process, and any earlier test that imports `app` will have called
    # load_dotenv() and repopulated it. That assertion passes or fails on test
    # ordering, not on anything llm_model does. The grep in Step 4 checks it properly.

    def test_google_chat_model_unavailable_without_key(self):
        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ["LLM_PROVIDER"] = "google"
        module = importlib.import_module("llm_model")

        self.assertIsNone(module.get_chat_model())

    def test_collection_name_is_derived_from_embedding_provider(self):
        """Different embedding providers must never share a Chroma collection."""
        module = importlib.import_module("llm_model")

        os.environ["EMBEDDING_PROVIDER"] = "google"
        google_name = module.get_collection_name()
        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        ollama_name = module.get_collection_name()

        self.assertNotEqual(google_name, ollama_name)

    def test_unknown_provider_raises(self):
        os.environ["LLM_PROVIDER"] = "not-a-provider"
        module = importlib.import_module("llm_model")

        with self.assertRaises(ValueError):
            module.get_chat_model()
```

**Step 2: Run it and watch it fail**

```bash
python -m pytest tests/test_model_connectors.py -q
```

Expected: FAIL — `llm_model` currently defines module-level `llm` and `embedding_model` and has no `get_chat_model`.

**Step 3: Rewrite `llm_model.py`**

```python
"""Model connectors. Provider names appear here and nowhere else."""

import os

DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_EMBEDDING_PROVIDER = "google"

# Chroma is dimension-locked per collection. Two providers at the same dimension
# (text-embedding-004 and nomic-embed-text are both 768) will be silently accepted
# into one collection and then compared across unrelated embedding spaces, which
# produces wrong neighbours with no error. The collection name therefore encodes
# the provider and model, so switching connectors creates a new collection.
COLLECTION_NAMES = {
    ("google", "models/text-embedding-004"): "docs_google_te004",
    ("ollama", "nomic-embed-text"): "docs_ollama_nomic",
}


def _llm_provider():
    return os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().lower()


def _embedding_provider():
    return os.getenv("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER).strip().lower()


def get_chat_model():
    """Return a chat model, or None when the provider is not configured."""
    provider = _llm_provider()

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        # reasoning=True keeps `content` clean and isolates Qwen3's thinking in
        # additional_kwargs["reasoning_content"]. reasoning=False is NOT the safe
        # choice: it strips the <think> tags and leaves the reasoning prose inside
        # content, where nothing downstream can recognise it. Verified on qwen3:4b.
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            reasoning=True,
            # Qwen3 degrades and repeats under greedy decoding. Do not use 0.
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.7")),
            top_p=0.8,
            top_k=20,
        )

    if provider == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            return None
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.0-flash"),
            temperature=0,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {provider!r}")


def get_embedding_model():
    """Return an embedding model, or None when the provider is not configured."""
    provider = _embedding_provider()

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    if provider == "google":
        if not os.getenv("GOOGLE_API_KEY"):
            return None
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBED_MODEL", "models/text-embedding-004")
        )

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


def get_embedding_descriptor():
    provider = _embedding_provider()
    if provider == "ollama":
        return provider, os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    if provider == "google":
        return provider, os.getenv("GOOGLE_EMBED_MODEL", "models/text-embedding-004")
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider!r}")


def get_collection_name():
    """Chroma collection dedicated to the active embedding provider and model."""
    descriptor = get_embedding_descriptor()
    if descriptor in COLLECTION_NAMES:
        return COLLECTION_NAMES[descriptor]
    provider, model = descriptor
    safe_model = "".join(c if c.isalnum() else "_" for c in model).strip("_")
    return f"docs_{provider}_{safe_model}"


def describe_configuration():
    """Reporting helper for /health. Must not construct clients."""
    return {
        "llm_provider": _llm_provider(),
        "embedding_provider": _embedding_provider(),
        "collection": get_collection_name(),
    }
```

`load_dotenv()` moves to `app.py` startup only. Library modules must never call it.

**Step 4: Run the new tests**

```bash
python -m pytest tests/test_model_connectors.py -q
```

Expected: PASS (4 passed)

**Step 5: Update every caller**

`llm_model` no longer exports `llm` or `embedding_model`. Fix all importers:

```bash
grep -rn "from llm_model import\|import llm_model" --include=*.py . | grep -v .worktrees
```

Each becomes a call site: `llm = get_chat_model()` inside the function that needs it, never at module level.

**Step 6: Confirm the baseline failures are now green**

```bash
python -m pytest tests/ -q
```

Expected: **the five original failures now pass.** The total count is higher than 10 because this task also adds `tests/test_model_connectors.py`; the total keeps growing as later tasks add files, so judge this step by "no remaining failures", not by a fixed number. If any of the original five still fail, something still performs I/O at import — find it before continuing.

Then confirm the root cause is gone for good:

```bash
grep -n "load_dotenv" llm_model.py langchain_config.py
```

Expected: **no output.** `load_dotenv()` belongs to `app.py` startup only. A library module that calls it re-creates exactly the bug that made those five tests fail.

**Step 7: Commit**

```bash
git add llm_model.py tests/test_model_connectors.py app.py langchain_config.py
git commit -m "refactor: replace module-level models with env-driven connectors"
```

---

## Task 3: Provider-scoped vector store

**Files:**
- Modify: `langchain_config.py`
- Test: `tests/test_vector_store_scope.py` (create)

**Step 1: Write the failing test**

```python
import os
import unittest

from llm_model import get_collection_name


class VectorStoreScopeTests(unittest.TestCase):
    def test_google_and_ollama_never_share_a_collection(self):
        """768-dim collision: nomic-embed-text and text-embedding-004 are both 768,
        so Chroma would accept both into one collection and silently return
        neighbours computed across two unrelated embedding spaces."""
        os.environ["EMBEDDING_PROVIDER"] = "google"
        google = get_collection_name()
        os.environ["EMBEDDING_PROVIDER"] = "ollama"
        ollama = get_collection_name()

        self.assertNotEqual(google, ollama)
        self.assertNotEqual(google, "langchain")
        self.assertNotEqual(ollama, "langchain")
```

**Step 2: Run it**

```bash
python -m pytest tests/test_vector_store_scope.py -q
```

**Step 3: Thread the collection name through Chroma**

In `langchain_config.py`, replace every bare `Chroma(...)` / `Chroma.from_documents(...)` construction with one lazy accessor:

```python
def get_vector_store():
    """Build the Chroma handle on demand. Never at import."""
    embedding_model = get_embedding_model()
    if embedding_model is None:
        return None
    return Chroma(
        collection_name=get_collection_name(),
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model,
    )
```

Delete the module-level `vector_store` global and the `global vector_store` statements. Every function takes the handle from `get_vector_store()`.

**Step 4: Run the full suite**

```bash
python -m pytest tests/ -q
```

**Step 5: Commit**

```bash
git add langchain_config.py tests/test_vector_store_scope.py
git commit -m "fix: scope Chroma collection to the active embedding provider"
```

**Note on existing data:** the legacy `langchain` collection stays on disk, orphaned and Google-embedded. It is not migrated and not deleted. Re-ingest under the new collection when switching providers; document this rather than automating it.

---

## Task 4: Deterministic `search_documents` tool

**Files:**
- Modify: `langchain_config.py`
- Test: `tests/test_search_documents_tool.py` (create)

**Step 1: Write the failing test**

```python
import unittest
from unittest.mock import patch

from langchain_core.documents import Document


class SearchDocumentsToolTests(unittest.TestCase):
    def test_tool_name_is_function_call_safe(self):
        """Gemini and Qwen3 both require ^[a-zA-Z0-9_-]+$. 'RAG Retriever' fails."""
        import langchain_config

        tool = langchain_config.build_rag_tool("user-1")
        self.assertEqual(tool.name, "search_documents")
        self.assertNotIn(" ", tool.name)

    def test_duplicate_chunks_are_collapsed(self):
        import langchain_config

        docs = [
            Document(page_content="same text", metadata={"source": "a.pdf"}),
            Document(page_content="same text", metadata={"source": "a.pdf"}),
            Document(page_content="other text", metadata={"source": "b.pdf"}),
        ]
        with patch.object(langchain_config, "retrieve_for_user", return_value=docs):
            result = langchain_config.build_rag_tool("user-1").invoke(
                {"query": "anything"}
            )

        self.assertEqual(result.count("same text"), 1)

    def test_result_carries_citations(self):
        import langchain_config

        docs = [Document(page_content="body", metadata={"source": "a.pdf"})]
        with patch.object(langchain_config, "retrieve_for_user", return_value=docs):
            result = langchain_config.build_rag_tool("user-1").invoke(
                {"query": "anything"}
            )

        self.assertIn("a.pdf", result)

    def test_empty_retrieval_is_explicit(self):
        """The model must be told retrieval found nothing, so it does not invent."""
        import langchain_config

        with patch.object(langchain_config, "retrieve_for_user", return_value=[]):
            result = langchain_config.build_rag_tool("user-1").invoke(
                {"query": "anything"}
            )

        self.assertIn("No matching", result)
```

**Step 2: Run it, expect failure**

```bash
python -m pytest tests/test_search_documents_tool.py -q
```

**Step 3: Implement**

Replace `build_rag_tool` and delete the `RetrievalQA` / `load_qa_chain` imports entirely:

```python
from langchain_core.tools import tool


def retrieve_for_user(user_id, query, k=4):
    """Similarity search scoped to one account. Seam for tests."""
    store = get_vector_store()
    if store is None or not user_id:
        return []
    return store.similarity_search(query, k=k, filter={"user_id": user_id})


def build_rag_tool(user_id):
    """A deterministic retrieval tool scoped to one account.

    Every step here is code, not model judgement. That matters more with a 4B
    model: what the model cannot decide, it cannot get wrong.
    """

    @tool("search_documents")
    def search_documents(query: str) -> str:
        """Search the signed-in user's uploaded documents. Returns excerpts with sources."""
        documents = retrieve_for_user(user_id, query)
        if not documents:
            return "No matching passages were found in the user's documents."

        seen = set()
        blocks = []
        for document in documents:
            body = (document.page_content or "").strip()
            if not body or body in seen:
                continue
            seen.add(body)
            source = (document.metadata or {}).get("source") or "Unknown source"
            blocks.append(f"[source: {source}]\n{body}")

        return "\n\n".join(blocks)

    return search_documents
```

The old `rag_tool_func` retrieved twice — once via `retriever.invoke`, then again inside `RetrievalQA`. This retrieves once.

**Step 4: Run tests**

```bash
python -m pytest tests/test_search_documents_tool.py -q
```

Expected: 4 passed

**Step 5: Commit**

```bash
git add langchain_config.py tests/test_search_documents_tool.py
git commit -m "refactor: make RAG retrieval a deterministic search_documents tool"
```

---

## Task 5: `create_agent` with typed context and structured output

**Files:**
- Modify: `langchain_config.py`
- Test: `tests/test_build_agent.py` (create)

**Step 1: Write the failing test**

```python
import unittest
from unittest.mock import patch


class BuildAgentTests(unittest.TestCase):
    def test_returns_none_when_no_model_configured(self):
        import langchain_config

        with patch.object(langchain_config, "get_chat_model", return_value=None):
            self.assertIsNone(langchain_config.build_agent("user-1"))

    def test_model_without_bind_tools_fails_loudly(self):
        """create_agent requires bind_tools. qwen3:4b has it (verified), so a model
        that lacks it means a misconfigured provider — say so, do not silently
        degrade into a different product."""
        import langchain_config

        class NoToolModel:
            pass

        with patch.object(langchain_config, "get_chat_model", return_value=NoToolModel()):
            with self.assertRaises(RuntimeError) as caught:
                langchain_config.build_agent("user-1")

        self.assertIn("bind_tools", str(caught.exception))

    def test_rag_tool_absent_for_anonymous_callers(self):
        import langchain_config

        with patch.object(langchain_config, "build_rag_tool") as rag:
            langchain_config.build_agent(None)
            rag.assert_not_called()
```

**Step 2: Run it**

```bash
python -m pytest tests/test_build_agent.py -q
```

**Step 3: Implement**

```python
from typing import Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field
from langchain.agents import create_agent


@dataclass
class AgentContext:
    """Typed request context. Replaces closure-captured globals."""
    user_id: Optional[str] = None


class AgentSections(BaseModel):
    """Schema-enforced replacement for the [CHAT]/[STATEMENT_SUGGESTION] markers.

    A 4B model follows freeform marker conventions far less reliably than Gemini,
    so the contract is enforced by schema instead of recovered by regex.
    """
    chat_response: str = Field(description="The conversational answer.")
    statement_suggestion: str = Field(default="", description="University personal statement improvements.")
    job_statement_suggestion: str = Field(default="", description="Job application statement improvements.")
    report_suggestion: str = Field(default="", description="Report structure and clarity improvements.")


def build_agent(user_id=None, checkpointer=None):
    model = get_chat_model()
    if model is None:
        return None

    if not hasattr(model, "bind_tools"):
        # Verified: qwen3:4b via ChatOllama does support bind_tools. Reaching this
        # branch means the configured provider cannot run a tool loop at all, so a
        # RAG agent is not what would come back. Fail loudly rather than quietly
        # serving a different, worse product under the same name.
        raise RuntimeError(
            f"LLM_PROVIDER={os.getenv('LLM_PROVIDER')} returned a model without "
            "bind_tools; create_agent cannot run a tool loop with it."
        )

    tools = list(get_search_tools())
    if user_id:
        tools.insert(0, build_rag_tool(user_id))

    return create_agent(
        model,
        tools=tools,
        system_prompt=AGENT_PREFIX,
        response_format=AgentSections,
        context_schema=AgentContext,
        checkpointer=checkpointer,
    )
```

Keep the tool list small. A 4B degrades quickly as tool count grows: `search_documents` plus optional web search is the budget.

**Step 4: Run tests**

```bash
python -m pytest tests/test_build_agent.py -q
```

**Step 5: Live smoke test against the real model**

```bash
ollama list | grep qwen3
python -c "
import os
os.environ['LLM_PROVIDER']='ollama'
from langchain_config import build_agent
a = build_agent('user-1')
out = a.invoke({'messages':[('user','Say hello in one sentence.')]})
print('structured:', out.get('structured_response'))
"
```

Expected: a populated `AgentSections` with a non-empty `chat_response`. Verified working on `qwen3:4b` during design: the tool loop and `response_format` coexist because the structured answer is emitted as its own tool call.

**Step 6: Commit**

```bash
git add langchain_config.py tests/test_build_agent.py
git commit -m "feat: build the chat agent with create_agent and typed context"
```

---

## Task 6: Conversation memory via SqliteSaver

**Files:**
- Modify: `langchain_config.py`, `app.py`
- Test: `tests/test_agent_memory.py` (create)

**Step 1: Write the failing test**

```python
import tempfile
import unittest


class AgentMemoryTests(unittest.TestCase):
    def test_checkpointer_persists_turns_per_thread(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        import langchain_config

        with tempfile.TemporaryDirectory() as tmp:
            with SqliteSaver.from_conn_string(f"{tmp}/checkpoints.db") as saver:
                agent = langchain_config.build_agent("user-1", checkpointer=saver)
                if agent is None:
                    self.skipTest("No model configured")

                config = {"configurable": {"thread_id": "user-1"}}
                agent.invoke({"messages": [("user", "My name is Ada.")]}, config)
                state = agent.get_state(config)

                self.assertGreater(len(state.values["messages"]), 1)

    def test_threads_are_isolated_between_users(self):
        from langgraph.checkpoint.sqlite import SqliteSaver
        import langchain_config

        with tempfile.TemporaryDirectory() as tmp:
            with SqliteSaver.from_conn_string(f"{tmp}/checkpoints.db") as saver:
                agent = langchain_config.build_agent("user-1", checkpointer=saver)
                if agent is None:
                    self.skipTest("No model configured")

                agent.invoke(
                    {"messages": [("user", "My name is Ada.")]},
                    {"configurable": {"thread_id": "user-1"}},
                )
                other = agent.get_state({"configurable": {"thread_id": "user-2"}})

                self.assertEqual(other.values.get("messages", []), [])
```

**Step 2: Run it**

```bash
python -m pytest tests/test_agent_memory.py -q
```

**Step 3: Wire the checkpointer into the app**

The OIDC branch deleted `ConversationBufferMemory` and now passes `chat_history=[]`, so the agent is currently stateless between turns. The checkpointer restores that.

In `app.py`, open one long-lived `SqliteSaver` over `sqlite_db/checkpoints.db` at startup and pass it to `build_agent`.

**Thread affinity matters here.** `SqliteSaver.from_conn_string` is a context manager, fine inside a test but wrong for app startup, and Flask runs the streaming generator on a different thread than the one that opened the connection. Construct it from an explicit connection instead:

```python
connection = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
CHECKPOINTER = SqliteSaver(connection)
```

Without `check_same_thread=False`, the first streamed request raises a thread-affinity error. Invoke with `config={"configurable": {"thread_id": user_id}}` and `context=AgentContext(user_id=user_id)`.

Keep the SQLite `chat_history` table as the **display** record. The checkpointer is the agent's **working state**. They are separate on purpose — do not try to unify them.

**Step 4: Run tests**

```bash
python -m pytest tests/ -q
```

**Step 5: Commit**

```bash
git add langchain_config.py app.py tests/test_agent_memory.py
git commit -m "feat: restore conversation memory with a LangGraph checkpointer"
```

---

## Task 7: Streaming via graph stream events

**Files:**
- Modify: `app.py` (`_generate_agent_response_stream`, and delete `StreamingCallbackHandler`)
- Test: `tests/test_chat_streaming.py` (create)

**The constraint:** `on_agent_action`, `on_tool_end`, and `on_agent_finish` **do not fire under LangGraph**. The queue-and-callback producer must be replaced or chat breaks.

**The contract that must not change:** `static/lc-chat.js:261` opens `new EventSource('/chat?...')` and consumes `{type: token|thought|error|stream_end}`. Keep emitting exactly those shapes. The frontend is not modified in this task.

**Step 1: Write the failing test**

```python
import json
import unittest


class ChatStreamingTests(unittest.TestCase):
    def test_stream_emits_known_event_types_and_terminates(self):
        import app as app_module

        client = app_module.app.test_client()
        response = client.get("/chat?message=hello")
        body = response.get_data(as_text=True)

        events = [
            json.loads(line[len("data: "):])
            for line in body.splitlines()
            if line.startswith("data: ")
        ]
        self.assertTrue(events)
        for event in events:
            self.assertIn(event["type"], {"token", "thought", "error", "stream_end", "final"})
        self.assertEqual(events[-1]["type"], "stream_end")
```

**Step 2: Run it**

```bash
python -m pytest tests/test_chat_streaming.py -q
```

**Do not let this test reach the live model.** With `LLM_PROVIDER=ollama` a real
`/chat` request runs actual inference: slow on every suite run, and a hard failure on
any machine without Ollama. Patch `build_agent` to return a stub whose `.stream()`
yields fixed chunks. The subject of this test is the SSE event contract, not model
output.

**Step 3: Replace the producer**

```python
for mode, chunk in agent.stream(
    {"messages": [("user", user_input)]},
    config={"configurable": {"thread_id": user_id}},
    context=AgentContext(user_id=user_id),
    stream_mode=["messages", "updates"],
):
    if mode == "messages":
        message, _ = chunk
        if getattr(message, "content", ""):
            yield sse({"type": "token", "content": message.content})
    elif mode == "updates":
        for node, update in chunk.items():
            yield sse({"type": "thought", "content": f"{node}"})
```

Then emit the final `structured_response` as one `{"type": "final"}` event, followed by `{"type": "stream_end"}`.

**Known UX trade-off:** with `response_format`, the four sections arrive as a structured tool call, so they cannot themselves be token-streamed. Tokens and tool activity stream for progress; the sections land at the end. This is accepted in the design — do not "fix" it by reverting to regex markers.

Delete `StreamingCallbackHandler` and the `queue` plumbing once the test passes. `clean_agent_output` and `parse_agent_output_sections` also become dead once structured output lands — remove them and their tests together, in this task, so nothing half-migrated survives.

**Step 4: Run tests**

```bash
python -m pytest tests/ -q
```

**Step 5: Commit**

```bash
git add app.py tests/test_chat_streaming.py
git commit -m "refactor: stream chat from LangGraph events instead of callbacks"
```

**Out of scope:** converting the prompt-bearing `GET /chat?message=...` into true POST streaming with cancellation and timeouts. That stays roadmap item 5.

---

## Task 8: LangSmith tracing

**Files:**
- Modify: `app.py`, `langchain_config.py`, `.env.example`
- Test: `tests/test_tracing_metadata.py` (create)

**Step 1: Write the failing test**

```python
import os
import unittest


class TracingMetadataTests(unittest.TestCase):
    def test_run_metadata_identifies_the_model_provider(self):
        """Provider tagging is what makes Gemini vs Qwen3 comparable in LangSmith."""
        import langchain_config

        os.environ["LLM_PROVIDER"] = "ollama"
        metadata = langchain_config.run_metadata("user-1")

        self.assertEqual(metadata["llm_provider"], "ollama")
        self.assertEqual(metadata["thread_id"], "user-1")
        self.assertIn("embedding_provider", metadata)

    def test_tracing_is_disabled_by_default(self):
        os.environ.pop("LANGSMITH_TRACING", None)
        import langchain_config

        self.assertFalse(langchain_config.tracing_enabled())
```

**Step 2: Run it**

```bash
python -m pytest tests/test_tracing_metadata.py -q
```

**Step 3: Implement**

Tracing is env-driven — `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` — and off unless explicitly enabled. Attach `run_metadata(user_id)` to every agent invocation so runs carry `thread_id`, `llm_provider`, and model name.

Add the three variables to `.env.example` with empty values and a comment that tracing sends prompts and documents to LangSmith.

**Step 4: Run tests, then commit**

```bash
python -m pytest tests/ -q
git add app.py langchain_config.py .env.example tests/test_tracing_metadata.py
git commit -m "feat: add opt-in LangSmith tracing with provider metadata"
```

---

## Task 9: LangSmith evaluation harness

**Files:**
- Create: `evals/dataset.py`, `evals/run_eval.py`, `evals/fixtures/`
- Delete: `langsmith_eval.ipynb`
- Test: `tests/test_eval_dataset.py` (create)

This is what makes the local move decidable rather than a guess: Qwen3-4B against Gemini on one dataset, answering whether the 4B holds quality and where it fails first.

**Step 1: Build fixtures and a golden set**

Two or three short documents in `evals/fixtures/`, and 10–15 questions in `evals/dataset.py` as `(question, expected_answer, expected_source)`. Include at least:
- one answerable only from a fixture,
- one **unanswerable**, where the correct behaviour is admitting no passage matched,
- one needing two documents.

**Step 2: Write the failing test**

```python
import unittest


class EvalDatasetTests(unittest.TestCase):
    def test_dataset_is_well_formed(self):
        from evals.dataset import GOLDEN_SET

        self.assertGreaterEqual(len(GOLDEN_SET), 10)
        for case in GOLDEN_SET:
            self.assertTrue(case["question"])
            self.assertIn("expected_source", case)

    def test_dataset_covers_the_unanswerable_case(self):
        """A RAG agent that never says 'not found' is a hallucination machine."""
        from evals.dataset import GOLDEN_SET

        self.assertTrue(any(c.get("unanswerable") for c in GOLDEN_SET))
```

**Step 3: Write `evals/run_eval.py`**

A script — not a notebook — taking `--provider ollama|google`, ingesting fixtures under a fixture-scoped `user_id`, running the agent over `GOLDEN_SET`, and reporting **correctness** and **groundedness** (is every claim supported by a retrieved passage). Tag each run with the provider so both appear in one LangSmith comparison.

**Step 4: Run both providers**

```bash
python -m evals.run_eval --provider ollama
python -m evals.run_eval --provider google
```

Record both scores in the design doc. Expect Qwen3-4B to fail first on tool-call formatting and multi-hop retrieval — those are the results that decide whether it can be primary.

**Step 5: Delete the superseded notebook**

`langsmith_eval.ipynb` uses `langchain_teddynote` and raw `google.generativeai`, neither of which survives this migration.

```bash
git rm langsmith_eval.ipynb
```

**Step 6: Commit**

```bash
git add evals/ tests/test_eval_dataset.py
git commit -m "feat: add LangSmith eval harness comparing local and hosted models"
```

---

## Task 10: Update docs and roadmap

**Files:**
- Modify: `README.md`, `docs/integration-roadmap.md`

**Steps**

1. README: the tech stack becomes LangGraph, and the model section documents `LLM_PROVIDER` / `EMBEDDING_PROVIDER`, the Ollama setup (`ollama pull qwen3:4b`), and the mermaid diagram redrawn around `create_agent`.
2. README: state plainly that switching embedding providers requires re-ingestion, because collections are provider-scoped and **the 768-dimension collision between `text-embedding-004` and `nomic-embed-text` fails silently rather than loudly**.
3. Roadmap: mark item 6 (RAG lifecycle and answer quality) advanced by citations, deduplication, and evaluation. Leave item 5 (POST streaming) open — this plan deliberately did not do it.
4. Commit.

```bash
git add README.md docs/integration-roadmap.md
git commit -m "docs: describe the LangGraph agent and provider connectors"
```

---

## Final verification

```bash
python -m pytest tests/ -q          # all green
grep -rn "initialize_agent\|RetrievalQA\|load_qa_chain\|ConversationBufferMemory" --include=*.py . | grep -v .worktrees
```

The grep must return **nothing**. Any hit means a piece of the old stack survived the migration.
