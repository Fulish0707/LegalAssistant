"""Run the golden set against the RAG agent for one provider.

    python -m evals.run_eval --provider ollama
    python -m evals.run_eval --provider google

Why a script and not a notebook: this has to be runnable in CI, diffable, and
comparable across providers. The notebook it replaces could not be any of those.

What is scored, and by what:

  retrieval      DETERMINISTIC, and currently a sanity check rather than a
                 score: each fixture is short enough to become one chunk, so a
                 three-fixture corpus at k=4 returns everything for every query
                 and the expected source is always present. It proves ingestion
                 and the user_id filter work. Grow the fixtures past a chunk
                 each before reading it as a retrieval quality number.
  tool use       DETERMINISTIC. Whether the run contains a ToolMessage, i.e.
                 whether the agent called search_documents at all. Without this
                 the unanswerable score is unreadable: a model that declines
                 everything, including questions the documents answer, scores
                 full marks on the unanswerable cases for the wrong reason.
  correctness    DETERMINISTIC. One of the case's accepted spellings of the
                 fact must appear in the agent's answer. Short facts only --
                 numbers, names, dates -- so containment is a fair test.
  decline        DETERMINISTIC, unanswerable cases only. The answer must say it
                 could not find the information, and must not contain the
                 fabricated specifics the case forbids.
  groundedness   MODEL-JUDGED, and the only model-judged number here. One extra
                 LLM call per answerable case asks whether every claim in the
                 answer is supported by the retrieved passages. A 4B judging its
                 own output is weak evidence; treat it as a smell test, not a
                 score, and pass --no-judge to skip it.
  parse failure  DETERMINISTIC. qwen3:4b populates `structured_response` only
                 sometimes, so a run where the sections cannot be extracted is
                 counted and reported on its own rather than scored as a wrong
                 answer.

Isolation: the fixtures are ingested under a dedicated user_id into a
throwaway persist directory (RAG_PERSIST_DIR), which is deleted on the way out.
Nothing is written to the real knowledge base, and nothing is written to the
legacy `langchain` collection -- the collection name is provider-scoped.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
EVAL_USER_ID = "eval-fixtures-do-not-use-for-real-accounts"

# Phrases that count as an honest "I could not find it". Kept in English and
# Korean because AGENT_PREFIX tells the agent to follow the user's language;
# the golden set is English, so the English list is what normally fires.
DECLINE_PHRASES = (
    "no matching",
    "not found",
    "not stated",
    "not specified",
    "not mentioned",
    "does not specify",
    "does not state",
    "does not mention",
    "doesn't specify",
    "doesn't state",
    "doesn't mention",
    "no information",
    "not available",
    "cannot find",
    "can't find",
    "could not find",
    "couldn't find",
    "unable to find",
    "no mention",
    "is not included",
    "don't have access",
    "do not have access",
    "찾을 수 없",
    "언급되어 있지 않",
    "정보가 없",
)

JUDGE_PROMPT = """You are grading one answer from a document question-answering system.

RETRIEVED PASSAGES:
{passages}

QUESTION: {question}

ANSWER: {answer}

Is every factual claim in the ANSWER supported by the RETRIEVED PASSAGES?
Reply with exactly one word: GROUNDED or UNGROUNDED."""


def normalise(text):
    """Lowercase and drop thousands separators, so 4,500 and 4500 are one fact."""
    return (text or "").lower().replace(",", "").replace(" ", " ")


def contains_any(text, needles):
    haystack = normalise(text)
    return any(normalise(needle) in haystack for needle in needles if needle)


def load_fixtures():
    files = sorted(f for f in os.listdir(FIXTURES_DIR) if f.endswith(".txt"))
    return [(name, open(os.path.join(FIXTURES_DIR, name), encoding="utf-8").read()) for name in files]


def configure_provider(provider):
    """Pin both providers before anything constructs a client.

    Setting only LLM_PROVIDER would leave EMBEDDING_PROVIDER at its default of
    google, so an `--provider ollama` run would still need a Google key to
    ingest and would fail in a way that looks like a bug in this harness.
    """
    os.environ["LLM_PROVIDER"] = provider
    os.environ["EMBEDDING_PROVIDER"] = provider
    # Web search is removed for the duration of the run. This measures retrieval
    # over the fixtures; leaving SerpAPI in the tool list lets the agent answer a
    # fixture question from the open web -- and, with the key in this .env being
    # rejected, it spent its turn relaying a SerpAPI error instead of answering.
    os.environ.pop("SERPAPI_API_KEY", None)


def preflight(provider):
    """Prove the provider actually works before spending minutes on the set.

    get_chat_model() returns None only when a key is absent. A key that is
    present but rejected produces a model that raises on first use, so both
    the embedding and the chat path are exercised here and the failure is
    reported as one line naming the provider, not as a traceback.
    """
    from llm_model import get_chat_model, get_embedding_model

    embeddings = get_embedding_model()
    if embeddings is None:
        return f"embedding provider {provider!r} is not configured (missing API key)."
    try:
        embeddings.embed_query("preflight")
    except Exception as error:
        return f"embedding provider {provider!r} rejected a test call: {error}"

    model = get_chat_model()
    if model is None:
        return f"chat provider {provider!r} is not configured (missing API key)."
    try:
        model.invoke("Reply with the single word OK.")
    except Exception as error:
        return f"chat provider {provider!r} rejected a test call: {error}"
    return None


def ingest_fixtures():
    from langchain_config import ingest_text_to_vector_store

    total = 0
    for name, text in load_fixtures():
        result = ingest_text_to_vector_store(text, name, user_id=EVAL_USER_ID)
        if not result.get("ok"):
            raise RuntimeError(f"Could not ingest {name}: {result.get('message')}")
        total += result.get("chunk_count", 0)
    return total


def answer_of(result):
    """The chat answer, plus whether the structured contract held.

    Returns (answer, parsed). parsed is False when neither the response schema
    nor the JSON fallback yielded sections -- reported separately, because a
    formatting failure is a different finding from a wrong answer.
    """
    from langchain_config import extract_agent_sections

    sections = extract_agent_sections(result)
    if sections:
        return sections["chat_response"], True
    # AIMessage only. Falling back to "the last message with content" picks up
    # ToolMessages, so a failed tool call would be scored as the model's answer.
    from langchain_core.messages import AIMessage

    messages = (result or {}).get("messages") or []
    for message in reversed(messages):
        content = getattr(message, "content", None)
        if isinstance(message, AIMessage) and isinstance(content, str) and content.strip():
            return content.strip(), False
    return "", False


def judge_groundedness(model, question, answer, passages):
    try:
        verdict = model.invoke(
            JUDGE_PROMPT.format(passages=passages or "(none)", question=question, answer=answer)
        )
        text = getattr(verdict, "content", "") or ""
        return "UNGROUNDED" not in text.upper()
    except Exception:
        return None


def run(provider, judge_enabled, limit):
    from langchain_core.messages import ToolMessage

    from langchain_config import build_agent, build_run_config, retrieve_for_user
    from llm_model import get_chat_model, get_collection_name

    from evals.dataset import GOLDEN_SET

    print(f"collection: {get_collection_name()}  persist_dir: {os.environ['RAG_PERSIST_DIR']}")
    chunks = ingest_fixtures()
    print(f"ingested {chunks} chunks from {len(load_fixtures())} fixtures\n", flush=True)

    agent = build_agent(EVAL_USER_ID)
    judge_model = get_chat_model() if judge_enabled else None

    cases = GOLDEN_SET[:limit] if limit else GOLDEN_SET
    rows = []
    for case in cases:
        started = time.time()
        documents = retrieve_for_user(EVAL_USER_ID, case["question"], k=4)
        sources = {(d.metadata or {}).get("source") for d in documents}
        passages = "\n\n".join(d.page_content for d in documents)

        expected_sources = [s for s in (case.get("expected_source"), case.get("also_requires_source")) if s]
        retrieval_ok = all(s in sources for s in expected_sources) if expected_sources else None

        config = build_run_config(EVAL_USER_ID)
        config["tags"] = list(config["tags"]) + ["eval", f"case:{case['id']}"]
        config["metadata"] = {**config["metadata"], "eval_case": case["id"]}
        try:
            result = agent.invoke({"messages": [("user", case["question"])]}, config)
            answer, parsed = answer_of(result)
            tool_called = any(isinstance(m, ToolMessage) for m in (result or {}).get("messages") or [])
            error = None
        except Exception as exc:
            answer, parsed, tool_called, error = "", False, False, str(exc)

        if case.get("unanswerable"):
            declined = contains_any(answer, DECLINE_PHRASES)
            fabricated = contains_any(answer, case.get("forbidden", []))
            # tool_called is part of the criterion on purpose. Declining without
            # searching is not the behaviour being measured; it is the same
            # refusal that loses the answerable cases, and scoring it as a pass
            # would hide the actual failure.
            correct = tool_called and declined and not fabricated
        else:
            correct = contains_any(answer, case["accept"])

        grounded = None
        if judge_model is not None and answer and not case.get("unanswerable"):
            grounded = judge_groundedness(judge_model, case["question"], answer, passages)

        rows.append({
            "id": case["id"],
            "correct": correct,
            "retrieval_ok": retrieval_ok,
            "parsed": parsed,
            "tool_called": tool_called,
            "grounded": grounded,
            "seconds": time.time() - started,
            "error": error,
            "answer": " ".join(answer.split())[:160],
        })
        mark = "PASS" if correct else "FAIL"
        retrieval_mark = {True: "hit", False: "MISS", None: "-"}[retrieval_ok]
        # flush: a run takes minutes, so per-case progress has to be visible
        # while it happens, not only when the process exits.
        print(f"[{mark}] {case['id']:<24} retrieval={retrieval_mark:<4} "
              f"searched={'y' if tool_called else 'N'} parsed={'y' if parsed else 'n'} "
              f"grounded={grounded} {rows[-1]['seconds']:.1f}s", flush=True)
        if not correct:
            print(f"       answer: {rows[-1]['answer'] or error}", flush=True)

    return rows


def report(provider, rows, elapsed):
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    retrieval_scored = [r for r in rows if r["retrieval_ok"] is not None]
    retrieval_hits = sum(1 for r in retrieval_scored if r["retrieval_ok"])
    judged = [r for r in rows if r["grounded"] is not None]
    grounded = sum(1 for r in judged if r["grounded"])
    parse_failures = sum(1 for r in rows if not r["parsed"])
    searched = sum(1 for r in rows if r["tool_called"])

    print("\n" + "=" * 62)
    print(f"provider: {provider}   cases: {total}   wall clock: {elapsed:.0f}s")
    print(f"correctness  (deterministic): {correct}/{total}")
    if retrieval_scored:
        print(f"retrieval    (deterministic): {retrieval_hits}/{len(retrieval_scored)}"
              "   [sanity check only: 3 chunks, k=4 returns all of them]")
    print(f"searched docs(deterministic): {searched}/{total}")
    print(f"parse success(deterministic): {total - parse_failures}/{total}")
    if judged:
        print(f"groundedness (model-judged) : {grounded}/{len(judged)}")
    else:
        print("groundedness (model-judged) : skipped")
    failures = [r["id"] for r in rows if not r["correct"]]
    print(f"failed cases: {', '.join(failures) if failures else 'none'}")
    print("=" * 62)
    return 0 if correct == total else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the RAG golden set against one provider.")
    parser.add_argument("--provider", choices=("ollama", "google"), default="ollama")
    parser.add_argument("--no-judge", action="store_true", help="Skip the model-judged groundedness call.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N cases.")
    args = parser.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv()
    configure_provider(args.provider)

    # Outside the repo, and removed on the way out: the fixture vectors must not
    # reach the developer's own chroma_db/ even if this run crashes halfway.
    persist_dir = tempfile.mkdtemp(prefix="ps_agent_eval_")
    os.environ["RAG_PERSIST_DIR"] = persist_dir
    started = time.time()
    try:
        problem = preflight(args.provider)
        if problem:
            print(f"Cannot run the evaluation: {problem}", file=sys.stderr)
            print(f"Nothing was ingested. Fix the {args.provider} configuration and retry.", file=sys.stderr)
            return 2
        rows = run(args.provider, not args.no_judge, args.limit)
        return report(args.provider, rows, time.time() - started)
    finally:
        shutil.rmtree(persist_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
