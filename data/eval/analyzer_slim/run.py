"""Preregistered check (PROTOCOL.md): full vs slim QueryAnalyzer on the 274 final-test questions.
Writes runs.jsonl (one row per question and arm) incrementally; resume-safe."""
import json
import sys
import time
from pathlib import Path

from legal_rag.config import load_settings
from legal_rag.factory import build_chat_model
from legal_rag.rag.analyzer import QueryAnalyzer

HERE = Path(__file__).parent
F = Path("/home/layke/Documents/MyLLMAgent/data/eval/reports/chat_pipeline/final-test-2026-10-06")
OUT = HERE / "runs.jsonl"

settings = load_settings()
assert settings.llm.model == "exaone3.5:7.8b", settings.llm.model


class Timed:
    """Wraps the chat model to record wall time, output tokens and parse failures of the analyzer call."""

    def __init__(self, inner):
        self.inner, self.name, self.last = inner, inner.name, {}

    def generate(self, *a, **k):
        t = time.perf_counter()
        try:
            r = self.inner.generate(*a, **k)
        except Exception as exc:
            self.last = {"wall_s": round(time.perf_counter() - t, 3), "error": type(exc).__name__}
            raise
        self.last = {"wall_s": round(time.perf_counter() - t, 3), "parsed": r.parsed is not None,
                     "out_tokens": (r.raw or {}).get("eval_count")}
        return r


model = Timed(build_chat_model(settings))
arms = {"full": QueryAnalyzer(model, full=True), "slim": QueryAnalyzer(model, full=False)}
questions = [(f"{name}:{i}", json.loads(line)["question"]) for name in ("formal", "everyday")
             for i, line in enumerate((F / f"{name}.frozen.jsonl").read_text().splitlines())]
done = {(r["qid"], r["arm"]) for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else set()
extra = sys.argv[1:] == ["--repeat"]  # full-full reference on every 5th question (50+)
with OUT.open("a") as fh:
    for n, (qid, q) in enumerate(questions):
        if extra:
            plan = [("full2", arms["full"])] if n % 5 == 0 else []
        else:
            plan = [("full", arms["full"]), ("slim", arms["slim"])]
            if n % 2:
                plan.reverse()
        for arm, analyzer in plan:
            if (qid, arm) in done:
                continue
            model.last = {}
            a = analyzer.analyze(q, history=[])
            fh.write(json.dumps({"qid": qid, "arm": arm, **model.last, "analysis": a.model_dump(mode="json")}, ensure_ascii=False) + "\n")
            fh.flush()
        if n % 25 == 0:
            print(n, qid, flush=True)
print("done")
