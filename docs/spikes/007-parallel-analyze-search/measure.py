"""Parallel vs sequential analyze+search on the serving stack (EXAONE, bge-m3 CPU, fp16 reranker).
Measures wall time from the turn's start to the moment generation would start, and whether the evidence
bundle is the same. Generation is skipped (it does not change). Arms alternate per question."""
import json
import sys
import time
from pathlib import Path

from legal_rag.config import load_settings
from legal_rag.factory import build_orchestrator
from legal_rag.rag.orchestrator import Orchestrator

OUT = Path(__file__).with_name("parallel_runs.jsonl")
F = Path("/home/layke/Documents/MyLLMAgent/data/eval/reports/chat_pipeline/final-test-2026-10-06")
qs = [json.loads(line)["question"] for name in ("formal", "everyday") for line in (F / f"{name}.frozen.jsonl").read_text().splitlines()]
picks = list(range(3, 274, 7))[: int(sys.argv[1]) if len(sys.argv) > 1 else 40]


class Stop(Exception):
    pass


captured = {}


def fake_generate(self, question, bundle, *a, **k):
    captured["t"] = time.perf_counter()
    captured["evidence"] = [item.chunk_id for item in bundle.items]
    raise Stop


Orchestrator._generate_verified = fake_generate
base = load_settings()
shared = build_orchestrator(base)  # one reranker on the GPU; the arms differ only in settings.rag.parallel_search
orchs = {}
for arm, flag in (("parallel", True), ("sequential", False)):
    s = base.model_copy(update={"rag": base.rag.model_copy(update={"parallel_search": flag})})
    orchs[arm] = s


class _Arm:
    def __init__(self, settings):
        self.settings = settings

    def handle(self, *a, **k):
        shared.settings = self.settings
        return shared.handle(*a, **k)


orchs = {arm: _Arm(s) for arm, s in orchs.items()}
for o in orchs.values():  # warm-up
    try:
        o.handle(qs[1], conversation_id="warm")
    except Stop:
        pass
with OUT.open("w") as fh:
    for n, i in enumerate(picks):
        order = ["parallel", "sequential"] if n % 2 == 0 else ["sequential", "parallel"]
        for arm in order:
            captured.clear()
            t0 = time.perf_counter()
            try:
                r = orchs[arm].handle(qs[i], conversation_id=f"{arm}-{i}")
                row = {"q": i, "arm": arm, "kind": r.kind, "pre_gen_s": None}
            except Stop:
                row = {"q": i, "arm": arm, "kind": "answer", "pre_gen_s": round(captured["t"] - t0, 3), "evidence": captured["evidence"]}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
        print(n, flush=True)
