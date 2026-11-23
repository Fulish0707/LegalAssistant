"""Does Ollama keep the shared answer-system prefix cached across turns when the analyzer call runs in between?
Sends: answer(A) -> analysis(X) -> answer(B) with two never-asked user texts, and reports prefill per call."""
import time
from pathlib import Path

import httpx

P = Path("/home/layke/Documents/MyLLMAgent/src/legal_rag/llm/prompts")
answer_sys = (P / "answer_system.md").read_text()
analyze_sys = (P / "analyze_system.md").read_text()
c = httpx.Client(base_url="http://localhost:11434", timeout=300)
stamp = str(time.time())


def chat(system, user, n=8):
    r = c.post("/api/chat", json={"model": "exaone3.5:7.8b", "stream": False, "keep_alive": "10m",
                                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                                  "options": {"num_ctx": 12288, "num_predict": n, "temperature": 0}})
    b = r.json()
    return {"prompt_tokens": b["prompt_eval_count"], "prefill_ms": round(b["prompt_eval_duration"] / 1e6)}


filler = "근거 자료입니다. " * 600  # ~ evidence-sized user part
print("answer A ", chat(answer_sys, f"[{stamp}-A] 질문: 보증금 반환 {filler}"))
print("answer A2", chat(answer_sys, f"[{stamp}-A2] 질문: 해고 예고 {filler}"))
print("analysis ", chat(analyze_sys, f"[{stamp}-X] 현재 질문: 임금 체불"))
print("answer B ", chat(answer_sys, f"[{stamp}-B] 질문: 교통사고 합의 {filler}"))
print("system only tokens", chat(answer_sys, "", 1))
