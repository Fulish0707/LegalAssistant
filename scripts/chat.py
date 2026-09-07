#!/usr/bin/env python
"""Interactive CLI chat against the indexed corpus. The V0.3 milestone interface.

Example:
    uv run python scripts/chat.py --corpus-version v1 --trace
Commands inside the loop: /sources  /trace  /quit
"""

from __future__ import annotations

import uuid
from pathlib import Path

import typer

from legal_rag.cli import settings_from_options
from legal_rag.factory import build_orchestrator
from legal_rag.schemas import ChatResponse

app = typer.Typer(add_completion=False)


def render(response: ChatResponse, show_trace: bool) -> None:
    if response.kind == "clarification":
        typer.echo("답변을 드리기 전에 몇 가지 확인이 필요합니다:")
        for question in response.questions:
            typer.echo(f"  - {question}")
    elif response.kind == "rejected":
        typer.echo(f"근거를 확인할 수 없어 답변을 제공하지 못했습니다. ({response.reason})")
    else:
        answer = response.answer
        typer.echo(f"\n[요약] {answer.summary}")
        if answer.legal_issues:
            typer.echo("[쟁점] " + " / ".join(answer.legal_issues))
        typer.echo(f"[분석] {answer.analysis}")
        if answer.possible_actions:
            typer.echo("[가능한 조치]")
            for action in answer.possible_actions:
                typer.echo(f"  - {action}")
        if answer.additional_information_needed:
            typer.echo("[추가로 필요한 정보] " + " / ".join(answer.additional_information_needed))
        if answer.citations:
            typer.echo("[인용]")
            for citation in answer.citations:
                typer.echo(f"  {citation.source_id} · {citation.reference}")
        if answer.uncertainty:
            typer.echo(f"[불확실성] {answer.uncertainty}")
        typer.echo(f"[신뢰도] {answer.confidence.value}")
    if show_trace and response.trace:
        timings = ", ".join(f"{t.stage}={t.ms:.0f}ms" for t in response.trace.timings)
        typer.echo(f"\n(trace) policy={response.trace.policy_action} regenerations={response.trace.regenerations} {timings}", err=True)
        if response.verification and not response.verification.ok:
            typer.echo("(trace) issues: " + "; ".join(f"{i.kind}: {i.detail}" for i in response.verification.issues), err=True)


@app.command()
def main(
    profile: str = typer.Option(None),
    root: Path = typer.Option(None),
    corpus_version: str = typer.Option(None),
    conversation_id: str = typer.Option(None),
    trace: bool = typer.Option(False, "--trace/--no-trace"),
    embedding_backend: str = typer.Option(None),
    llm_backend: str = typer.Option(None),
) -> None:
    extra = {}
    if embedding_backend:
        extra["embedding"] = {"backend": embedding_backend}
    if llm_backend:
        extra["llm"] = {"backend": llm_backend}
    settings = settings_from_options(profile=profile, root=root, corpus_version=corpus_version, extra=extra or None)
    orchestrator = build_orchestrator(settings)
    conversation_id = conversation_id or uuid.uuid4().hex
    show_trace = trace
    last: ChatResponse | None = None
    typer.echo(f"legal_rag chat · model={settings.llm.model} · corpus={settings.corpus.version} · conversation={conversation_id}")
    typer.echo("질문을 입력하세요. /sources 근거 보기, /trace 추적 토글, /quit 종료")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/trace":
            show_trace = not show_trace
            typer.echo(f"trace {'on' if show_trace else 'off'}")
            continue
        if text == "/sources":
            typer.echo(last.evidence.render() if last and last.evidence else "(근거 없음)")
            continue
        last = orchestrator.handle(text, conversation_id=conversation_id)
        render(last, show_trace)


if __name__ == "__main__":
    app()
