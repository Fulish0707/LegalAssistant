"""Download current laws from law.go.kr as statute_json_v1, ready for scripts/ingest.py.

The key is the OC value registered at open.law.go.kr, read from LAW_GO_KR_OC (environment or .env). Without it the public
test key is used, which worked for the pilot but is not meant for a bulk load. Output files are named by the API's law number,
so a rerun skips laws that are already saved. Laws without article numbers (area-change rules and similar) are listed in
_skipped.log (JSON lines; the .log suffix keeps ingest.py from reading it as records) and not written.

Example:
    uv run python -m legal_rag.ingestion.law_api_cli --out-dir data/api_laws/statute-api-full
    uv run python scripts/ingest.py --raw-dir data/api_laws/statute-api-full --corpus-version statute-api-full-v1
    uv run python scripts/build_index.py --corpus-version statute-api-full-v1

A full load is 5,600 laws, about 80 minutes of downloading and about 315,000 chunks; embedding them takes hours. Qdrant holds
roughly 1.3 GiB for the result.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import typer
from dotenv import dotenv_values

from legal_rag.ingestion.law_api import LawApiError, fetch_law, list_current_laws

ROOT = Path(__file__).resolve().parents[3]

app = typer.Typer(add_completion=False)


@app.command()
def main(
    out_dir: Path = typer.Option(..., help="Directory that receives one JSON file per law."),
    names: list[str] = typer.Option(None, "--name", help="Only laws with exactly this name (repeatable); default is every current law."),
    limit: int = typer.Option(0, help="Stop after this many laws (smoke test)."),
    delay: float = typer.Option(0.2, help="Seconds between requests."),
) -> None:
    oc = os.environ.get("LAW_GO_KR_OC") or dotenv_values(ROOT / ".env").get("LAW_GO_KR_OC") or ""
    if not oc:
        typer.echo("LAW_GO_KR_OC is not set; using the public test key (fine for a few laws, not for a bulk load).", err=True)
        oc = "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    squash = lambda s: re.sub(r"\s+", "", s)
    entries = list_current_laws(oc)
    if names:
        wanted = {squash(n) for n in names}
        entries = [e for e in entries if squash(e["법령명한글"]) in wanted]
    entries = entries[:limit] if limit else entries
    typer.echo(f"{len(entries)} laws to fetch")
    saved = skipped = 0
    started = time.perf_counter()
    with (out_dir / "_skipped.log").open("a", encoding="utf-8") as skip_log:
        for number, entry in enumerate(entries, 1):
            target = out_dir / f"{entry['법령일련번호']}.json"
            if target.exists():
                saved += 1
                continue
            try:
                record = fetch_law(oc, entry)
            except LawApiError as error:
                raise typer.BadParameter(f"the API rejected the request: {error}") from error
            if record is None:
                skipped += 1
                skip_log.write(json.dumps({"mst": entry["법령일련번호"], "name": entry["법령명한글"], "kind": entry["법령구분명"]}, ensure_ascii=False) + "\n")
            else:
                record["law_name"] = entry["법령명한글"]  # the list name is the one other corpora and the eval match on
                target.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
                saved += 1
            if number % 200 == 0:
                typer.echo(f"{number}/{len(entries)} saved={saved} skipped={skipped} {time.perf_counter() - started:.0f}s")
            time.sleep(delay)
    typer.echo(f"done: saved={saved} skipped={skipped} (no article structure)")


if __name__ == "__main__":
    app()
