"""Append-only relevance judgments.

A judgment is identified by (judge model, prompt hash, case, chunk) and is never rewritten.
Changing the model or prompt creates new keys, so judgments from different judges coexist
and are never mixed silently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class JudgeKey:
    model: str
    prompt_sha256: str


class JudgmentStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._rows: dict[tuple, dict] = {}
        if self.path.exists():
            self._repair_unterminated_tail()
            with self.path.open(encoding='utf-8') as handle:
                for line in handle:
                    if line.strip():
                        self._remember(json.loads(line))

    def _repair_unterminated_tail(self) -> None:
        """A process killed mid-write leaves a final line without its newline.

        An undecodable tail was never a judgment, so it is cut off; a decodable one is
        terminated so the next append starts on its own line. Earlier lines are left
        alone and still fail loudly if corrupt.
        """
        data = self.path.read_bytes()
        if not data or data.endswith(b'\n'):
            return
        start = data.rfind(b'\n') + 1
        try:
            json.loads(data[start:].decode('utf-8'))
        except (UnicodeDecodeError, json.JSONDecodeError):
            with self.path.open('r+b') as handle:
                handle.truncate(start)
        else:
            with self.path.open('ab') as handle:
                handle.write(b'\n')

    def _remember(self, row: dict) -> None:
        judge = JudgeKey(row['judge_model'], row['prompt_sha256'])
        self._rows[self._key(judge, row['case_id'], row['chunk_id'])] = row

    @staticmethod
    def _key(judge: JudgeKey, case_id: str, chunk_id: str) -> tuple:
        return (judge.model, judge.prompt_sha256, case_id, chunk_id)

    def has(self, judge: JudgeKey, case_id: str, chunk_id: str) -> bool:
        return self._key(judge, case_id, chunk_id) in self._rows

    def append(self, judge: JudgeKey, *, case_id: str, chunk_id: str, document_id: str,
               grade: int | None, reason: str, raw: str | None = None) -> None:
        key = self._key(judge, case_id, chunk_id)
        if key in self._rows:
            raise ValueError(f'already judged: {key}')
        row = {'case_id': case_id, 'chunk_id': chunk_id, 'document_id': document_id, 'grade': grade,
               'reason': reason, 'judge_model': judge.model, 'prompt_sha256': judge.prompt_sha256,
               'judged_at': datetime.now(timezone.utc).isoformat(), 'raw': raw}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
        self._rows[key] = row

    def _for(self, judge: JudgeKey):
        return (row for (model, prompt, _, _), row in self._rows.items()
                if (model, prompt) == (judge.model, judge.prompt_sha256))

    def grades(self, judge: JudgeKey) -> dict[tuple[str, str], int]:
        return {(row['case_id'], row['chunk_id']): row['grade'] for row in self._for(judge)
                if row['grade'] is not None}

    def documents(self, judge: JudgeKey) -> dict[tuple[str, str], str]:
        return {(row['case_id'], row['chunk_id']): row['document_id'] for row in self._for(judge)}

    def failures(self, judge: JudgeKey) -> list[tuple[str, str]]:
        return sorted((row['case_id'], row['chunk_id']) for row in self._for(judge) if row['grade'] is None)
