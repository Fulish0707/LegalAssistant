# 미채택 시험 정리 기록

정리일: 2026-10-13.

## 정리 범위

미채택 지연 개선 후보와 의미 검증 후보의 시험 공간을 정리했다. 혼합 시험에서 통과한 질의 임베딩 재사용과 사건번호 가로 공백 처리는 현재 작업 폴더에 유지했다.

- 시험용 worktree 14개와 고유 커밋이 없는 브랜치 14개를 제거했다.
- 후보 소스·설정·문서·실행 상태를 먼저 압축 보관했다. 파일 해시와 심볼릭 링크 대상을 확인한 뒤 제거했다.
- 각 시험의 결과 문서, 계획, 설계, 원시 응답, 검토 이유와 측정 기록은 유지했다.
- 현재 애플리케이션의 수정 파일, 코퍼스·인덱스, 모델 가중치, 가상 환경과 다른 작업 공간은 보존했다.

제거한 worktree의 디스크 사용량은 약 159.2 MiB다. 압축 파일은 약 33.4 MiB다. 두 값의 차이는 약 125.8 MiB이며, 정리 메타데이터의 공간은 별도다.

## 보관한 시험

| 시험 공간 | 압축 파일 | 결과 문서 |
|---|---|---|
| `answer-recovery` | [answer-recovery.tar.gz](../../data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/answer-recovery.tar.gz) | [결과](answer-recovery-latency-2026-10-10.md) |
| `citation-latency` | [citation-latency.tar.gz](../../data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/citation-latency.tar.gz) | [결과](citation-latency-2026-10-11.md) |
| `context-budget` | [context-budget.tar.gz](../../data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/context-budget.tar.gz) | [결과](context-budget-2026-10-11.md) |
| `core-first` | [core-first.tar.gz](../../data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/core-first.tar.gz) | [결과](core-first-2026-10-12.md) |
| `evidence-dedup` | [evidence-dedup.tar.gz](../../data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/evidence-dedup.tar.gz) | [결과](evidence-dedup-2026-10-11.md) |
| `first-pass-budget` | [first-pass-budget.tar.gz](../../data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/first-pass-budget.tar.gz) | [결과](first-pass-budget-2026-10-11.md) |
| `kv-cache` | [kv-cache.tar.gz](../../data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/kv-cache.tar.gz) | [결과](kv-cache-2026-10-11.md) |
| `partial-repair` | [partial-repair.tar.gz](../../data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/partial-repair.tar.gz) | [결과](partial-repair-2026-10-12.md) |
| `prefill-batch` | [prefill-batch.tar.gz](../../data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/prefill-batch.tar.gz) | [결과](prefill-batch-2026-10-11.md) |
| `recovery-fixed-failure` | [recovery-fixed-failure.tar.gz](../../data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/recovery-fixed-failure.tar.gz) | [결과](recovery-fixed-failure-2026-10-11.md) |
| `semantic-verification` | [semantic-verification.tar.gz](../../data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/semantic-verification.tar.gz) | [결과](semantic-verification-2026-10-12.md) |
| `summary-first` | [summary-first.tar.gz](../../data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/summary-first.tar.gz) | [결과](summary-first-2026-10-12.md) |
| `temperature-zero` | [temperature-zero.tar.gz](../../data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/temperature-zero.tar.gz) | [결과](temperature-zero-2026-10-11.md) |
| `verified-chat-latency` | [verified-chat-latency.tar.gz](../../data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/verified-chat-latency.tar.gz) | [결과](verified-chat-latency-2026-10-10.md) |

## 보관 위치와 복원

보관 폴더: `data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/`.

- `manifest.json`: 시험별 브랜치, 기준 커밋, 작업 상태, 파일 해시, 링크 대상과 압축 파일 해시.
- `baseline.json`: 정리 전 현재 작업 폴더와 원자료의 보존 기준.
- `removed.json`: 실제 제거한 작업 공간과 브랜치.
- `final-audit.json`: 정리 후 보존 확인 결과.

압축 파일은 원자료 폴더와 함께 로컬에 보관한다. Git에서는 기존 원자료 제외 규칙을 따른다. 시험 문서의 기존 worktree 경로는 당시 실행 위치이며 현재는 존재하지 않는다.

압축 파일에는 시험 소스와 과거 실행 상태가 포함된다. `.git` 연결 파일과 다시 만들 수 있는 Python·pytest 캐시는 제외했다. 가상 환경과 보고서 링크는 링크 자체만 보관했으며 대상 폴더를 복사하지 않았다.

복원하려면 별도 빈 폴더에서 필요한 시험 하나만 풀어 소스를 확인한다. 현재 애플리케이션 폴더에 덮어쓰지 않는다. 예:

```bash
mkdir -p /tmp/myllm-experiment-review
tar -xzf data/eval/reports/chat_pipeline/unsuccessful-cleanup-2026-10-13/semantic-verification.tar.gz -C /tmp/myllm-experiment-review
```

측정 스크립트가 과거 worktree 경로를 사용하면 복원한 소스 위치에 맞춰 경로를 지정해야 한다. 과거 실행 상태는 기록용이며, 새로운 시험은 별도 실행 상태로 시작한다.

이번 정리에서는 추론이나 애플리케이션 테스트를 다시 실행하지 않았다. 압축 파일 대조, Git 상태, 원자료 해시와 보존 경로를 확인했다.
