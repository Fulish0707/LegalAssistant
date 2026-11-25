"""Contract tests for both atomic policy-state backends (no network)."""

import importlib
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier

import pytest


@pytest.fixture(params=["memory", "sqlite"])
def store(request, tmp_path):
    name = "legal_rag.rag.policy_state"
    assert importlib.util.find_spec(name) is not None, "policy-state module is required"
    module = importlib.import_module(name)
    if request.param == "memory":
        return module.InMemoryPolicyStateStore()
    return module.SQLitePolicyStateStore(tmp_path / "state.sqlite")


def test_claim_is_exclusive_and_fingerprint_is_checked(store):
    from legal_rag.rag.policy_state import RequestConflict, RequestInProgress, PolicyRequestError

    assert issubclass(RequestConflict, PolicyRequestError)
    assert issubclass(RequestInProgress, PolicyRequestError)
    assert issubclass(PolicyRequestError, RuntimeError)
    assert store.claim_request("c", "r", "fp") is None
    with pytest.raises(RequestInProgress):
        store.claim_request("c", "r", "fp")
    with pytest.raises(RequestConflict):
        store.claim_request("c", "r", "different")
    assert store.claim_request("other", "r", "fp") is None


def test_completed_request_replays_exact_response(store):
    from legal_rag.rag.policy_state import PolicyRequestError, RequestConflict

    response = '{ "answer": "한글", "items": [] }'
    store.claim_request("c", "r", "fp")
    store.complete_request("c", "r", response)
    assert store.claim_request("c", "r", "fp") == response
    with pytest.raises(RequestConflict):
        store.claim_request("c", "r", "other")
    with pytest.raises(PolicyRequestError):
        store.complete_request("c", "r", "overwrite")
    with pytest.raises(PolicyRequestError):
        store.fail_request("c", "r")
    assert store.claim_request("c", "r", "fp") == response


def test_failed_request_never_silently_retries(store):
    from legal_rag.rag.policy_state import PolicyRequestError, RequestFailed, RequestConflict

    assert issubclass(RequestFailed, PolicyRequestError)
    store.claim_request("c", "r", "fp")
    store.fail_request("c", "r")
    store.fail_request("c", "r")
    with pytest.raises(RequestFailed):
        store.claim_request("c", "r", "fp")
    with pytest.raises(RequestConflict):
        store.claim_request("c", "r", "other")
    with pytest.raises(PolicyRequestError):
        store.complete_request("c", "r", "{}")


@pytest.mark.parametrize("operation", ["complete_request", "fail_request"])
def test_unknown_request_cannot_transition(store, operation):
    from legal_rag.rag.policy_state import PolicyRequestError

    args = ("c", "r", "{}") if operation == "complete_request" else ("c", "r")
    with pytest.raises(PolicyRequestError):
        getattr(store, operation)(*args)
    assert store.claim_request("c", "r", "fp") is None


def test_research_budget_is_once_per_claimed_request(store):
    assert not store.research_used("c", "r")
    store.claim_request("c", "r", "fp")
    assert store.reserve_research("c", "r") is True
    assert store.research_used("c", "r") is True
    assert store.reserve_research("c", "r") is False
    store.claim_request("c", "next", "fp")
    assert store.reserve_research("c", "next") is True
    store.claim_request("other", "r", "fp")
    assert store.reserve_research("other", "r") is True


@pytest.mark.parametrize("status", ["unknown", "completed", "failed"])
@pytest.mark.parametrize("used", [False, True])
def test_research_rejects_nonrunning_requests(store, status, used):
    from legal_rag.rag.policy_state import PolicyRequestError

    if status != "unknown":
        store.claim_request("c", "r", "fp")
        if used:
            store.reserve_research("c", "r")
        if status == "completed":
            store.complete_request("c", "r", "{}")
        else:
            store.fail_request("c", "r")
    with pytest.raises(PolicyRequestError):
        store.reserve_research("c", "r")
    assert store.research_used("c", "r") is (used and status != "unknown")
    if status == "unknown":
        assert store.claim_request("c", "r", "fp") is None


def test_parallel_research_reservation_has_one_winner(store):
    store.claim_request("c", "r", "fp")
    barrier = Barrier(8)

    def reserve(_):
        barrier.wait()
        return store.reserve_research("c", "r")

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(8))) == 1


def test_clarification_budget_is_per_conversation_matter_issue(store):
    assert not store.clarification_used("c", "m", "i")
    store.claim_request("c", "r", "fp")
    assert store.reserve_clarification("c", "m", "i") is True
    assert store.clarification_used("c", "m", "i") is True
    assert store.reserve_clarification("c", "m", "i") is False
    store.claim_request("c", "next", "fp")
    assert store.reserve_clarification("c", "m", "i") is False
    assert store.reserve_clarification("c", "m2", "i") is True
    assert store.reserve_clarification("c", "m", "i2") is True
    store.claim_request("other", "r", "fp")
    assert store.reserve_clarification("other", "m", "i") is True


@pytest.mark.parametrize("status", ["unknown", "completed", "failed"])
def test_clarification_requires_running_request_in_same_conversation(store, status):
    from legal_rag.rag.policy_state import PolicyRequestError

    store.claim_request("other", "r", "fp")
    if status != "unknown":
        store.claim_request("c", "r", "fp")
        if status == "completed":
            store.complete_request("c", "r", "{}")
        else:
            store.fail_request("c", "r")
    with pytest.raises(PolicyRequestError):
        store.reserve_clarification("c", "m", "i")
    assert store.clarification_used("c", "m", "i") is False


def test_parallel_clarification_reservation_has_one_winner(store):
    store.claim_request("c", "r", "fp")
    barrier = Barrier(8)

    def reserve(_):
        barrier.wait()
        return store.reserve_clarification("c", "m", "i")

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(8))) == 1


def test_sqlite_reopen_preserves_requests_results_and_spent_budgets(tmp_path):
    from legal_rag.rag.policy_state import (
        PolicyRequestError, RequestConflict, RequestFailed, RequestInProgress,
        SQLitePolicyStateStore,
    )

    path = tmp_path / "nested" / "state.sqlite"
    first = SQLitePolicyStateStore(path)
    first.claim_request("c", "done", "fp")
    assert first.reserve_research("c", "done")
    assert first.reserve_clarification("c", "m", "i")
    first.complete_request("c", "done", '{ "answer": "보존" }')
    first.claim_request("c", "failed", "fp")
    assert first.reserve_research("c", "failed")
    first.fail_request("c", "failed")
    first.claim_request("c", "running", "fp")
    assert first.reserve_research("c", "running")

    reopened = SQLitePolicyStateStore(path)
    assert reopened.claim_request("c", "done", "fp") == '{ "answer": "보존" }'
    with pytest.raises(RequestConflict):
        reopened.claim_request("c", "done", "different")
    with pytest.raises(RequestFailed):
        reopened.claim_request("c", "failed", "fp")
    with pytest.raises(RequestInProgress):
        reopened.claim_request("c", "running", "fp")
    for request_id in ("done", "failed", "running"):
        assert reopened.research_used("c", request_id)
    for request_id in ("done", "failed"):
        with pytest.raises(PolicyRequestError):
            reopened.reserve_research("c", request_id)
    assert reopened.reserve_research("c", "running") is False
    assert reopened.clarification_used("c", "m", "i")
    assert reopened.reserve_clarification("c", "m", "i") is False
    assert reopened.reserve_clarification("c", "m", "new-issue") is True


@pytest.mark.parametrize("operation", ["claim", "research", "clarification"])
def test_sqlite_independent_instances_share_atomic_reservations(tmp_path, operation):
    from legal_rag.rag.policy_state import RequestInProgress, SQLitePolicyStateStore

    path = tmp_path / "shared.sqlite"
    stores = [SQLitePolicyStateStore(path) for _ in range(8)]
    if operation != "claim":
        stores[0].claim_request("c", "r", "fp")
    barrier = Barrier(8)

    def reserve(instance):
        barrier.wait()
        if operation == "research":
            return instance.reserve_research("c", "r")
        if operation == "clarification":
            return instance.reserve_clarification("c", "m", "i")
        try:
            instance.claim_request("c", "r", "fp")
            return True
        except RequestInProgress:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, stores)) == 1


def test_policy_tables_coexist_with_conversation_history(tmp_path):
    import sqlite3
    from legal_rag.rag.conversation import SQLiteConversationStore
    from legal_rag.rag.policy_state import SQLitePolicyStateStore
    from legal_rag.schemas import Turn

    path = tmp_path / "conversation.sqlite"
    history = SQLiteConversationStore(path)
    turn = Turn(role="user", content="기존 기록", created_at=datetime(2026, 1, 1, tzinfo=timezone.utc))
    history.append("c", turn)
    state = SQLitePolicyStateStore(path)
    state.claim_request("c", "r", "fp")
    state.reserve_clarification("c", "m", "i")
    state.reserve_research("c", "r")
    state.complete_request("c", "r", "{}")
    assert SQLiteConversationStore(path).get("c").turns == [turn]
    with sqlite3.connect(path) as conn:
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"turns", "policy_requests", "policy_research", "policy_clarifications"}


def test_parallel_claim_has_exactly_one_winner(store):
    from legal_rag.rag.policy_state import RequestInProgress

    barrier = Barrier(8)

    def claim(_):
        barrier.wait()
        try:
            store.claim_request("c", "r", "fp")
            return True
        except RequestInProgress:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(claim, range(8))) == 1
