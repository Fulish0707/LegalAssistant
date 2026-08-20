"""Atomic, privacy-preserving daily AI request quotas."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import ipaddress


DAILY_AI_REQUEST_LIMIT = 20


@dataclass(frozen=True)
class QuotaResult:
    allowed: bool
    used: int
    remaining: int
    limit: int
    quota_day: str


def utc_quota_day(now=None):
    """Return the current UTC calendar day as an ISO date."""
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).date().isoformat()


def normalize_client_ip(value):
    """Canonicalize an address before deriving its non-reversible quota identity."""
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    try:
        return str(ipaddress.ip_address(value.strip()))
    except ValueError:
        return "unknown"


def guest_ip_identity(client_ip, salt):
    """Hash a normalized guest address; raw addresses are never stored in SQLite."""
    if not isinstance(salt, str) or not salt:
        raise ValueError("A quota IP salt is required")
    normalized_ip = normalize_client_ip(client_ip)
    return hashlib.sha256(f"{salt}:{normalized_ip}".encode("utf-8")).hexdigest()


def client_ip_from_request(request, *, trusted_proxy_count=0):
    """Get the client address, trusting forwarded hops only when configured."""
    if not isinstance(trusted_proxy_count, int) or trusted_proxy_count < 0:
        raise ValueError("Trusted proxy count must be a non-negative integer")
    if trusted_proxy_count == 0:
        return normalize_client_ip(request.remote_addr)

    # Flask exposes only X-Forwarded-For values through access_route.  Add the
    # direct peer so the final ``trusted_proxy_count`` entries are the proxy
    # hops we explicitly trust.
    route = list(request.access_route) + [request.remote_addr]
    if len(route) > trusted_proxy_count:
        return normalize_client_ip(route[-(trusted_proxy_count + 1)])
    return normalize_client_ip(request.remote_addr)


def reserve_daily_quota(connection, identity_kind, identity_value, *, limit=DAILY_AI_REQUEST_LIMIT, now=None):
    """Atomically reserve one request, refusing reservations beyond ``limit``."""
    if identity_kind not in {"user", "guest_ip"}:
        raise ValueError("Unknown quota identity kind")
    if not isinstance(identity_value, str) or not identity_value:
        raise ValueError("A quota identity is required")
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("Quota limit must be a positive integer")

    quota_day = utc_quota_day(now)
    row = connection.execute(
        """
        INSERT INTO daily_ai_quota (identity_kind, identity_value, quota_day, request_count)
        VALUES (?, ?, ?, 1)
        ON CONFLICT(identity_kind, identity_value, quota_day) DO UPDATE SET
            request_count = daily_ai_quota.request_count + 1
        WHERE daily_ai_quota.request_count < ?
        RETURNING request_count
        """,
        (identity_kind, identity_value, quota_day, limit),
    ).fetchone()
    connection.commit()
    if row is None:
        used = connection.execute(
            """
            SELECT request_count FROM daily_ai_quota
            WHERE identity_kind=? AND identity_value=? AND quota_day=?
            """,
            (identity_kind, identity_value, quota_day),
        ).fetchone()[0]
        return QuotaResult(False, used, 0, limit, quota_day)

    used = row[0]
    return QuotaResult(True, used, limit - used, limit, quota_day)
