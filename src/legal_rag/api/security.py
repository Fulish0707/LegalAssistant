"""Authenticated internal service boundary. Only the web service can assert a workspace."""

from collections import OrderedDict
from contextlib import contextmanager
import hashlib
import hmac
import re
import ipaddress
from urllib.parse import urlsplit
from threading import BoundedSemaphore, Lock
import time

from fastapi import HTTPException, Request
from starlette.responses import JSONResponse


class RequestSecurity:
    def __init__(self, api):
        self.api = api
        self.slots = BoundedSemaphore(api.max_concurrent_requests)
        self.lock = Lock()
        self.windows = OrderedDict()

    def authenticate(self, request):
        key = self.api.service_token.get_secret_value() if self.api.service_token else ""
        if len(key) < 32 or key.startswith("replace-"):
            raise HTTPException(503, "service_auth_not_configured")
        if not hmac.compare_digest(request.headers.get("Authorization", "").encode(), ("Bearer " + key).encode()):
            raise HTTPException(401, "authentication_required")
        owner = request.headers.get("X-Workspace-ID", "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{16,128}", owner):
            raise HTTPException(400, "workspace_required")
        # Stable across service-token rotation; this header is accepted only from
        # the trusted service. Browser identities remain random, signed cookies.
        return hashlib.sha256(owner.encode()).hexdigest()

    def reserve(self, owner):
        bucket = int(time.monotonic()) // 60
        with self.lock:
            for key in list(self.windows):
                if self.windows[key][0] != bucket:
                    del self.windows[key]
            _, count = self.windows.get(owner, (bucket, 0))
            if count >= self.api.requests_per_minute or len(self.windows) >= 10000 and owner not in self.windows:
                raise HTTPException(429, "rate_limit_exceeded")
            self.windows[owner] = (bucket, count + 1)

    @contextmanager
    def work(self):
        if not self.slots.acquire(blocking=False):
            raise HTTPException(429, "service_busy")
        try:
            yield
        finally:
            self.slots.release()


def workspace(request: Request):
    security = request.app.state.security
    owner = security.authenticate(request)
    security.reserve(owner)
    return owner


def conversation_key(owner, conversation_id):
    return owner + ":" + hashlib.sha256(conversation_id.encode()).hexdigest()


def is_loopback_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or parsed.username or parsed.password or not parsed.hostname:
        return False
    if parsed.hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        return False


def check_egress(settings, *, documents=False):
    remote_model = settings.llm.backend == "gemini" or (
        settings.llm.backend == "ollama" and not is_loopback_url(settings.llm.base_url))
    remote_embedding = settings.embedding.backend == "ollama" and not is_loopback_url(settings.embedding.base_url)
    for enabled, url in ((settings.llm.backend == "ollama" and remote_model, settings.llm.base_url),
                         (remote_embedding, settings.embedding.base_url)):
        parsed = urlsplit(url)
        if enabled and (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password):
            raise HTTPException(503, "model_tls_required")
    if (remote_model or remote_embedding) and not settings.api.allow_external_models:
        raise HTTPException(503, "external_model_disabled")
    if documents and remote_model and not settings.api.allow_user_document_egress:
        raise HTTPException(403, "document_egress_disabled")


class BodyLimitMiddleware:
    """ASGI streaming byte limit; rejects before JSON decoding, including chunked bodies."""
    def __init__(self, app, limit=32768):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            event = await receive()
            if event["type"] == "http.disconnect":
                return
            body.extend(event.get("body", b""))
            if len(body) > self.limit:
                return await JSONResponse({"error": "payload_too_large"}, 413)(scope, receive, send)
            if not event.get("more_body"):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        async def private_send(event):
            if event["type"] == "http.response.start":
                event["headers"] = list(event.get("headers", [])) + [(b"cache-control", b"no-store"), (b"x-content-type-options", b"nosniff")]
            await send(event)
        await self.app(scope, bounded_receive, private_send)
