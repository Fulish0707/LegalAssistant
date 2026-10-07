"""Test clients that behave as the pages do.

Every unsafe request of the pages carries the session's CSRF token (LC.secureFetch in
static/lc-shell.js), and a chat turn is one POST that streams its answer (sendMessage in
static/lc-chat.js). BrowserClient does the first; chat() does the second.
"""

import json
import uuid

from flask.testing import FlaskClient

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


class BrowserClient(FlaskClient):
    """A Flask test client that sends the session's CSRF token with every unsafe request."""

    def open(self, *args, **kwargs):
        method = kwargs.get("method") or "GET"  # FlaskClient.post and friends pass method=
        if method.upper() not in SAFE_METHODS and not kwargs.pop("without_csrf", False):
            headers = dict(kwargs.pop("headers", None) or {})
            headers.setdefault("X-CSRF-Token", self.csrf_token())
            kwargs["headers"] = headers
        else:
            kwargs.pop("without_csrf", None)
        return super().open(*args, **kwargs)

    def csrf_token(self):
        return super().open("/get_initial_state", method="GET").get_json()["csrf_token"]


def use_browser_clients(app):
    """Make app.test_client() return BrowserClient. Every test client shares 127.0.0.1, so the
    per-IP burst limit is lifted too; test_request_controls checks it on its own. Returns the app."""
    app.test_client_class = BrowserClient
    app.config["REQUEST_BURST_LIMIT"] = 1_000_000
    return app


def chat(client, message, **kwargs):
    """One chat turn as the page sends it: the response streams the answer as server-sent events.

    Buffered unless `buffered=False`: a WSGI server closes a streamed response after sending it,
    which frees the turn's slot (limited_work); the test client closes only a buffered one.
    """
    payload = {"request_id": uuid.uuid4().hex}
    if message is not None:
        payload["message"] = message
    kwargs.setdefault("buffered", True)
    return client.post("/chat", json=payload, **kwargs)


def events(response):
    """The SSE events of a /chat response, in order."""
    return [json.loads(line[len("data: "):]) for line in response.get_data(as_text=True).splitlines()
            if line.startswith("data: ")]
