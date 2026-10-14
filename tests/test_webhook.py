"""Webhook authentication tests. Uses dummy keys; no request reaches GitHub,
OpenAI or Gemini (clients are constructed but never called)."""
import hashlib
import hmac
import importlib
import json

import pytest


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "s3cret")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    import api.index as app_module
    app_module = importlib.reload(app_module)
    return app_module.app.test_client()


def _sig(body: bytes, secret: str = "s3cret") -> str:
    return "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()


def test_rejects_missing_or_wrong_signature(client):
    body = json.dumps({"action": "opened"}).encode()
    assert client.post("/webhook/github", data=body).status_code == 401
    r = client.post("/webhook/github", data=body,
                    headers={"X-Hub-Signature-256": _sig(body, "wrong"), "X-GitHub-Event": "pull_request"})
    assert r.status_code == 401


def test_accepts_valid_signature_and_ignores_irrelevant_events(client):
    body = json.dumps({"action": "closed"}).encode()
    r = client.post("/webhook/github", data=body, content_type="application/json",
                    headers={"X-Hub-Signature-256": _sig(body), "X-GitHub-Event": "pull_request"})
    assert r.status_code == 200 and r.get_json() == {"status": "ignored"}

    r = client.post("/webhook/github", data=body, content_type="application/json",
                    headers={"X-Hub-Signature-256": _sig(body), "X-GitHub-Event": "push"})
    assert r.get_json() == {"status": "ignored"}


def test_refuses_all_webhooks_when_no_secret_configured(client, monkeypatch):
    import api.index as app_module
    monkeypatch.setattr(app_module, "GITHUB_WEBHOOK_SECRET", None)
    body = b"{}"
    r = client.post("/webhook/github", data=body,
                    headers={"X-Hub-Signature-256": _sig(body), "X-GitHub-Event": "pull_request"})
    assert r.status_code == 401
