import base64
import json

import pytest
from fastapi import HTTPException

from app.middleware.admin_role import _decode_jwt_payload, require_admin_role


def _make_jwt(payload: dict) -> str:
    """Build a JWT-shaped string (header.payload.signature). Signature is
    not validated by the middleware — Clerk handles that upstream."""
    encoded_payload = base64.urlsafe_b64encode(
        json.dumps(payload).encode()
    ).rstrip(b"=").decode()
    return f"hdr.{encoded_payload}.sig"


def test_decode_jwt_payload_returns_dict_for_valid_token():
    token = _make_jwt({"sub": "u_1", "publicMetadata": {"role": "admin"}})
    payload = _decode_jwt_payload(token)
    assert payload["sub"] == "u_1"
    assert payload["publicMetadata"]["role"] == "admin"


def test_decode_jwt_payload_returns_empty_dict_for_malformed_token():
    assert _decode_jwt_payload("not-a-jwt") == {}
    assert _decode_jwt_payload("only.two") == {}
    assert _decode_jwt_payload("a.invalid_base64!@#.b") == {}


def test_require_admin_role_passes_for_admin():
    token = _make_jwt({"sub": "u_1", "publicMetadata": {"role": "admin"}})
    payload = require_admin_role(token=token)
    assert payload["sub"] == "u_1"


def test_require_admin_role_rejects_non_admin():
    token = _make_jwt({"sub": "u_1", "publicMetadata": {"role": "user"}})
    with pytest.raises(HTTPException) as exc:
        require_admin_role(token=token)
    assert exc.value.status_code == 403


def test_require_admin_role_rejects_missing_metadata():
    token = _make_jwt({"sub": "u_1"})
    with pytest.raises(HTTPException) as exc:
        require_admin_role(token=token)
    assert exc.value.status_code == 403


def test_require_admin_role_rejects_malformed_token():
    with pytest.raises(HTTPException) as exc:
        require_admin_role(token="not-a-jwt")
    assert exc.value.status_code == 403
