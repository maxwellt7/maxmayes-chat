"""Role gate for admin endpoints.

Decodes the Clerk JWT bearer token and checks ``publicMetadata.role``.
Rejects with 403 when the role is not 'admin'. Token validation
(signature, expiry) is delegated to Clerk's edge-side handling and the
``require_bearer_token`` dependency that this builds on.
"""
import base64
import json

from fastapi import Depends, HTTPException, status

from app.middleware.clerk_auth import require_bearer_token


def _decode_jwt_payload(token: str) -> dict:
    """Best-effort JWT payload extraction. Returns empty dict on malformed
    tokens; the role check below will then 403 the request."""
    parts = token.split(".")
    if len(parts) != 3:
        return {}
    try:
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload)
        data = json.loads(decoded)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def require_admin_role(token: str = Depends(require_bearer_token)) -> dict:
    """Dependency that enforces admin-only access on a route.

    Returns the decoded JWT payload on success (so downstream code can
    access user_id, metadata, etc. without re-decoding). Raises 403 on
    any failure path (missing metadata, wrong role, malformed token).
    """
    payload = _decode_jwt_payload(token)
    metadata = payload.get("publicMetadata") or {}
    role = metadata.get("role") if isinstance(metadata, dict) else None
    if role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return payload
