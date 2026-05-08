import base64
import json

from fastapi import Header, HTTPException, status


async def require_bearer_token(
    authorization: str | None = Header(default=None),
) -> str:
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format",
        )

    return parts[1].strip()


def extract_user_id(token: str) -> str:
    """
    Extract a stable user_id from a bearer token.
    For Clerk JWTs, returns the `sub` claim (e.g., 'user_2abc...').
    For non-JWT tokens (Phase 1 placeholder), returns the token itself truncated.
    """
    parts = token.split(".")
    if len(parts) == 3:
        try:
            payload = parts[1]
            payload += "=" * (-len(payload) % 4)
            decoded = base64.urlsafe_b64decode(payload)
            data = json.loads(decoded)
            sub = data.get("sub")
            if sub and isinstance(sub, str):
                return sub[:255]
        except Exception:
            pass
    return token[:255]
