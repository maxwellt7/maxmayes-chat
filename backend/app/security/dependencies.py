"""FastAPI dependencies for authentication and authorization.

Every protected route depends on `get_current_principal`, which verifies the
Clerk session token cryptographically and resolves the caller's persona from
our own database. Routes that mutate configuration or expose corpus internals
additionally depend on `require_owner`.
"""
from __future__ import annotations

import logging

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db
from app.models.user_account import ROLE_MEMBER, ROLE_OWNER, UserAccount
from app.security.clerk_jwt import (
    ClerkTokenVerifier,
    TokenVerificationError,
    VerifiedToken,
)
from app.security.principal import Persona, Principal, persona_for_role

_log = logging.getLogger(__name__)

_UNAUTHORIZED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
    headers={"WWW-Authenticate": "Bearer"},
)

_verifier: ClerkTokenVerifier | None = None
_verifier_configured = False


def get_verifier() -> ClerkTokenVerifier:
    """Build the process-wide token verifier from settings.

    Raises at request time rather than import time so the app can still boot
    (and serve /health) with incomplete configuration — but every authenticated
    request will fail loudly until Clerk is configured. That is deliberate: a
    misconfigured auth layer must never silently allow traffic through.
    """
    global _verifier, _verifier_configured
    if _verifier_configured:
        if _verifier is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Authentication is not configured",
            )
        return _verifier

    jwks_url = settings.resolved_clerk_jwks_url
    pem = settings.clerk_jwt_public_key.strip() or None
    if not jwks_url and not pem:
        _verifier_configured = True
        _log.error(
            "Clerk verification is unconfigured: set CLERK_ISSUER (or "
            "CLERK_JWKS_URL, or CLERK_JWT_PUBLIC_KEY). All authenticated "
            "requests will be rejected."
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication is not configured",
        )

    _verifier = ClerkTokenVerifier(
        jwks_url=jwks_url or None,
        public_key_pem=pem,
        issuer=settings.clerk_issuer or None,
        authorized_parties=settings.authorized_parties,
    )
    _verifier_configured = True
    return _verifier


def reset_verifier_cache() -> None:
    """Test hook — forces the next `get_verifier()` call to rebuild."""
    global _verifier, _verifier_configured
    _verifier = None
    _verifier_configured = False


def _bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise _UNAUTHORIZED
    scheme, _, credentials = authorization.partition(" ")
    if scheme.lower() != "bearer" or not credentials.strip():
        raise _UNAUTHORIZED
    return credentials.strip()


async def get_verified_token(
    authorization: str | None = Header(default=None),
    verifier: ClerkTokenVerifier = Depends(get_verifier),
) -> VerifiedToken:
    token = _bearer_token(authorization)
    try:
        return verifier.verify(token)
    except TokenVerificationError:
        raise _UNAUTHORIZED from None


def _initial_role(clerk_user_id: str, email: str | None) -> str:
    """Decide the role for an account we are seeing for the first time."""
    if clerk_user_id in settings.owner_user_id_allowlist:
        return ROLE_OWNER
    if email and email.lower() in settings.owner_email_allowlist:
        return ROLE_OWNER
    return ROLE_MEMBER


def resolve_account(db: Session, verified: VerifiedToken) -> UserAccount:
    """Fetch or create the local account backing a verified Clerk identity.

    Clerk owns authentication; this table owns authorization. The first time we
    see a subject we create a row, applying the owner allowlist. Existing rows
    are never silently promoted — a role change is an explicit admin action —
    but an account already on the allowlist is repaired if it was demoted by a
    bad migration.
    """
    email = verified.claims.get("email")
    if not isinstance(email, str):
        email = None

    account = (
        db.query(UserAccount)
        .filter(UserAccount.clerk_user_id == verified.subject)
        .one_or_none()
    )
    if account is None:
        account = UserAccount(
            clerk_user_id=verified.subject,
            email=email,
            role=_initial_role(verified.subject, email),
        )
        db.add(account)
        db.commit()
        db.refresh(account)
        return account

    if account.role != ROLE_OWNER and _initial_role(
        verified.subject, email or account.email
    ) == ROLE_OWNER:
        account.role = ROLE_OWNER
        db.commit()

    return account


async def get_current_principal(
    verified: VerifiedToken = Depends(get_verified_token),
    db: Session = Depends(get_db),
) -> Principal:
    account = resolve_account(db, verified)
    if not account.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Account is disabled"
        )
    return Principal(
        user_id=account.clerk_user_id,
        persona=persona_for_role(account.role),
        session_id=verified.session_id,
        email=account.email,
    )


async def require_owner(
    principal: Principal = Depends(get_current_principal),
) -> Principal:
    if not principal.is_owner:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Owner access required"
        )
    return principal


async def require_member(
    principal: Principal = Depends(get_current_principal),
) -> Principal:
    if not principal.at_least(Persona.MEMBER):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Membership required"
        )
    return principal
