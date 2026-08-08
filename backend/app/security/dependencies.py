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
from app.security.owner_bootstrap import email_from_claims, is_allowlisted_owner
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

    Every input is environment-driven — `CLERK_ISSUER`, `CLERK_JWKS_URL`,
    `CLERK_JWT_PUBLIC_KEY`, `CLERK_AUTHORIZED_PARTIES` — so the same image runs
    against a development Clerk instance and a live one with no code change.

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
    """Decide the role for an account we are seeing for the first time.

    Deliberately conservative: the default is member, and the only way out of it
    is an explicit allowlist match on a claim the user cannot edit. Note that
    email will usually be `None` — Clerk session tokens do not carry one unless a
    JWT template adds it — so `OWNER_CLERK_USER_IDS` is the allowlist that can be
    relied on. If neither matches at first sign-in, startup reconciliation and
    `app.scripts.grant_owner` both fix it after the fact; see
    `app.security.owner_bootstrap`.
    """
    if is_allowlisted_owner(clerk_user_id, email):
        return ROLE_OWNER
    return ROLE_MEMBER


def resolve_account(db: Session, verified: VerifiedToken) -> UserAccount:
    """Fetch or create the local account backing a verified Clerk identity.

    Clerk owns authentication; this table owns authorization. The first time we
    see a subject we create a row, applying the owner allowlist. Existing rows
    are never silently promoted *except* by an allowlist match — that repair path
    is what makes setting `OWNER_CLERK_USER_IDS` after the owner's first login
    work. Nothing here can demote.
    """
    email = email_from_claims(verified.claims)

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

    changed = False
    if email and account.email != email:
        account.email = email
        changed = True
    if account.role != ROLE_OWNER and is_allowlisted_owner(
        verified.subject, email or account.email
    ):
        account.role = ROLE_OWNER
        changed = True
        _log.warning(
            "owner_promoted_by_allowlist clerk_user_id=%s", verified.subject
        )
    if changed:
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
        _log.warning(
            "owner_access_denied user_id=%s persona=%s",
            principal.user_id,
            principal.persona.value,
        )
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
