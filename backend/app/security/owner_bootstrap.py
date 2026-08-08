"""Getting the owner role onto the owner's account, without a chicken-and-egg.

The failure this exists to prevent: the owner signs up before
`OWNER_CLERK_USER_IDS` is set, is created as a member, and is now locked out of
the admin surface that would let them fix it. Nothing in the product can grant
the role, because granting roles is an owner-only action.

Two escapes, both of which work after the fact:

1. **Startup reconciliation.** On every boot, any existing account whose Clerk ID
   or email appears in the allowlist is promoted. Setting the environment
   variable and redeploying is therefore sufficient — no login required, no
   ordering requirement between "configure the allowlist" and "first sign-in".
2. **A CLI grant** (`python -m app.scripts.grant_owner <clerk_user_id>`) for the
   case where the allowlist cannot be changed quickly, e.g. a platform whose
   environment edits trigger a slow redeploy.

Promotion is one-directional. Reconciliation never demotes: an allowlist that was
shortened by accident must not silently strip the only owner, and revoking access
is a deliberate act performed against the account, not a side effect of a config
edit.
"""
from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.user_account import ROLE_OWNER, UserAccount

_log = logging.getLogger(__name__)

# Clerk session tokens carry, by default: `sub`, `sid`, `iss`, `azp`, `exp`,
# `iat`, `nbf`, plus version/feature claims (`v`, `fva`, `fea`, `pla`, `o`).
# There is deliberately no email: Clerk keeps session tokens small and expects
# the backend to call the Backend API, or a JWT template to be configured, if it
# needs profile data.
#
# So an email claim may or may not be present depending on whether a custom
# claim has been added to the instance's session token template, and the claim
# name is whatever was chosen there. These are the names that appear in Clerk's
# own template examples and docs; each is checked, and absence is normal rather
# than an error.
#
# The consequence for authorization: email is recorded as a convenience label
# only. `OWNER_CLERK_USER_IDS` is the reliable allowlist because `sub` is always
# present and is not user-editable. `OWNER_EMAILS` works only on instances whose
# token template supplies an email claim, and is treated as best-effort.
_EMAIL_CLAIM_CANDIDATES = (
    "email",
    "email_address",
    "primary_email_address",
    "primary_email",
    "user_email",
)


def email_from_claims(claims: dict) -> str | None:
    """Best-effort email extraction from a *verified* token's claims.

    Only reached after signature verification, so the value is as trustworthy as
    the Clerk instance that minted it. It is still not used as an authorization
    input on its own — see the module docstring.
    """
    for name in _EMAIL_CLAIM_CANDIDATES:
        value = claims.get(name)
        if isinstance(value, str) and "@" in value:
            return value
    # Some templates nest profile data one level down.
    for container in ("user", "public_metadata", "publicMetadata"):
        nested = claims.get(container)
        if isinstance(nested, dict):
            for name in _EMAIL_CLAIM_CANDIDATES:
                value = nested.get(name)
                if isinstance(value, str) and "@" in value:
                    return value
    return None


def is_allowlisted_owner(clerk_user_id: str, email: str | None) -> bool:
    if clerk_user_id in settings.owner_user_id_allowlist:
        return True
    if email and email.lower() in settings.owner_email_allowlist:
        return True
    return False


def reconcile_owner_allowlist(db: Session) -> int:
    """Promote every existing account matching the allowlist. Returns the count.

    Idempotent, so it is safe to run on every boot.
    """
    id_allowlist = settings.owner_user_id_allowlist
    email_allowlist = settings.owner_email_allowlist

    if not id_allowlist and not email_allowlist:
        _log.warning(
            "owner_allowlist_empty: OWNER_CLERK_USER_IDS and OWNER_EMAILS are "
            "both unset. No account can be promoted to owner automatically; "
            "admin endpoints will reject everyone. Set OWNER_CLERK_USER_IDS "
            "and restart, or run `python -m app.scripts.grant_owner <id>`."
        )
        return 0

    accounts = db.execute(
        select(UserAccount).where(UserAccount.role != ROLE_OWNER)
    ).scalars().all()

    promoted = 0
    for account in accounts:
        if is_allowlisted_owner(account.clerk_user_id, account.email):
            account.role = ROLE_OWNER
            promoted += 1
            _log.warning(
                "owner_promoted_by_allowlist clerk_user_id=%s", account.clerk_user_id
            )
    if promoted:
        db.commit()
    return promoted


def warn_if_no_owner(db: Session) -> None:
    """Log loudly when the instance has no owner at all.

    Not fixed automatically: promoting the first account to sign up would hand
    the admin surface, and with it the entire private corpus, to whichever
    stranger registered first. On a public product that is not an acceptable
    bootstrap.
    """
    exists = db.execute(
        select(UserAccount.id).where(UserAccount.role == ROLE_OWNER).limit(1)
    ).first()
    if exists is None:
        _log.warning(
            "no_owner_account: no user_accounts row has role='owner'. Every "
            "admin endpoint will return 403 until one does. Add the Clerk user "
            "ID to OWNER_CLERK_USER_IDS and restart, or run "
            "`python -m app.scripts.grant_owner <clerk_user_id>`."
        )
