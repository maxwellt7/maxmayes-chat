"""Grant (or revoke) the owner role from the command line.

The recovery path for the case the allowlist cannot cover: the owner has already
signed up as a member and there is no in-app way to promote them, because
promoting is itself an owner-only action.

    python -m app.scripts.grant_owner user_2abc...
    python -m app.scripts.grant_owner --revoke user_2abc...
    python -m app.scripts.grant_owner --list

Runs against whatever `DATABASE_URL` is configured for the shell it is invoked
from, so it is usable locally and inside a deployed container's shell.
"""
from __future__ import annotations

import argparse
import sys

from sqlalchemy import select

from app.db.database import SessionLocal
from app.models.user_account import ROLE_MEMBER, ROLE_OWNER, UserAccount


def _list_accounts() -> int:
    db = SessionLocal()
    try:
        rows = db.execute(
            select(UserAccount).order_by(UserAccount.created_at)
        ).scalars().all()
        if not rows:
            print("no user_accounts rows yet — nobody has signed in")
            return 0
        for row in rows:
            print(
                f"{row.clerk_user_id}\trole={row.role}\tactive={row.is_active}\t"
                f"email={row.email or '-'}"
            )
        return 0
    finally:
        db.close()


def _set_role(clerk_user_id: str, role: str) -> int:
    db = SessionLocal()
    try:
        account = db.execute(
            select(UserAccount).where(UserAccount.clerk_user_id == clerk_user_id)
        ).scalar_one_or_none()
        if account is None:
            print(
                f"no account for {clerk_user_id!r}. The user must sign in once "
                "so the account row exists, then re-run this.",
                file=sys.stderr,
            )
            return 1
        previous = account.role
        account.role = role
        db.commit()
        print(f"{clerk_user_id}: {previous} -> {role}")
        return 0
    finally:
        db.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clerk_user_id", nargs="?", help="Clerk user ID (sub claim)")
    parser.add_argument(
        "--revoke",
        action="store_true",
        help="demote to member instead of promoting to owner",
    )
    parser.add_argument(
        "--list", action="store_true", help="list all accounts and their roles"
    )
    args = parser.parse_args(argv)

    if args.list:
        return _list_accounts()
    if not args.clerk_user_id:
        parser.error("clerk_user_id is required unless --list is given")
    return _set_role(
        args.clerk_user_id, ROLE_MEMBER if args.revoke else ROLE_OWNER
    )


if __name__ == "__main__":
    raise SystemExit(main())
