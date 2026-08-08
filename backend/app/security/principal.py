"""Who is asking, and what they are allowed to see.

`Persona` is the single authorization primitive the answer engine understands.
It is derived server-side from a verified identity and a database role. It is
never accepted from a request body, query parameter, or token claim, and no
downstream stage may widen it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.models.user_account import ROLE_OWNER


class Persona(str, Enum):
    """Trust tier governing corpus access.

    Ordered from least to most privileged. Comparison helpers below rely on
    that ordering, so do not reorder without updating `_RANK`.
    """

    PUBLIC = "public"
    MEMBER = "member"
    OWNER = "owner"


_RANK = {Persona.PUBLIC: 0, Persona.MEMBER: 1, Persona.OWNER: 2}


def persona_for_role(role: str | None) -> Persona:
    """Map a stored account role onto a persona.

    Anything unrecognised — including `None` — resolves to the least privileged
    persona. Failing closed is the only acceptable default when the input is a
    role string that may have come from an older migration or a typo.
    """
    if role == ROLE_OWNER:
        return Persona.OWNER
    if role is None:
        return Persona.PUBLIC
    return Persona.MEMBER


@dataclass(frozen=True)
class Principal:
    """An authenticated caller."""

    user_id: str
    persona: Persona
    session_id: str | None = None
    email: str | None = None

    @property
    def is_owner(self) -> bool:
        return self.persona is Persona.OWNER

    def at_least(self, minimum: Persona) -> bool:
        return _RANK[self.persona] >= _RANK[minimum]


# The caller for unauthenticated surfaces — public web chat, Instagram DMs.
# It has a stable synthetic user_id so telemetry still groups sensibly, and the
# most restrictive persona.
def anonymous_principal(external_id: str) -> Principal:
    return Principal(user_id=external_id, persona=Persona.PUBLIC)
