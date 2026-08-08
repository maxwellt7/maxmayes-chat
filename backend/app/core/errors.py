"""Error taxonomy: what the server records versus what the client is told.

The rule this module enforces is that no exception text crosses the network. A
Pinecone client error carries the index and namespace it failed against, so
`str(exc)` on a retrieval failure publishes private index names — `journal`,
`health`, `therapy-sessions` — to whoever provoked it. An OpenAI error carries
part of the request. A SQLAlchemy error carries the SQL.

So every error the client can trigger is mapped to a small closed set of codes
with fixed, generic messages. The detail goes to the log, correlated by
`request_id`, which is also the only variable part of the client response.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClientError:
    """The client-visible form of a failure. Contains nothing derived from an
    exception message, a query, or any provider payload."""

    code: str
    message: str
    status_code: int

    def payload(self, request_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"error": {"code": self.code, "message": self.message}}
        if request_id:
            body["request_id"] = request_id
        return body


# The closed set. Adding a case here is a deliberate decision about what an
# attacker is allowed to learn; deriving one from an exception is not.
INTERNAL = ClientError(
    "internal_error",
    "Something went wrong on our end. Please try again.",
    500,
)
UPSTREAM_UNAVAILABLE = ClientError(
    "upstream_unavailable",
    "A service this request depends on is unavailable. Please try again shortly.",
    502,
)
KNOWLEDGE_BASE_UNAVAILABLE = ClientError(
    "knowledge_base_unavailable",
    "I can't reach my knowledge base right now. Please try again shortly.",
    503,
)
NOT_CONFIGURED = ClientError(
    "not_configured",
    "This feature isn't available right now.",
    503,
)
RATE_LIMITED = ClientError(
    "rate_limited",
    "You're sending requests too quickly. Please wait a moment and try again.",
    429,
)
DAILY_CAPACITY_REACHED = ClientError(
    "daily_capacity_reached",
    "This service has reached its daily capacity. Please try again tomorrow.",
    429,
)


class AppError(Exception):
    """An error whose client-facing form is decided at raise time.

    `internal_detail` is for the log only and is never rendered into a response.
    """

    client_error: ClientError = INTERNAL

    def __init__(self, internal_detail: str = "") -> None:
        super().__init__(internal_detail or self.client_error.code)
        self.internal_detail = internal_detail


class ConfigurationError(AppError):
    client_error = NOT_CONFIGURED


class UpstreamProviderError(AppError):
    client_error = UPSTREAM_UNAVAILABLE


class KnowledgeBaseUnavailableError(AppError):
    client_error = KNOWLEDGE_BASE_UNAVAILABLE


class RateLimitedError(AppError):
    client_error = RATE_LIMITED

    def __init__(self, internal_detail: str = "", retry_after: int = 60) -> None:
        super().__init__(internal_detail)
        self.retry_after = retry_after


class SpendCeilingExceededError(AppError):
    client_error = DAILY_CAPACITY_REACHED


def client_error_for(exc: BaseException) -> ClientError:
    """Map any exception to its client-safe form, defaulting to opaque.

    Unrecognised exceptions become `INTERNAL` rather than being inspected. That
    default is the point of the function: a new provider SDK raising a new
    exception type must not be able to leak by simply not having been considered.
    """
    if isinstance(exc, AppError):
        return exc.client_error
    return INTERNAL


def log_and_convert(
    exc: BaseException,
    *,
    request_id: str | None = None,
    context: str = "",
) -> ClientError:
    """Record the full failure server-side and return what to tell the client."""
    client = client_error_for(exc)
    detail = getattr(exc, "internal_detail", "") or ""
    _log.error(
        "request_failed code=%s request_id=%s context=%s exc=%s detail=%s",
        client.code,
        request_id or "-",
        context or "-",
        type(exc).__name__,
        detail,
        exc_info=True,
    )
    return client
