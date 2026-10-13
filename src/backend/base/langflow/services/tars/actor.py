"""Who is operating the browser, for TARS components run from the Langflow canvas.

TARS scopes knowledge bases to the pwc_tars user asking. LibreChat's MCP calls name
that user in the ``x-langflow-global-var-tars_user_id`` header, but a run started from
the canvas or playground carries no such header, and when an admin has switched into
another workspace the account running the flow is not the person operating it.

So signing in with a ticket also sets an actor cookie: the operator's pwc_tars id,
bound to the Langflow account that session belongs to. A middleware binds it for the
life of each request, and building a graph for that same account puts it where the
TARS components already look (``request_variables["TARS_USER_ID"]``) unless the run
names a user itself.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jwt

from langflow.services.tars.ticket import (
    CLOCK_LEEWAY_SECONDS,
    TICKET_ALGORITHM,
    TicketNotConfiguredError,
    ticket_secret,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

ACTOR_COOKIE = "tars_actor_lf"
ACTOR_AUDIENCE = "langflow-actor"
ACTOR_ISSUER = "langflow"
TARS_USER_VARIABLE = "TARS_USER_ID"


@dataclass(frozen=True)
class TarsActor:
    tars_id: str
    langflow_user_id: str


_current_actor: ContextVar[TarsActor | None] = ContextVar("tars_actor", default=None)


def encode_actor(tars_id: str, langflow_user_id: str, lifetime_seconds: int, secret: str) -> str:
    now = int(time.time())
    claims = {
        "iss": ACTOR_ISSUER,
        "aud": ACTOR_AUDIENCE,
        "sub": tars_id,
        "uid": langflow_user_id,
        "iat": now,
        "exp": now + lifetime_seconds,
    }
    return jwt.encode(claims, secret, algorithm=TICKET_ALGORITHM)


def decode_actor(token: str | None) -> TarsActor | None:
    """The actor a cookie names, or None for anything missing, forged, expired or unconfigured."""
    if not token:
        return None
    try:
        claims = jwt.decode(
            token,
            ticket_secret(),
            algorithms=[TICKET_ALGORITHM],
            audience=ACTOR_AUDIENCE,
            issuer=ACTOR_ISSUER,
            leeway=CLOCK_LEEWAY_SECONDS,
            options={"require": ["exp", "sub", "uid"]},
        )
    except (jwt.PyJWTError, TicketNotConfiguredError):
        return None
    tars_id, user_id = str(claims["sub"]).strip(), str(claims["uid"]).strip()
    return TarsActor(tars_id=tars_id, langflow_user_id=user_id) if tars_id and user_id else None


@contextmanager
def bind_actor(cookie_value: str | None) -> Iterator[None]:
    """Bind the request's actor (if its cookie verifies) for everything the request runs."""
    token = _current_actor.set(decode_actor(cookie_value))
    try:
        yield
    finally:
        _current_actor.reset(token)


def current_actor() -> TarsActor | None:
    return _current_actor.get()


def apply_actor(graph: Any, langflow_user_id: str | None) -> None:
    """Give a graph built for the bound session's own account the operator's pwc_tars id.

    Only when the actor cookie was issued for this very account, so a cookie left over from
    another session (or a flow run for some other account) never lends its identity, and
    never over a ``TARS_USER_ID`` the run already carries (LibreChat's MCP header wins).
    """
    actor = current_actor()
    if actor is None or not langflow_user_id or actor.langflow_user_id != str(langflow_user_id):
        return
    context = dict(graph.context)
    request_variables = dict(context.get("request_variables") or {})
    if request_variables.get(TARS_USER_VARIABLE):
        return
    request_variables[TARS_USER_VARIABLE] = actor.tars_id
    context["request_variables"] = request_variables
    graph.context = context
