"""One-time sign-in tickets LibreChat issues for the Langflow it embeds.

LibreChat authenticates the user against pwc_tars, then signs a short-lived
HS256 ticket with the secret both sides share (``TARS_LANGFLOW_SSO_SECRET``).
Langflow trades that ticket for its own session, so the user never sees a
Langflow login page. A ticket names the pwc_tars account (``sub`` = its
``sys_user.id``) and whether that account is a pwc_tars admin.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import jwt

TICKET_SECRET_ENV = "TARS_LANGFLOW_SSO_SECRET"  # noqa: S105 # pragma: allowlist secret
TICKET_ALGORITHM = "HS256"
TICKET_ISSUER = "librechat"
TICKET_AUDIENCE = "langflow"
TARS_PROVIDER = "tars"
MIN_SECRET_LENGTH = 32
MAX_TICKET_LIFETIME_SECONDS = 300
CLOCK_LEEWAY_SECONDS = 5


class TicketError(Exception):
    """The ticket is not one this Langflow accepts."""


class TicketNotConfiguredError(TicketError):
    """The shared secret is missing or too short, so no ticket can be verified."""


@dataclass(frozen=True)
class TarsTicket:
    """A verified ticket. ``tars_id`` is always the person operating the browser.

    An admin may ask to enter someone else's workspace instead of their own:
    ``act_as`` names that pwc_tars user, ``act_as_system`` the system account
    (``LANGFLOW_SUPERUSER``). Neither set means their own account.
    """

    tars_id: str
    username: str
    name: str | None
    email: str | None
    is_admin: bool
    jti: str
    expires_at: float
    act_as: str | None = None
    act_as_system: bool = False

    @property
    def acts_as_someone_else(self) -> bool:
        return self.act_as_system or self.act_as is not None


class TicketReplayGuard:
    """Remembers redeemed ticket ids until they expire.

    Process-local on purpose: Langflow runs a single worker (the MCP SSE
    transport LibreChat uses requires it), so one process sees every ticket.
    """

    def __init__(self) -> None:
        self._redeemed: dict[str, float] = {}

    def redeem(self, ticket: TarsTicket, now: float | None = None) -> bool:
        """Mark the ticket used; False when it was already redeemed."""
        current = time.time() if now is None else now
        self._redeemed = {jti: until for jti, until in self._redeemed.items() if until > current}
        if ticket.jti in self._redeemed:
            return False
        self._redeemed[ticket.jti] = ticket.expires_at + CLOCK_LEEWAY_SECONDS
        return True


replay_guard = TicketReplayGuard()


def ticket_secret() -> str:
    secret = os.getenv(TICKET_SECRET_ENV, "").strip()
    if len(secret) < MIN_SECRET_LENGTH:
        msg = f"{TICKET_SECRET_ENV} must be set to at least {MIN_SECRET_LENGTH} characters"
        raise TicketNotConfiguredError(msg)
    return secret


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def decode_ticket(token: str, secret: str) -> TarsTicket:
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[TICKET_ALGORITHM],
            audience=TICKET_AUDIENCE,
            issuer=TICKET_ISSUER,
            leeway=CLOCK_LEEWAY_SECONDS,
            options={"require": ["exp", "iat", "jti", "sub", "aud", "iss"]},
        )
    except jwt.PyJWTError as exc:
        raise TicketError(str(exc)) from exc

    if claims["exp"] - claims["iat"] > MAX_TICKET_LIFETIME_SECONDS:
        msg = f"ticket lifetime exceeds {MAX_TICKET_LIFETIME_SECONDS}s"
        raise TicketError(msg)

    tars_id = _optional_text(claims["sub"])
    jti = _optional_text(claims["jti"])
    if not tars_id or not jti:
        msg = "ticket is missing its subject or id"
        raise TicketError(msg)

    act_as = _optional_text(claims.get("act_as"))
    act_as_system = claims.get("act_as_system") is True
    if act_as and act_as_system:
        msg = "ticket names both a user and the system account to act as"
        raise TicketError(msg)
    if act_as == tars_id:
        act_as = None

    return TarsTicket(
        tars_id=tars_id,
        username=_optional_text(claims.get("preferred_username")) or tars_id,
        name=_optional_text(claims.get("name")),
        email=_optional_text(claims.get("email")),
        is_admin=claims.get("admin") is True,
        jti=jti,
        expires_at=float(claims["exp"]),
        act_as=act_as,
        act_as_system=act_as_system,
    )
