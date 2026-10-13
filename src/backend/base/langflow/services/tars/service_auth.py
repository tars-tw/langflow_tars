"""Server-to-server calls from LibreChat's backend to Langflow's ``/api/v1/tars`` service routes.

LibreChat signs each call with a short-lived HS256 bearer using the same shared
secret as the sign-in tickets, under its own audience so a sign-in ticket can
never be replayed as a service call and vice versa.
"""

from __future__ import annotations

from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from lfx.log.logger import logger

from langflow.services.tars.ticket import (
    CLOCK_LEEWAY_SECONDS,
    TICKET_ALGORITHM,
    TICKET_ISSUER,
    TicketNotConfiguredError,
    ticket_secret,
)

SERVICE_AUDIENCE = "langflow-service"
MAX_SERVICE_TOKEN_LIFETIME_SECONDS = 120

_bearer = HTTPBearer(auto_error=False)


class ServiceTokenError(Exception):
    """The bearer is not a service token this Langflow accepts."""


def verify_service_token(token: str, secret: str) -> None:
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[TICKET_ALGORITHM],
            audience=SERVICE_AUDIENCE,
            issuer=TICKET_ISSUER,
            leeway=CLOCK_LEEWAY_SECONDS,
            options={"require": ["exp", "iat", "aud", "iss"]},
        )
    except jwt.PyJWTError as exc:
        raise ServiceTokenError(str(exc)) from exc
    if claims["exp"] - claims["iat"] > MAX_SERVICE_TOKEN_LIFETIME_SECONDS:
        msg = f"service token lifetime exceeds {MAX_SERVICE_TOKEN_LIFETIME_SECONDS}s"
        raise ServiceTokenError(msg)


async def require_tars_service(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    """Route dependency: only LibreChat's backend, holding the shared secret, gets through."""
    try:
        secret = ticket_secret()
    except TicketNotConfiguredError as exc:
        logger.error(f"TARS service call refused, sign-in is not configured: {exc}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Not configured") from exc
    if credentials is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing service token")
    try:
        verify_service_token(credentials.credentials, secret)
    except ServiceTokenError as exc:
        logger.warning(f"Rejected a TARS service token: {exc}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid service token") from exc


TarsService = Depends(require_tars_service)
