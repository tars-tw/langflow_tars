"""Sign-in for the Langflow embedded in LibreChat.

LibreChat points its iframe at ``/api/v1/tars/sso?ticket=...``. A valid ticket
is traded for an ordinary Langflow session (the same cookies ``/login`` sets)
on the Langflow account mapped to the ticket's pwc_tars user, created on first
use, and the browser is sent on to the app. A pwc_tars admin's ticket may instead
name the system account or another user's account: the admin then works in that
workspace without its password (LibreChat audits every such switch).

Every sign-in also records who is operating the browser (the ticket's ``sub``) in an
actor cookie, so TARS components run from the canvas act for that person — see
``langflow.services.tars.actor``. ``/api/v1/tars/logout`` clears it with the session.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from lfx.log.logger import logger

from langflow.api.utils import DbSession
from langflow.initial_setup.setup import get_or_create_default_folder
from langflow.services.auth.exceptions import InactiveUserError
from langflow.services.auth.external import ExternalIdentity
from langflow.services.auth.service import AuthService
from langflow.services.database.models.user.model import User
from langflow.services.deps import get_auth_service, get_settings_service, get_variable_service
from langflow.services.tars.accounts import find_system_account, find_tars_account
from langflow.services.tars.actor import ACTOR_COOKIE, encode_actor
from langflow.services.tars.ticket import (
    TARS_PROVIDER,
    TarsTicket,
    TicketError,
    TicketNotConfiguredError,
    decode_ticket,
    replay_guard,
    ticket_secret,
)

router = APIRouter(tags=["TARS"], include_in_schema=False)

_EXPIRED = (status.HTTP_401_UNAUTHORIZED, "登入連結已失效，請回到主畫面重新開啟 Workflow。")  # noqa: RUF001
_INACTIVE = (status.HTTP_403_FORBIDDEN, "此帳號已停用，無法進入 Workflow。")  # noqa: RUF001
_NOT_ADMIN = (status.HTTP_403_FORBIDDEN, "只有管理員可以切換 Workflow 身分。")
_NO_SYSTEM_ACCOUNT = (status.HTTP_404_NOT_FOUND, "找不到 Workflow 系統帳號，請確認 LANGFLOW_SUPERUSER 設定。")  # noqa: RUF001
_NO_ACCOUNT = (status.HTTP_404_NOT_FOUND, "此使用者尚未進入過 Workflow，沒有可切換的工作區。")  # noqa: RUF001
_NOT_CONFIGURED = (status.HTTP_503_SERVICE_UNAVAILABLE, "Workflow 尚未完成登入設定，請聯絡系統管理員。")  # noqa: RUF001


def _failure(failure: tuple[int, str]) -> HTMLResponse:
    status_code, message = failure
    page = (
        '<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><title>Workflow</title></head>'
        f'<body style="font-family:sans-serif;padding:2rem">{message}</body></html>'
    )
    return HTMLResponse(page, status_code=status_code, headers={"Cache-Control": "no-store"})


async def _resolve_user(ticket: TarsTicket, db: DbSession) -> User:
    """Find or create the Langflow account for the ticket's pwc_tars user.

    The account is keyed by (``tars``, pwc_tars user id) in ``SSOUserProfile``, so a
    rename in pwc_tars keeps the same account. The provider is fixed here rather than
    read from ``EXTERNAL_AUTH_PROVIDER``, which belongs to the separate external
    trusted-identity feature: changing that setting must never orphan these accounts.
    """
    auth = get_auth_service()
    if not isinstance(auth, AuthService):
        msg = f"TARS sign-in needs the built-in auth service, got {type(auth).__name__}"
        raise TypeError(msg)

    identity = ExternalIdentity(
        provider=TARS_PROVIDER,
        subject=ticket.tars_id,
        username=ticket.username,
        email=ticket.email,
        name=ticket.name,
    )
    user = await auth._materialize_external_user(identity, db)  # noqa: SLF001
    if user.is_superuser != ticket.is_admin:
        user.is_superuser = ticket.is_admin
        db.add(user)
        await db.flush()
    return user


async def _resolve_switch_target(ticket: TarsTicket, db: DbSession) -> User | tuple[int, str]:
    """The account an admin asked to enter, or the failure to show instead."""
    user = await find_system_account(db) if ticket.act_as_system else await find_tars_account(db, ticket.act_as or "")
    if user is None:
        return _NO_SYSTEM_ACCOUNT if ticket.act_as_system else _NO_ACCOUNT
    if not user.is_active:
        return _INACTIVE
    logger.info(
        f"TARS admin {ticket.tars_id} entered Workflow as "
        f"{'the system account' if ticket.act_as_system else f'tars user {ticket.act_as}'}"
    )
    return user


def _set_session_cookies(response: Response, tokens: dict, user: User, actor_tars_id: str, secret: str) -> None:
    auth_settings = get_settings_service().auth_settings
    response.set_cookie(
        ACTOR_COOKIE,
        encode_actor(actor_tars_id, str(user.id), auth_settings.REFRESH_TOKEN_EXPIRE_SECONDS, secret),
        httponly=True,
        samesite=auth_settings.REFRESH_SAME_SITE,
        secure=auth_settings.REFRESH_SECURE,
        expires=auth_settings.REFRESH_TOKEN_EXPIRE_SECONDS,
        domain=auth_settings.COOKIE_DOMAIN,
    )
    response.set_cookie(
        "refresh_token_lf",
        tokens["refresh_token"],
        httponly=auth_settings.REFRESH_HTTPONLY,
        samesite=auth_settings.REFRESH_SAME_SITE,
        secure=auth_settings.REFRESH_SECURE,
        expires=auth_settings.REFRESH_TOKEN_EXPIRE_SECONDS,
        domain=auth_settings.COOKIE_DOMAIN,
    )
    response.set_cookie(
        "access_token_lf",
        tokens["access_token"],
        httponly=auth_settings.ACCESS_HTTPONLY,
        samesite=auth_settings.ACCESS_SAME_SITE,
        secure=auth_settings.ACCESS_SECURE,
        expires=auth_settings.ACCESS_TOKEN_EXPIRE_SECONDS,
        domain=auth_settings.COOKIE_DOMAIN,
    )
    response.set_cookie(
        "apikey_tkn_lflw",
        str(user.store_api_key or ""),
        httponly=auth_settings.ACCESS_HTTPONLY,
        samesite=auth_settings.ACCESS_SAME_SITE,
        secure=auth_settings.ACCESS_SECURE,
        expires=None,
        domain=auth_settings.COOKIE_DOMAIN,
    )


@router.get("/sso")
async def sign_in_with_ticket(ticket: str, db: DbSession) -> Response:
    try:
        secret = ticket_secret()
        claims = decode_ticket(ticket, secret)
    except TicketNotConfiguredError as exc:
        logger.error(f"TARS sign-in is not configured: {exc}")
        return _failure(_NOT_CONFIGURED)
    except TicketError as exc:
        logger.warning(f"Rejected a TARS sign-in ticket: {exc}")
        return _failure(_EXPIRED)

    if not replay_guard.redeem(claims):
        logger.warning(f"Rejected a reused TARS sign-in ticket for tars user {claims.tars_id}")
        return _failure(_EXPIRED)

    if claims.acts_as_someone_else:
        if not claims.is_admin:
            logger.warning(f"Refused a Workflow identity switch by non-admin tars user {claims.tars_id}")
            return _failure(_NOT_ADMIN)
        target = await _resolve_switch_target(claims, db)
        if not isinstance(target, User):
            return _failure(target)
        user = target
    else:
        try:
            user = await _resolve_user(claims, db)
        except InactiveUserError:
            logger.warning(f"Refused TARS sign-in for inactive Langflow account of tars user {claims.tars_id}")
            return _failure(_INACTIVE)

    # An admin visiting someone's workspace is not that person signing in.
    tokens = await get_auth_service().create_user_tokens(
        user_id=user.id, db=db, update_last_login=claims.act_as is None
    )
    await get_variable_service().initialize_user_variables(user.id, db)
    await get_or_create_default_folder(db, user.id)

    response = RedirectResponse("/", status_code=status.HTTP_302_FOUND, headers={"Cache-Control": "no-store"})
    _set_session_cookies(response, tokens, user, actor_tars_id=claims.tars_id, secret=secret)
    return response


@router.post("/logout")
async def sign_out() -> Response:
    """End the Langflow session LibreChat opened: the session cookies and the actor cookie.

    LibreChat calls this when its user logs out, from the browser (Langflow is same-site),
    so the next person at that browser cannot open Langflow as the previous one.
    """
    auth_settings = get_settings_service().auth_settings
    response = Response(status_code=status.HTTP_204_NO_CONTENT, headers={"Cache-Control": "no-store"})
    for name, httponly, samesite, secure in (
        (
            "refresh_token_lf",
            auth_settings.REFRESH_HTTPONLY,
            auth_settings.REFRESH_SAME_SITE,
            auth_settings.REFRESH_SECURE,
        ),
        ("access_token_lf", auth_settings.ACCESS_HTTPONLY, auth_settings.ACCESS_SAME_SITE, auth_settings.ACCESS_SECURE),
        ("apikey_tkn_lflw", auth_settings.ACCESS_HTTPONLY, auth_settings.ACCESS_SAME_SITE, auth_settings.ACCESS_SECURE),
        (ACTOR_COOKIE, True, auth_settings.REFRESH_SAME_SITE, auth_settings.REFRESH_SECURE),
    ):
        response.delete_cookie(
            name, httponly=httponly, samesite=samesite, secure=secure, domain=auth_settings.COOKIE_DOMAIN
        )
    return response
