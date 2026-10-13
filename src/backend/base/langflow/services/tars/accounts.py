"""Langflow accounts as the TARS integration sees them.

Each pwc_tars user who has entered Workflow owns one Langflow account, linked by an
``SSOUserProfile`` row (provider ``tars``, ``sso_user_id`` = pwc_tars ``sys_user.id``).
The system account is ``LANGFLOW_SUPERUSER``: the shared account that existed before
per-user sign-in, which keeps the flows everyone uses and is never linked to a person.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlmodel import select

from langflow.services.database.models.auth import SSOUserProfile
from langflow.services.database.models.user.crud import get_user_by_id, get_user_by_username
from langflow.services.database.models.user.model import User
from langflow.services.deps import get_settings_service
from langflow.services.tars.ticket import TARS_PROVIDER

if TYPE_CHECKING:
    from datetime import datetime

    from sqlmodel.ext.asyncio.session import AsyncSession


@dataclass(frozen=True)
class TarsAccount:
    tars_id: str
    username: str
    last_login_at: datetime | None


async def find_tars_account(db: AsyncSession, tars_id: str) -> User | None:
    """The Langflow account of a pwc_tars user, or None if they never entered Workflow."""
    stmt = select(SSOUserProfile.user_id).where(
        SSOUserProfile.sso_provider == TARS_PROVIDER,
        SSOUserProfile.sso_user_id == tars_id,
    )
    user_id = (await db.exec(stmt)).first()
    return await get_user_by_id(db, user_id) if user_id else None


async def find_system_account(db: AsyncSession) -> User | None:
    """``LANGFLOW_SUPERUSER``, provided it is a superuser no pwc_tars user is linked to."""
    username = get_settings_service().auth_settings.SUPERUSER
    user = await get_user_by_username(db, username) if username else None
    if user is None or not user.is_superuser:
        return None
    linked = select(SSOUserProfile.id).where(SSOUserProfile.user_id == user.id)
    if (await db.exec(linked)).first() is not None:
        return None
    return user


async def list_tars_accounts(db: AsyncSession) -> list[TarsAccount]:
    """Every active Langflow account linked to a pwc_tars user, by username."""
    stmt = (
        select(SSOUserProfile.sso_user_id, User.username, User.last_login_at)
        .join(User, User.id == SSOUserProfile.user_id)
        .where(SSOUserProfile.sso_provider == TARS_PROVIDER, User.is_active.is_(True))  # type: ignore[attr-defined]
        .order_by(User.username)
    )
    rows = (await db.exec(stmt)).all()
    return [TarsAccount(tars_id=tars_id, username=username, last_login_at=last) for tars_id, username, last in rows]
