"""Handing a departed user's flows to the system account.

When a pwc_tars user loses Workflow (their role or groups no longer grant it, their account is
disabled, or it is deleted), LibreChat asks Langflow to move every flow they own into one new
project of the system account, named after them, where admins can review it. The flows arrive
unpublished (MCP off), so nothing becomes visible to everyone until an admin decides it should.
Their own global variables and files are not moved; a flow that used them needs re-pointing.

Idempotent: once moved there is nothing left to move, so repeating a transfer does nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING

from sqlmodel import col, select, update

from langflow.services.database.models.flow.model import Flow
from langflow.services.database.models.flow_version.model import FlowVersion
from langflow.services.database.models.folder.model import Folder
from langflow.services.tars.accounts import find_system_account, find_tars_account

if TYPE_CHECKING:
    from collections.abc import Iterator
    from uuid import UUID

    from sqlmodel.ext.asyncio.session import AsyncSession


class TransferError(Exception):
    """The transfer cannot proceed (no system account to receive the flows)."""


@dataclass(frozen=True)
class TransferResult:
    moved: int
    project: str | None
    deactivated: bool


def _first_free(candidates: Iterator[str], taken: set[str]) -> str:
    return next(candidate for candidate in candidates if candidate not in taken)


# Fullwidth parentheses, as the product's Traditional Chinese labels use them.
_OPEN, _CLOSE = "\uff08", "\uff09"


def _project_names(label: str) -> Iterator[str]:
    yield f"{label}{_OPEN}已移交{_CLOSE}"
    yield from (f"{label}{_OPEN}已移交 {n}{_CLOSE}" for n in count(2))


def _flow_names(original: str, label: str) -> Iterator[str]:
    yield original
    yield f"{original}{_OPEN}{label}{_CLOSE}"
    yield from (f"{original}{_OPEN}{label} {n}{_CLOSE}" for n in count(2))


async def _names_of(db: AsyncSession, owner_id: UUID, column) -> set[str]:
    return {value for value in (await db.exec(select(column).where(Flow.user_id == owner_id))).all() if value}


async def transfer_account(db: AsyncSession, tars_id: str, *, label: str, deactivate: bool) -> TransferResult:
    """Move every flow ``tars_id`` owns into a new system-account project; optionally deactivate them."""
    account = await find_tars_account(db, tars_id)
    if account is None:
        return TransferResult(moved=0, project=None, deactivated=False)
    system = await find_system_account(db)
    if system is None:
        msg = "No system account (LANGFLOW_SUPERUSER) to receive the flows"
        raise TransferError(msg)

    flows = list((await db.exec(select(Flow).where(Flow.user_id == account.id).order_by(col(Flow.id)))).all())
    project: str | None = None
    if flows:
        folder_names = set((await db.exec(select(Folder.name).where(Folder.user_id == system.id))).all())
        project = _first_free(_project_names(label), folder_names)
        folder = Folder(name=project, user_id=system.id, description=f"{label} 失去 Workflow 權限後移交的流程")
        db.add(folder)
        await db.flush()

        flow_names = await _names_of(db, system.id, Flow.name)
        endpoint_names = await _names_of(db, system.id, Flow.endpoint_name)
        for flow in flows:
            flow.name = _first_free(_flow_names(flow.name, label), flow_names)
            flow_names.add(flow.name)
            if flow.endpoint_name:
                if flow.endpoint_name in endpoint_names:
                    flow.endpoint_name = None
                else:
                    endpoint_names.add(flow.endpoint_name)
            flow.user_id = system.id
            flow.folder_id = folder.id
            flow.mcp_enabled = False
            db.add(flow)
        await db.exec(
            update(FlowVersion)
            .where(col(FlowVersion.flow_id).in_([flow.id for flow in flows]))
            .values(user_id=system.id)
        )

    deactivated = False
    if deactivate and account.is_active:
        account.is_active = False
        db.add(account)
        deactivated = True
    await db.flush()
    return TransferResult(moved=len(flows), project=project, deactivated=deactivated)
