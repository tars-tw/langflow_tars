"""The Langflow flows LibreChat offers a pwc_tars user as tools.

Everyone gets the system account's MCP-enabled flows (the shared ones); each user also gets
their own. Turning a flow's MCP switch on is what publishes it, in any of the owner's projects.

Names follow Langflow's own project MCP server (``build_mcp_tool_name_map``, per project, in
``Flow.id`` order), so a call can be handed straight to its ``handle_call_tool``. On top of
that, a flow's personal tools carry ``my_`` so they never collide with a shared one, and a
name two of an owner's projects both produce is de-duplicated with a numeric suffix.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Literal

from lfx.base.mcp.constants import MAX_MCP_TOOL_NAME_LENGTH
from lfx.base.mcp.util import build_mcp_tool_name_map, get_unique_name
from sqlmodel import col, select

from langflow.services.database.models.flow.model import Flow
from langflow.services.tars.accounts import find_system_account, find_tars_account

if TYPE_CHECKING:
    from uuid import UUID

    from sqlmodel.ext.asyncio.session import AsyncSession

    from langflow.services.database.models.user.model import User

PERSONAL_PREFIX = "my_"
TARS_NODE_TYPES = frozenset({"TarsAgent", "TarsTool"})
MODEL_FIELDS = ("ctx__model_name", "model_name")

Owner = Literal["system", "personal"]


@dataclass(frozen=True)
class TarsTool:
    name: str
    """What LibreChat sees and sends back."""
    flow: Flow
    owner: Owner
    owner_user: User
    project_tool_name: str
    """The flow's name on its own project's MCP server, which ``handle_call_tool`` resolves."""


def flow_tars_model(flow: Flow) -> str | None:
    """The model a flow's TARS node names, or None when it leaves the choice to pwc_tars."""
    for node in (flow.data or {}).get("nodes", []):
        data = node.get("data") or {}
        if data.get("type") not in TARS_NODE_TYPES:
            continue
        template = (data.get("node") or {}).get("template") or {}
        for field in MODEL_FIELDS:
            value = (template.get(field) or {}).get("value")
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


async def _published_flows(db: AsyncSession, owner_id: UUID) -> list[Flow]:
    stmt = (
        select(Flow)
        .where(
            Flow.user_id == owner_id,
            Flow.mcp_enabled == True,  # noqa: E712
            Flow.is_component == False,  # noqa: E712
            col(Flow.folder_id).is_not(None),
        )
        .order_by(col(Flow.folder_id), col(Flow.id))
    )
    return list((await db.exec(stmt)).all())


async def _owner_tools(
    db: AsyncSession, owner_user: User, owner: Owner, prefix: str, taken: set[str]
) -> list[TarsTool]:
    tools: list[TarsTool] = []
    flows = await _published_flows(db, owner_user.id)
    for _, project_flows in groupby(flows, key=lambda flow: flow.folder_id):
        for project_tool_name, flow in build_mcp_tool_name_map(list(project_flows), is_action=True).items():
            name = get_unique_name(f"{prefix}{project_tool_name}", MAX_MCP_TOOL_NAME_LENGTH + len(prefix), taken)
            taken.add(name)
            tools.append(
                TarsTool(
                    name=name,
                    flow=flow,
                    owner=owner,
                    owner_user=owner_user,
                    project_tool_name=project_tool_name,
                )
            )
    return tools


async def collect_tars_tools(db: AsyncSession, tars_id: str | None) -> list[TarsTool]:
    """The shared tools, plus ``tars_id``'s own when that user has an active Langflow account."""
    taken: set[str] = set()
    system = await find_system_account(db)
    tools = await _owner_tools(db, system, "system", "", taken) if system else []
    if not tars_id:
        return tools
    account = await find_tars_account(db, tars_id)
    if account is None or not account.is_active or (system is not None and account.id == system.id):
        return tools
    return tools + await _owner_tools(db, account, "personal", PERSONAL_PREFIX, taken)
