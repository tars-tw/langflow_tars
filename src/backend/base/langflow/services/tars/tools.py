"""The Langflow flows LibreChat offers a pwc_tars user as tools.

Everyone gets the system account's MCP-enabled flows (the shared ones); each user also gets
their own. Turning a flow's MCP switch on is what publishes it, in any of the owner's projects.

LibreChat reads them one project at a time (``collect_tars_projects``): each project is its own
MCP server there, and its tools carry exactly the names Langflow's own project MCP server gives
them (``build_mcp_tool_name_map``, per project, in ``Flow.id`` order), so a call can be handed
straight to its ``handle_call_tool``.

The older aggregate view (``collect_tars_tools``) flattens every project into one list: a flow's
personal tools carry ``my_`` so they never collide with a shared one, and a name two of an
owner's projects both produce is de-duplicated with a numeric suffix.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Literal

from lfx.base.mcp.constants import MAX_MCP_TOOL_NAME_LENGTH
from lfx.base.mcp.util import build_mcp_tool_name_map, get_unique_name
from sqlmodel import col, select

from langflow.services.database.models.flow.model import Flow
from langflow.services.database.models.folder.model import Folder
from langflow.services.tars.accounts import find_system_account, find_tars_account, list_tars_account_users

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


def _is_tars_node(node_type: object) -> bool:
    """Whether a node is a TARS node, built in (``TarsAgent``) or from an extension bundle.

    An extension component's type is ``ext:<bundle>:<class name>@<slot>``
    (``ext:tars:TarsAgentComponent@extra``), and its class name carries ``Component``.
    """
    if not isinstance(node_type, str):
        return False
    if node_type.startswith("ext:"):
        node_type = node_type.rsplit(":", 1)[-1].split("@", 1)[0].removesuffix("Component")
    return node_type in TARS_NODE_TYPES


def flow_tars_model(flow: Flow) -> str | None:
    """The model a flow's TARS node names, or None when it leaves the choice to pwc_tars."""
    for node in (flow.data or {}).get("nodes", []):
        data = node.get("data") or {}
        if not _is_tars_node(data.get("type")):
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


@dataclass(frozen=True)
class TarsProject:
    """One of an owner's projects holding at least one published flow."""

    id: UUID
    name: str
    description: str | None
    owner: Owner
    owner_user: User
    tars_id: str | None
    """The pwc_tars user a personal project belongs to; None for the system account's."""
    tools: list[TarsTool]
    """Named as the project's own MCP server names them."""


async def _owner_projects(db: AsyncSession, owner_user: User, owner: Owner, tars_id: str | None) -> list[TarsProject]:
    flows = await _published_flows(db, owner_user.id)
    if not flows:
        return []
    folder_ids = {flow.folder_id for flow in flows}
    rows = await db.exec(select(Folder).where(col(Folder.id).in_(folder_ids)))
    folders = {folder.id: folder for folder in rows.all()}
    projects: list[TarsProject] = []
    for folder_id, project_flows in groupby(flows, key=lambda flow: flow.folder_id):
        folder = folders.get(folder_id)
        if folder is None:
            continue
        tools = [
            TarsTool(name=name, flow=flow, owner=owner, owner_user=owner_user, project_tool_name=name)
            for name, flow in build_mcp_tool_name_map(list(project_flows), is_action=True).items()
        ]
        projects.append(
            TarsProject(
                id=folder_id,
                name=folder.name,
                description=folder.description,
                owner=owner,
                owner_user=owner_user,
                tars_id=tars_id,
                tools=tools,
            )
        )
    return projects


async def _chatting_account(db: AsyncSession, tars_id: str | None, system: User | None) -> User | None:
    """``tars_id``'s active Langflow account, unless it is the system account itself."""
    if not tars_id:
        return None
    account = await find_tars_account(db, tars_id)
    if account is None or not account.is_active or (system is not None and account.id == system.id):
        return None
    return account


async def collect_tars_projects(db: AsyncSession, tars_id: str | None) -> list[TarsProject]:
    """The shared projects, plus ``tars_id``'s own when that user has an active Langflow account."""
    system = await find_system_account(db)
    projects = await _owner_projects(db, system, "system", None) if system else []
    account = await _chatting_account(db, tars_id, system)
    if account is None:
        return projects
    return projects + await _owner_projects(db, account, "personal", tars_id)


async def find_tars_project(db: AsyncSession, tars_id: str | None, project_id: UUID) -> TarsProject | None:
    """The project if ``tars_id`` may use it: a shared one, or their own."""
    return next((project for project in await collect_tars_projects(db, tars_id) if project.id == project_id), None)


async def list_all_tars_projects(db: AsyncSession) -> list[TarsProject]:
    """Every project LibreChat may have to serve: the shared ones and each pwc_tars user's own.

    LibreChat registers one MCP server per project up front and hides the personal ones from
    everyone but their owner, so this is the whole set rather than one user's view.
    """
    system = await find_system_account(db)
    projects = await _owner_projects(db, system, "system", None) if system else []
    for tars_id, account in await list_tars_account_users(db):
        if system is not None and account.id == system.id:
            continue
        projects += await _owner_projects(db, account, "personal", tars_id)
    return projects


def _aggregate_tools(projects: list[TarsProject], prefix: str, taken: set[str]) -> list[TarsTool]:
    tools: list[TarsTool] = []
    for project in projects:
        for tool in project.tools:
            name = get_unique_name(f"{prefix}{tool.project_tool_name}", MAX_MCP_TOOL_NAME_LENGTH + len(prefix), taken)
            taken.add(name)
            tools.append(
                TarsTool(
                    name=name,
                    flow=tool.flow,
                    owner=tool.owner,
                    owner_user=tool.owner_user,
                    project_tool_name=tool.project_tool_name,
                )
            )
    return tools


def aggregate_tars_tools(projects: list[TarsProject]) -> list[TarsTool]:
    """``projects``' tools flattened into the aggregate server's names."""
    taken: set[str] = set()
    shared = [project for project in projects if project.owner == "system"]
    personal = [project for project in projects if project.owner == "personal"]
    return _aggregate_tools(shared, "", taken) + _aggregate_tools(personal, PERSONAL_PREFIX, taken)


async def collect_tars_tools(db: AsyncSession, tars_id: str | None) -> list[TarsTool]:
    """The shared tools, plus ``tars_id``'s own, flattened into the aggregate server's names."""
    return aggregate_tars_tools(await collect_tars_projects(db, tars_id))
