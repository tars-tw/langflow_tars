"""LibreChat's view of Langflow flows: what each pwc_tars user may chat with, and running it.

Both routes authenticate as LibreChat already does for Langflow's MCP: the system account's
API key (``x-api-key``), and they read who is chatting from the header LibreChat already sends
(``x-langflow-global-var-tars_user_id``). The key is refused unless it is the system
account's, since it decides whose personal flows are listed and run.

* ``GET  /api/v1/tars/flows`` — the tools for that user, for LibreChat to keep its agents in step.
* ``/api/v1/tars/mcp/streamable`` — an MCP server (Streamable HTTP) publishing the same tools.
  A call is handed to Langflow's own project tool handler as the flow's owner, so permission
  checks, progress and the caller's request variables behave exactly as on a project server.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Annotated, Any

import anyio
from anyio.abc import TaskStatus
from fastapi import APIRouter, Depends, HTTPException, Request, status
from lfx.base.mcp.util import sanitize_mcp_name
from lfx.log.logger import logger
from mcp import types
from mcp.server import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic import BaseModel

from langflow.api.utils import CurrentActiveMCPUser, DbSessionReadOnly, extract_global_variables_from_headers
from langflow.api.v1.mcp import ResponseNoOp
from langflow.api.v1.mcp_projects import get_project_task_group
from langflow.api.v1.mcp_utils import (
    authenticated_caller_ctx,
    current_request_headers_ctx,
    current_request_variables_ctx,
    current_user_ctx,
    handle_call_tool,
    handle_mcp_errors,
)
from langflow.helpers.flow import json_schema_from_flow
from langflow.services.database.models.user.model import User
from langflow.services.deps import session_scope
from langflow.services.tars.accounts import find_system_account
from langflow.services.tars.actor import TARS_USER_VARIABLE
from langflow.services.tars.tools import TarsTool, collect_tars_tools, flow_tars_model

router = APIRouter(tags=["TARS"], include_in_schema=False)

_chatting_tars_user: ContextVar[str | None] = ContextVar("tars_chatting_user", default=None)


async def require_system_caller(current_user: CurrentActiveMCPUser, db: DbSessionReadOnly) -> User:
    """Only the system account's credential may speak for whichever pwc_tars user is chatting."""
    system = await find_system_account(db)
    if system is None or current_user.id != system.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only the system account may call this")
    return current_user


SystemCaller = Annotated[User, Depends(require_system_caller)]


def chatting_tars_user(request: Request) -> str | None:
    value = extract_global_variables_from_headers(request.headers).get(TARS_USER_VARIABLE, "").strip()
    return value or None


PRODUCT_LABEL = "workflow"
"""What LibreChat's users call a flow; they never see the Langflow brand."""

# ASCII word boundaries, as LibreChat's ``labels.ts`` matches (JS ``\b``): "Langflow的" still counts.
_BRAND_PATTERN = re.compile(r"\blangflow\b", re.IGNORECASE | re.ASCII)


def _description(tool: TarsTool) -> str:
    """A flow's description as LibreChat shows it and hands it to the model.

    Langflow seeds new flows with taglines that name itself ("Design Dialogues with Langflow."),
    so the brand is rewritten rather than passed through.
    """
    flow = tool.flow
    description = flow.action_description or flow.description or f"Tool generated from flow: {tool.name}"
    return _BRAND_PATTERN.sub(PRODUCT_LABEL, description)


class TarsToolRead(BaseModel):
    tool_name: str
    flow_id: str
    name: str
    description: str
    owner: str
    tars_model: str | None
    action_name: str
    """The flow's action name as Langflow's project listing reports it (sanitized, untruncated):
    the key LibreChat named its shared agents by before this list existed."""


@router.get("/flows")
async def list_flows(request: Request, _caller: SystemCaller, db: DbSessionReadOnly) -> list[TarsToolRead]:
    """The tools the chatting user may use: shared flows plus their own, MCP switch on."""
    tools = await collect_tars_tools(db, chatting_tars_user(request))
    return [
        TarsToolRead(
            tool_name=tool.name,
            flow_id=str(tool.flow.id),
            name=tool.flow.name,
            description=_description(tool),
            owner=tool.owner,
            tars_model=flow_tars_model(tool.flow),
            action_name=sanitize_mcp_name(tool.flow.action_name or tool.flow.name),
        )
        for tool in tools
    ]


async def _tools_for_chatting_user() -> list[TarsTool]:
    async with session_scope() as db:
        return await collect_tars_tools(db, _chatting_tars_user.get())


class TarsMCPServer:
    """One stateless Streamable HTTP MCP server for every user; each request says who is chatting."""

    def __init__(self) -> None:
        self.server = Server("langflow-tars")
        self.session_manager = StreamableHTTPSessionManager(self.server, stateless=True)
        self._manager_lock = anyio.Lock()
        self._manager_started = False

        @self.server.list_tools()
        @handle_mcp_errors
        async def list_tools(_request: types.ListToolsRequest) -> types.ListToolsResult:
            result = types.ListToolsResult(tools=await self.list_tools())
            # As on the project server: the SDK only clears its cache on the list[Tool] branch.
            self.server._tool_cache.clear()  # noqa: SLF001
            return result

        @self.server.call_tool()
        @handle_mcp_errors
        async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
            return await self.call_tool(name, arguments)

    async def list_tools(self) -> list[types.Tool]:
        """The chatting user's tools, as MCP tool definitions."""
        return [
            types.Tool(
                name=tool.name,
                # The name is sanitized to ASCII, which empties a CJK flow name ("製作48腳位比對表格"
                # -> "_48"); the title keeps what the flow is called, for clients to show instead.
                title=tool.flow.name,
                description=_description(tool),
                inputSchema=json_schema_from_flow(tool.flow),
            )
            for tool in await _tools_for_chatting_user()
        ]

    async def call_tool(self, name: str, arguments: dict) -> list[types.TextContent]:
        """Run the named tool's flow as its owner, through Langflow's own project tool handler."""
        tool = next((tool for tool in await _tools_for_chatting_user() if tool.name == name), None)
        if tool is None:
            msg = (
                f"Tool '{name}' not found: its flow was renamed, unpublished or deleted in Langflow. "
                "Re-select the flow's tool in the agent builder."
            )
            raise ValueError(msg)
        # The flow runs as its owner, who is also the caller of record: LibreChat's system
        # credential vouches for the chatting user, so this is not an anonymous public call.
        user_token = current_user_ctx.set(tool.owner_user)
        caller_token = authenticated_caller_ctx.set(tool.owner_user.id)
        try:
            return await handle_call_tool(
                name=tool.project_tool_name,
                arguments=arguments,
                server=self.server,
                project_id=tool.flow.folder_id,
                is_action=True,
            )
        finally:
            authenticated_caller_ctx.reset(caller_token)
            current_user_ctx.reset(user_token)

    async def ensure_session_manager_running(self) -> None:
        if self._manager_started:
            return
        async with self._manager_lock:
            if self._manager_started:
                return
            await get_project_task_group().start_task(self._run_session_manager)

    async def _run_session_manager(self, *, task_status: TaskStatus[None] = anyio.TASK_STATUS_IGNORED) -> None:
        try:
            async with self.session_manager.run():
                self._manager_started = True
                task_status.started()
                await anyio.sleep_forever()
        finally:
            self._manager_started = False


_mcp_server: TarsMCPServer | None = None


def get_tars_mcp_server() -> TarsMCPServer:
    global _mcp_server  # noqa: PLW0603
    if _mcp_server is None:
        _mcp_server = TarsMCPServer()
    return _mcp_server


@router.head("/mcp/streamable")
async def mcp_health() -> ResponseNoOp:
    return ResponseNoOp(status_code=status.HTTP_200_OK)


@router.api_route("/mcp/streamable", methods=["GET", "POST", "DELETE"], response_class=ResponseNoOp)
@router.api_route("/mcp/streamable/", methods=["GET", "POST", "DELETE"], response_class=ResponseNoOp)
async def handle_mcp(request: Request, _caller: SystemCaller) -> Any:
    server = get_tars_mcp_server()
    await server.ensure_session_manager_running()

    chatting_token = _chatting_tars_user.set(chatting_tars_user(request))
    variables = extract_global_variables_from_headers(request.headers, include_auth_headers=True)
    variables_token = current_request_variables_ctx.set(variables or None)
    headers_token = current_request_headers_ctx.set(request.headers)
    try:
        await server.session_manager.handle_request(request.scope, request.receive, request._send)  # noqa: SLF001
    except HTTPException:
        raise
    except Exception as exc:
        await logger.aexception(f"Error handling the TARS MCP request: {exc!s}")
        raise HTTPException(status_code=500, detail="Internal server error in the TARS MCP transport") from exc
    finally:
        current_request_headers_ctx.reset(headers_token)
        current_request_variables_ctx.reset(variables_token)
        _chatting_tars_user.reset(chatting_token)
    return ResponseNoOp(status_code=status.HTTP_200_OK)
