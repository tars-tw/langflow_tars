"""LibreChat's view of Langflow flows: what each pwc_tars user may chat with, and running it.

Both routes authenticate as LibreChat already does for Langflow's MCP: the system account's
API key (``x-api-key``), and they read who is chatting from the header LibreChat already sends
(``x-langflow-global-var-tars_user_id``). The key is refused unless it is the system
account's, since it decides whose personal flows are listed and run.

* ``GET  /api/v1/tars/flows`` — the tools for that user, each with its project, for LibreChat to
  keep its per-project agents in step.
* ``GET  /api/v1/tars/projects`` — every project holding a published flow, shared and personal,
  for LibreChat to register one MCP server per project (it hides personal ones from other users).
* ``/api/v1/tars/mcp/project/{project_id}/streamable`` — one project's tools as an MCP server
  (Streamable HTTP), named as Langflow's own project server names them. A personal project
  answers only its owner; anyone else gets 404.
* ``/api/v1/tars/mcp/streamable`` — every project's tools flattened into one server (deprecated:
  what LibreChat used before it split Workflow by project; kept for configs still pinned to it).

A call is handed to Langflow's own project tool handler as the flow's owner, so permission
checks, progress and the caller's request variables behave exactly as on a project server.
"""

from __future__ import annotations

import re
from contextvars import ContextVar
from typing import Annotated, Any
from uuid import UUID

import anyio
from anyio.abc import TaskStatus
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
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
from langflow.services.tars.files import (
    FILES_FIELD,
    WORKFLOW_FILES_ARGUMENT,
    accepts_files,
    parse_refs,
    stage_librechat_files,
    without_files_field,
)
from langflow.services.tars.tools import (
    TarsTool,
    aggregate_tars_tools,
    collect_tars_projects,
    collect_tars_tools,
    find_tars_project,
    flow_tars_model,
    list_all_tars_projects,
)

router = APIRouter(tags=["TARS"], include_in_schema=False)

_chatting_tars_user: ContextVar[str | None] = ContextVar("tars_chatting_user", default=None)
_chatting_project: ContextVar[UUID | None] = ContextVar("tars_chatting_project", default=None)


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


def _unbranded(text: str) -> str:
    return _BRAND_PATTERN.sub(PRODUCT_LABEL, text)


def _description(tool: TarsTool) -> str:
    """A flow's description as LibreChat shows it and hands it to the model.

    Langflow seeds new flows with taglines that name itself ("Design Dialogues with Langflow."),
    so the brand is rewritten rather than passed through.
    """
    flow = tool.flow
    return _unbranded(flow.action_description or flow.description or f"Tool generated from flow: {tool.name}")


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
    project_id: str
    project_name: str
    project_description: str
    project_tool_name: str
    """The tool's name on its project's server (``/mcp/project/{project_id}/streamable``)."""


class TarsProjectRead(BaseModel):
    project_id: str
    name: str
    description: str
    owner: str
    tars_id: str | None
    """The pwc_tars user a personal project belongs to; None for a shared one."""


@router.get("/flows")
async def list_flows(request: Request, _caller: SystemCaller, db: DbSessionReadOnly) -> list[TarsToolRead]:
    """The tools the chatting user may use: shared flows plus their own, MCP switch on."""
    projects = await collect_tars_projects(db, chatting_tars_user(request))
    project_of = {tool.flow.id: project for project in projects for tool in project.tools}
    return [
        TarsToolRead(
            tool_name=tool.name,
            flow_id=str(tool.flow.id),
            name=tool.flow.name,
            description=_description(tool),
            owner=tool.owner,
            tars_model=flow_tars_model(tool.flow),
            action_name=sanitize_mcp_name(tool.flow.action_name or tool.flow.name),
            project_id=str(project_of[tool.flow.id].id),
            project_name=project_of[tool.flow.id].name,
            project_description=_unbranded(project_of[tool.flow.id].description or ""),
            project_tool_name=tool.project_tool_name,
        )
        for tool in aggregate_tars_tools(projects)
    ]


@router.get("/projects")
async def list_projects(_caller: SystemCaller, db: DbSessionReadOnly) -> list[TarsProjectRead]:
    """Every project holding a published flow, shared and personal alike."""
    return [
        TarsProjectRead(
            project_id=str(project.id),
            name=project.name,
            description=_unbranded(project.description or ""),
            owner=project.owner,
            tars_id=project.tars_id,
        )
        for project in await list_all_tars_projects(db)
    ]


async def _tools_for_chatting_user() -> list[TarsTool]:
    """The aggregate server's tools, or one project's when the request came in on its server."""
    async with session_scope() as db:
        project_id = _chatting_project.get()
        if project_id is None:
            return await collect_tars_tools(db, _chatting_tars_user.get())
        project = await find_tars_project(db, _chatting_tars_user.get(), project_id)
        return project.tools if project else []


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
        tools: list[types.Tool] = []
        for tool in await _tools_for_chatting_user():
            # One flow whose schema cannot be derived must not take every other tool down with it.
            try:
                input_schema = without_files_field(json_schema_from_flow(tool.flow))
            except Exception as e:  # noqa: BLE001
                await logger.aerror(f"Flow excluded from TARS tool list -- {tool.name} ({tool.flow.id}): {e!s}")
                continue
            tools.append(
                types.Tool(
                    name=tool.name,
                    # The name is sanitized to ASCII, which empties a CJK flow name ("製作48腳位比對表格"
                    # -> "_48"); the title keeps what the flow is called, for clients to show instead.
                    title=tool.flow.name,
                    description=_description(tool),
                    inputSchema=input_schema,
                )
            )
        return tools

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
        arguments = await _with_librechat_files(tool, arguments)
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


async def _with_librechat_files(tool: TarsTool, arguments: dict) -> dict:
    """The call's arguments with LibreChat's file references swapped for the staged files.

    ``files`` is never taken from the caller: it names server paths, so only the staged LibreChat
    attachments go there, and only for a flow that opted in by putting Chat Input's Files on the node.
    """
    arguments = dict(arguments or {})
    refs = parse_refs(arguments.pop(WORKFLOW_FILES_ARGUMENT, None))
    arguments.pop(FILES_FIELD, None)
    if refs and accepts_files(json_schema_from_flow(tool.flow)):
        keys = await stage_librechat_files(tool.flow.id, refs)
        if keys:
            arguments[FILES_FIELD] = keys
    return arguments


def get_tars_mcp_server() -> TarsMCPServer:
    global _mcp_server  # noqa: PLW0603
    if _mcp_server is None:
        _mcp_server = TarsMCPServer()
    return _mcp_server


@router.head("/mcp/streamable")
async def mcp_health() -> ResponseNoOp:
    return ResponseNoOp(status_code=status.HTTP_200_OK)


@router.get("/mcp/streamable")
@router.get("/mcp/streamable/")
async def refuse_mcp_stream() -> Response:
    """No standalone SSE stream: the server is stateless, so nothing could ever be pushed on one.

    Clients treat 405 as "no stream offered". Holding one open instead leaves a GET for every
    connection that the client aborts when it closes -- LibreChat right after listing the tools --
    and an aborted GET fails in the HTTP middleware chain with "No response returned". Answered
    before the caller is authenticated, so it never waits on the database.
    """
    return Response(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, headers={"Allow": "POST, DELETE"})


@router.api_route("/mcp/streamable", methods=["POST", "DELETE"], response_class=ResponseNoOp)
@router.api_route("/mcp/streamable/", methods=["POST", "DELETE"], response_class=ResponseNoOp)
async def handle_mcp(request: Request, _caller: SystemCaller) -> Any:
    return await _serve_mcp(request, None)


@router.head("/mcp/project/{project_id}/streamable")
async def project_mcp_health(project_id: UUID) -> ResponseNoOp:  # noqa: ARG001
    return ResponseNoOp(status_code=status.HTTP_200_OK)


@router.get("/mcp/project/{project_id}/streamable")
@router.get("/mcp/project/{project_id}/streamable/")
async def refuse_project_mcp_stream(project_id: UUID) -> Response:  # noqa: ARG001
    """As on the aggregate server: stateless, so no standalone stream is offered."""
    return Response(status_code=status.HTTP_405_METHOD_NOT_ALLOWED, headers={"Allow": "POST, DELETE"})


@router.api_route("/mcp/project/{project_id}/streamable", methods=["POST", "DELETE"], response_class=ResponseNoOp)
@router.api_route("/mcp/project/{project_id}/streamable/", methods=["POST", "DELETE"], response_class=ResponseNoOp)
async def handle_project_mcp(project_id: UUID, request: Request, _caller: SystemCaller, db: DbSessionReadOnly) -> Any:
    """One project's tools, for whoever may use it.

    Anyone may use a shared project, only its owner a personal one. Every other caller, and a
    project with nothing published, gets 404.
    """
    if await find_tars_project(db, chatting_tars_user(request), project_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
    return await _serve_mcp(request, project_id)


async def _serve_mcp(request: Request, project_id: UUID | None) -> Any:
    server = get_tars_mcp_server()
    await server.ensure_session_manager_running()

    project_token = _chatting_project.set(project_id)
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
        _chatting_project.reset(project_token)
    return ResponseNoOp(status_code=status.HTTP_200_OK)
