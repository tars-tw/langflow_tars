from uuid import UUID

import pytest
from httpx import AsyncClient
from langflow.api.v1.mcp_utils import authenticated_caller_ctx, current_user_ctx
from langflow.api.v1.tars import tools as tars_tools_api
from langflow.api.v1.tars.tools import _chatting_tars_user, get_tars_mcp_server
from langflow.initial_setup.setup import get_or_create_default_folder
from langflow.services.database.models.api_key.crud import create_api_key
from langflow.services.database.models.api_key.model import ApiKeyCreate
from langflow.services.database.models.flow.model import Flow
from langflow.services.database.models.folder.model import Folder
from langflow.services.database.models.user.crud import get_user_by_username
from langflow.services.deps import session_scope
from langflow.services.tars.accounts import find_tars_account
from langflow.services.tars.ticket import TICKET_SECRET_ENV
from langflow.services.tars.tools import collect_tars_tools, flow_tars_model
from lfx.components.input_output import ChatInput, ChatOutput
from lfx.graph.graph.base import Graph

from tests.unit.api.v1.test_tars_sso import SECRET, make_ticket

FLOWS_URL = "api/v1/tars/flows"
TARS_HEADER = "x-langflow-global-var-tars_user_id"


@pytest.fixture
def sso_secret(monkeypatch):
    monkeypatch.setenv(TICKET_SECRET_ENV, SECRET)


def chat_flow_data() -> dict:
    """A real, buildable flow: MCP derives each tool's input schema from it."""
    chat_input = ChatInput()
    chat_output = ChatOutput()
    chat_output.set(input_value=chat_input.message_response)
    return Graph(chat_input, chat_output).dump()["data"]


def tars_agent_flow_data(model: str) -> dict:
    return {
        "nodes": [
            {"data": {"type": "ChatInput", "node": {"template": {}}}},
            {"data": {"type": "TarsAgent", "node": {"template": {"model_name": {"value": model}}}}},
        ],
        "edges": [],
    }


async def enter_workflow(client: AsyncClient, tars_id: str, username: str) -> UUID:
    client.cookies.clear()
    response = await client.get(
        "api/v1/tars/sso", params={"ticket": make_ticket(sub=tars_id, preferred_username=username)}
    )
    assert response.status_code == 302
    client.cookies.clear()
    async with session_scope() as db:
        return (await find_tars_account(db, tars_id)).id


async def add_flow(
    owner_id: UUID, folder_id: UUID, name: str, *, mcp: bool = True, data: dict | None = None, action: str | None = None
) -> Flow:
    async with session_scope() as db:
        flow = Flow(
            name=name,
            action_name=action,
            user_id=owner_id,
            folder_id=folder_id,
            mcp_enabled=mcp,
            data=data or chat_flow_data(),
        )
        db.add(flow)
        await db.flush()
        await db.refresh(flow)
        return flow


async def default_folder(owner_id: UUID) -> UUID:
    async with session_scope() as db:
        return (await get_or_create_default_folder(db, owner_id)).id


async def add_folder(owner_id: UUID, name: str) -> UUID:
    async with session_scope() as db:
        folder = Folder(name=name, user_id=owner_id)
        db.add(folder)
        await db.flush()
        return folder.id


async def system_user():
    async with session_scope() as db:
        return await get_user_by_username(db, "langflow")


async def api_key_for(user_id: UUID) -> str:
    async with session_scope() as db:
        return (await create_api_key(db, ApiKeyCreate(name="tars-test"), user_id=user_id)).api_key


@pytest.fixture
async def workspace(client: AsyncClient, sso_secret):  # noqa: ARG001
    """A system account with shared flows in two projects, and two pwc_tars users with their own."""
    system = await system_user()
    shared_project = await default_folder(system.id)
    second_project = await add_folder(system.id, "Second")
    await add_flow(system.id, shared_project, "guide")
    await add_flow(system.id, shared_project, "draft", mcp=False)
    await add_flow(system.id, second_project, "guide copy", action="guide")

    alice = await enter_workflow(client, "tars-alice", "alice")
    await add_flow(alice, await default_folder(alice), "guide")
    await add_flow(alice, await default_folder(alice), "private", mcp=False)

    bob = await enter_workflow(client, "tars-bob", "bob")
    await add_flow(bob, await default_folder(bob), "bobs")
    return {"system": system, "alice": alice, "bob": bob}


async def tools_for(tars_id: str | None):
    async with session_scope() as db:
        return await collect_tars_tools(db, tars_id)


async def test_everyone_gets_the_shared_tools_and_only_their_own(workspace):
    tools = await tools_for("tars-alice")

    assert [(tool.name, tool.owner) for tool in tools] == [
        ("guide", "system"),
        ("guide_1", "system"),
        ("my_guide", "personal"),
    ]
    assert {tool.owner_user.id for tool in tools if tool.owner == "personal"} == {workspace["alice"]}
    assert all(tool.project_tool_name == "guide" for tool in tools)


async def test_without_a_chatting_user_only_the_shared_tools_are_offered(workspace):  # noqa: ARG001
    assert [tool.name for tool in await tools_for(None)] == ["guide", "guide_1"]
    assert [tool.name for tool in await tools_for("tars-nobody")] == ["guide", "guide_1"]


async def test_a_deactivated_user_keeps_only_the_shared_tools(workspace):
    async with session_scope() as db:
        from langflow.services.database.models.user.model import User

        user = await db.get(User, workspace["alice"])
        user.is_active = False
        db.add(user)

    assert [tool.name for tool in await tools_for("tars-alice")] == ["guide", "guide_1"]


def test_reads_the_model_a_tars_node_names():
    assert flow_tars_model(Flow(name="x", data=tars_agent_flow_data("gpt-5.4-mini"))) == "gpt-5.4-mini"
    assert flow_tars_model(Flow(name="x", data=tars_agent_flow_data("  "))) is None
    assert flow_tars_model(Flow(name="x", data=chat_flow_data())) is None


async def test_flows_list_needs_the_system_accounts_credential(client: AsyncClient, workspace):
    assert (await client.get(FLOWS_URL)).status_code in (401, 403)

    alice_key = await api_key_for(workspace["alice"])
    refused = await client.get(FLOWS_URL, headers={"x-api-key": alice_key, TARS_HEADER: "tars-bob"})
    assert refused.status_code == 403


async def test_flows_list_for_the_chatting_user(client: AsyncClient, workspace):
    system_key = await api_key_for(workspace["system"].id)

    response = await client.get(FLOWS_URL, headers={"x-api-key": system_key, TARS_HEADER: "tars-bob"})

    assert response.status_code == 200
    listed = response.json()
    # Which of the two shared projects claims the bare name follows their ids; both stay listed.
    assert [(row["tool_name"], row["owner"]) for row in listed] == [
        ("guide", "system"),
        ("guide_1", "system"),
        ("my_bobs", "personal"),
    ]
    assert {row["name"] for row in listed} == {"guide", "guide copy", "bobs"}
    assert all(row["tars_model"] is None for row in listed)
    assert {row["action_name"] for row in listed} == {"guide", "bobs"}


async def test_mcp_lists_the_chatting_users_tools(workspace):  # noqa: ARG001
    token = _chatting_tars_user.set("tars-alice")
    try:
        names = [tool.name for tool in await get_tars_mcp_server().list_tools()]
    finally:
        _chatting_tars_user.reset(token)

    assert names == ["guide", "guide_1", "my_guide"]


async def test_mcp_runs_a_personal_tool_as_its_owner(workspace, monkeypatch):
    seen = {}

    async def fake_handle_call_tool(*, name, arguments, server, project_id, is_action):  # noqa: ARG001
        seen.update(
            name=name,
            arguments=arguments,
            project_id=project_id,
            is_action=is_action,
            user=current_user_ctx.get().id,
            caller=authenticated_caller_ctx.get(),
        )
        return []

    monkeypatch.setattr(tars_tools_api, "handle_call_tool", fake_handle_call_tool)
    token = _chatting_tars_user.set("tars-alice")
    try:
        await get_tars_mcp_server().call_tool("my_guide", {"input_value": "hi"})
    finally:
        _chatting_tars_user.reset(token)

    assert seen["name"] == "guide"
    assert seen["is_action"] is True
    assert seen["user"] == workspace["alice"]
    assert seen["caller"] == workspace["alice"]
    assert seen["project_id"] == await default_folder(workspace["alice"])


async def test_mcp_refuses_someone_elses_personal_tool(workspace):  # noqa: ARG001
    token = _chatting_tars_user.set("tars-alice")
    try:
        with pytest.raises(ValueError, match="not found"):
            await get_tars_mcp_server().call_tool("my_bobs", {})
    finally:
        _chatting_tars_user.reset(token)
