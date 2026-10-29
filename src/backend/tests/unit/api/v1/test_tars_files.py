import os
import time
from pathlib import Path
from uuid import uuid4

import anyio
import httpx
import pytest
from httpx import AsyncClient
from langflow.api.v1.tars import tools as tars_tools_api
from langflow.api.v1.tars.tools import _chatting_tars_user, get_tars_mcp_server
from langflow.helpers.flow import get_flow_input_tweaks, json_schema_from_flow
from langflow.services.database.models.flow.model import Flow
from langflow.services.deps import get_settings_service, get_storage_service
from langflow.services.tars import files as tars_files
from langflow.services.tars.files import (
    LIBRECHAT_URL_ENV,
    MAX_BYTES_ENV,
    WORKFLOW_FILES_ARGUMENT,
    LibreChatFileRef,
    parse_refs,
    safe_file_name,
    stage_librechat_files,
)
from lfx.schema.message import Message

from tests.unit.api.v1.test_tars_tools import (  # noqa: F401 - sso_secret is a fixture
    add_flow,
    chat_flow_data,
    default_folder,
    enter_workflow,
    sso_secret,
)

LIBRECHAT = "http://librechat.test"


def ref(file_id: str, filename: str) -> dict:
    return {"id": file_id, "filename": filename, "path": f"/api/tars/files/{file_id}?u=user-1&exp=1&sig=x&s=workflow"}


def files_flow_data() -> dict:
    """A flow that opted in: its Chat Input's Files field is on the node."""
    data = chat_flow_data()
    for node in data["nodes"]:
        if node["data"]["type"] == "ChatInput":
            node["data"]["node"]["template"]["files"]["advanced"] = False
    return data


@pytest.fixture
def storage_dir(client: AsyncClient, tmp_path, monkeypatch):  # noqa: ARG001
    """The storage service writes under ``tmp_path``, never the developer's own Langflow storage."""
    monkeypatch.setattr(get_storage_service(), "data_dir", anyio.Path(tmp_path))
    monkeypatch.setattr(get_settings_service().settings, "config_dir", str(tmp_path))
    return tmp_path


@pytest.fixture
def librechat(storage_dir, monkeypatch):  # noqa: ARG001
    """LibreChat's signed file route, serving whatever the test puts in ``served``."""
    served: dict[str, bytes] = {}
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        file_id = request.url.path.rsplit("/", 1)[-1]
        if file_id not in served:
            return httpx.Response(404)
        return httpx.Response(200, content=served[file_id])

    monkeypatch.setenv(LIBRECHAT_URL_ENV, LIBRECHAT)
    monkeypatch.setattr(tars_files, "_http_client", lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    return served, requested


def staged(flow_id, key: str) -> Path:
    assert key.startswith(f"{flow_id}/")
    return Path(str(get_storage_service().data_dir)) / key


def test_parse_refs_keeps_only_librechat_file_references():
    refs = parse_refs(
        [
            ref("doc-1", "M14D5121632A(2S) (1).pdf"),
            {"id": "doc-2", "filename": "x.pdf", "path": "http://evil.test/api/tars/files/doc-2?s=workflow"},
            {"id": "doc-3", "filename": "x.pdf", "path": "/api/tars/files/doc-1?s=workflow"},
            {"id": "../etc", "filename": "x.pdf", "path": "/api/tars/files/../etc?"},
            "not a ref",
        ]
    )

    assert refs == [LibreChatFileRef(id="doc-1", filename="M14D5121632A(2S) (1).pdf", path=ref("doc-1", "")["path"])]
    assert parse_refs(None) == []


def test_safe_file_name_is_a_single_storage_segment():
    assert safe_file_name("M14D5121632A(2M).pdf", "id") == "M14D5121632A(2M).pdf"
    assert safe_file_name("../../secret_key", "id") == "secret_key"
    assert safe_file_name("a\\b/spec..v2.pdf", "id") == "spec.v2.pdf"
    assert safe_file_name(".librechat-doc-1", "id") == "id"
    assert safe_file_name("", "id") == "id"
    assert safe_file_name(None, "id") == "id"
    long_name = safe_file_name("x" * 300 + ".pdf", "id")
    assert len(long_name) == 200
    assert long_name.endswith(".pdf")


async def test_stages_under_the_original_name_and_never_overwrites(client: AsyncClient, librechat):  # noqa: ARG001
    served, requested = librechat
    flow_id = uuid4()
    served.update({"a": b"first", "b": b"second", "c": b"first"})

    first = await stage_librechat_files(flow_id, parse_refs([ref("a", "spec.txt")]))
    other = await stage_librechat_files(flow_id, parse_refs([ref("b", "spec.txt")]))
    same = await stage_librechat_files(flow_id, parse_refs([ref("c", "spec.txt")]))

    assert first == [f"{flow_id}/spec.txt"]
    assert other == [f"{flow_id}/spec (2).txt"]
    assert same == first
    assert staged(flow_id, first[0]).read_bytes() == b"first"
    assert staged(flow_id, other[0]).read_bytes() == b"second"
    assert requested[0] == f"{LIBRECHAT}{ref('a', '')['path']}"


async def test_a_file_that_cannot_be_fetched_is_left_out(client: AsyncClient, librechat, monkeypatch):  # noqa: ARG001
    served, _ = librechat
    served["big"] = b"x" * 64
    served["ok"] = b"fine"
    monkeypatch.setenv(MAX_BYTES_ENV, "32")
    flow_id = uuid4()

    keys = await stage_librechat_files(
        flow_id, parse_refs([ref("missing", "a.txt"), ref("big", "b.txt"), ref("ok", "c.txt")])
    )

    assert keys == [f"{flow_id}/c.txt"]


@pytest.mark.usefixtures("client")
async def test_nothing_is_fetched_without_the_librechat_origin(librechat, monkeypatch):
    served, requested = librechat
    served["a"] = b"x"
    monkeypatch.delenv(LIBRECHAT_URL_ENV)

    assert await stage_librechat_files(uuid4(), parse_refs([ref("a", "a.txt")])) == []
    assert requested == []


async def test_unused_files_are_purged_after_the_ttl(client: AsyncClient, librechat):  # noqa: ARG001
    served, _ = librechat
    served.update({"old": b"old", "new": b"new"})
    flow_id = uuid4()
    [old_key] = await stage_librechat_files(flow_id, parse_refs([ref("old", "old.txt")]))
    old_file = staged(flow_id, old_key)
    old_marker = old_file.parent / ".librechat-old"
    stale = time.time() - 2 * 60 * 60
    for path in (old_file, old_marker):
        os.utime(path, (stale, stale))

    await stage_librechat_files(flow_id, parse_refs([ref("new", "new.txt")]))

    assert not old_file.exists()
    assert not old_marker.exists()
    assert (old_file.parent / "new.txt").read_bytes() == b"new"


async def test_a_staged_file_reaches_the_model_input(client: AsyncClient, librechat):  # noqa: ARG001
    served, _ = librechat
    served["doc"] = b"63-ball BGA pinout: A1 = VCC"
    flow_id = uuid4()
    [key] = await stage_librechat_files(flow_id, parse_refs([ref("doc", "MX30LF2G28AD.txt")]))

    message = Message(text="compare", files=[key], flow_id=str(flow_id))
    contents = message.get_file_content_dicts()

    assert any("MX30LF2G28AD.txt" in str(part) and "A1 = VCC" in str(part) for part in contents)


def test_only_a_flow_with_files_on_the_node_takes_them():
    opted_in = Flow(name="compare", data=files_flow_data())
    plain = Flow(name="guide", data=chat_flow_data())

    assert "files" in json_schema_from_flow(opted_in)["properties"]
    assert "files" not in json_schema_from_flow(plain)["properties"]
    assert get_flow_input_tweaks(opted_in, {"files": ["k"]})
    assert get_flow_input_tweaks(plain, {"files": ["k"]}) == {}


@pytest.mark.usefixtures("sso_secret")
async def test_mcp_hands_staged_files_only_to_a_flow_that_takes_them(client: AsyncClient, librechat, monkeypatch):
    served, requested = librechat
    served["doc"] = b"datasheet"
    alice = await enter_workflow(client, "tars-files-alice", "files-alice")
    folder = await default_folder(alice)
    compare = await add_flow(alice, folder, "compare", data=files_flow_data())
    await add_flow(alice, folder, "guide")
    seen: dict[str, dict] = {}

    async def fake_handle_call_tool(*, name, arguments, server, project_id, is_action):  # noqa: ARG001
        seen[name] = arguments
        return []

    monkeypatch.setattr(tars_tools_api, "handle_call_tool", fake_handle_call_tool)
    forged = {
        "input_value": "hi",
        "files": [f"{compare.id}/../secret_key"],
        WORKFLOW_FILES_ARGUMENT: [ref("doc", "a.pdf")],
    }
    token = _chatting_tars_user.set("tars-files-alice")
    try:
        listed = {tool.name: tool for tool in await get_tars_mcp_server().list_tools()}
        await get_tars_mcp_server().call_tool("my_compare", forged)
        await get_tars_mcp_server().call_tool("my_guide", forged)
    finally:
        _chatting_tars_user.reset(token)

    assert "files" not in listed["my_compare"].inputSchema["properties"]
    assert seen["compare"] == {"input_value": "hi", "files": [f"{compare.id}/a.pdf"]}
    assert seen["guide"] == {"input_value": "hi"}
    assert len(requested) == 1


@pytest.mark.usefixtures("sso_secret")
async def test_a_real_run_gives_chat_input_the_staged_files(client: AsyncClient, librechat, monkeypatch):
    served, _ = librechat
    served["doc"] = b"63-ball BGA pinout: A1 = VCC"
    alice = await enter_workflow(client, "tars-run-alice", "run-alice")
    await add_flow(alice, await default_folder(alice), "compare", data=files_flow_data())
    created: list[Message] = []
    create = Message.create.__func__

    async def spy(cls, **kwargs):
        message = await create(cls, **kwargs)
        created.append(message)
        return message

    monkeypatch.setattr(Message, "create", classmethod(spy))
    token = _chatting_tars_user.set("tars-run-alice")
    try:
        await get_tars_mcp_server().call_tool(
            "my_compare", {"input_value": "compare", WORKFLOW_FILES_ARGUMENT: [ref("doc", "spec.txt")]}
        )
    finally:
        _chatting_tars_user.reset(token)

    chat_input = next(message for message in created if message.files)
    assert [Path(path).name for path in chat_input.files] == ["spec.txt"]
    assert any("A1 = VCC" in str(part) for part in chat_input.get_file_content_dicts())
