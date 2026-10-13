import time

import jwt
import pytest
from langflow.api.utils.flow_utils import build_graph_from_data
from langflow.services.tars.actor import (
    TARS_USER_VARIABLE,
    apply_actor,
    bind_actor,
    current_actor,
    decode_actor,
    encode_actor,
)
from langflow.services.tars.ticket import TICKET_SECRET_ENV
from lfx.components.input_output import ChatInput, ChatOutput
from lfx.graph.graph.base import Graph

SECRET = "tars-actor-test-secret-0123456789abcdef"  # noqa: S105 # pragma: allowlist secret
USER_ID = "8f0f6f7e-0000-4000-8000-000000000001"


@pytest.fixture(autouse=True)
def sso_secret(monkeypatch):
    monkeypatch.setenv(TICKET_SECRET_ENV, SECRET)


def cookie(tars_id: str = "tars-1", user_id: str = USER_ID, lifetime: int = 3600, secret: str = SECRET) -> str:
    return encode_actor(tars_id, user_id, lifetime, secret)


def request_variables(graph: Graph) -> dict:
    return dict(graph.context.get("request_variables") or {})


def test_a_cookie_names_the_operator_and_the_session_account():
    actor = decode_actor(cookie())

    assert actor is not None
    assert (actor.tars_id, actor.langflow_user_id) == ("tars-1", USER_ID)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="missing"),
        pytest.param("not-a-jwt", id="garbage"),
        pytest.param(cookie(secret="another-secret-that-is-long-enough-to-sign"), id="forged"),  # noqa: S106
        pytest.param(cookie(lifetime=-60), id="expired"),
        pytest.param(
            jwt.encode(
                {"aud": "langflow", "iss": "langflow", "sub": "tars-1", "uid": USER_ID, "exp": int(time.time()) + 60},
                SECRET,
                algorithm="HS256",
            ),
            id="other-audience",
        ),
    ],
)
def test_anything_else_names_no_one(value):
    assert decode_actor(value) is None


def test_nothing_is_trusted_without_the_shared_secret(monkeypatch):
    value = cookie()
    monkeypatch.delenv(TICKET_SECRET_ENV)

    assert decode_actor(value) is None


def test_the_actor_is_bound_only_for_the_request():
    assert current_actor() is None
    with bind_actor(cookie()):
        assert current_actor().tars_id == "tars-1"
    assert current_actor() is None


def test_a_graph_for_the_bound_account_gets_the_operator():
    graph = Graph()

    with bind_actor(cookie()):
        apply_actor(graph, USER_ID)

    assert request_variables(graph) == {TARS_USER_VARIABLE: "tars-1"}


def test_a_graph_for_another_account_is_left_alone():
    graph = Graph()

    with bind_actor(cookie()):
        apply_actor(graph, "someone-else")

    assert request_variables(graph) == {}


def test_a_user_named_by_the_run_itself_wins():
    graph = Graph()
    graph.context = {"request_variables": {TARS_USER_VARIABLE: "from-librechat", "OTHER": "kept"}}

    with bind_actor(cookie()):
        apply_actor(graph, USER_ID)

    assert request_variables(graph) == {TARS_USER_VARIABLE: "from-librechat", "OTHER": "kept"}


def test_without_a_bound_actor_nothing_changes():
    graph = Graph()

    apply_actor(graph, USER_ID)

    assert request_variables(graph) == {}


def chat_flow_payload() -> dict:
    chat_input = ChatInput()
    chat_output = ChatOutput()
    chat_output.set(input_value=chat_input.message_response)
    return Graph(chat_input, chat_output).dump()["data"]


async def test_building_a_flow_for_the_bound_account_carries_the_operator(client):  # noqa: ARG001
    with bind_actor(cookie(tars_id="tars-operator")):
        graph = await build_graph_from_data(
            "c54f9130-f2fa-4a3e-b22a-3856d946351b", chat_flow_payload(), user_id=USER_ID, flow_name="x"
        )

    assert request_variables(graph)[TARS_USER_VARIABLE] == "tars-operator"


async def test_building_a_flow_for_another_account_does_not(client):  # noqa: ARG001
    with bind_actor(cookie(tars_id="tars-operator")):
        graph = await build_graph_from_data(
            "c54f9130-f2fa-4a3e-b22a-3856d946351b", chat_flow_payload(), user_id="someone-else", flow_name="x"
        )

    assert TARS_USER_VARIABLE not in request_variables(graph)
