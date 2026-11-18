"""TARS as the unified model provider: pickers list what pwc_tars allows, runs act for the chatter."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import httpx
import pytest
import respx
from fastapi import status
from langflow.schema.message import Message
from langflow.services.database.models.auth import SSOUserProfile
from langflow.services.deps import session_scope
from langflow.services.tars import models as tars_models
from langflow.services.tars.actor import bind_actor, encode_actor
from langflow.services.tars.models import (
    BASE_URL_ENV,
    LANGUAGE_MODELS_ONLY_ENV,
    PROVIDER,
    SERVICE_KEY_ENV,
    TarsChatModel,
    chatting_tars_id,
    fetch_live_tars_models,
    register_tars_model_provider,
)
from langflow.services.tars.ticket import TARS_PROVIDER, TICKET_SECRET_ENV
from lfx.base.models.provider_registry import is_registered, resolve_provider_id, unregister_provider
from lfx.base.models.unified_models import get_llm
from lfx.components.input_output import ChatInput, ChatOutput
from lfx.graph.graph.base import Graph
from lfx.memory.flow_context import (
    reset_current_message_executor_id,
    reset_current_request_variables,
    set_current_message_executor_id,
    set_current_request_variables,
)
from lfx.services.model_provider_policy import (
    ModelProviderPolicyContext,
    ModelProviderPolicyError,
    ModelProviderPolicyPurpose,
    ModelProviderPolicySnapshot,
    language_model_provider_lock,
    lock_language_models_to,
    resolve_model_provider_policy,
)

if TYPE_CHECKING:
    from httpx import AsyncClient

TARS_URL = "http://tars.test:5000"
MODELS_URL = f"{TARS_URL}/api/langflow-service/models"
SERVICE_KEY = "svc-key"  # pragma: allowlist secret
SECRET = "tars-models-test-secret-0123456789abcdef"  # noqa: S105 # pragma: allowlist secret
TARS_MODELS = [
    {"name": "gpt-5.4-mini", "display_name": "GPT mini", "tool_calling": True},
    {"name": "gemma3", "display_name": "gemma3", "tool_calling": True},
]


@pytest.fixture
def tars_env(monkeypatch):
    monkeypatch.setenv(SERVICE_KEY_ENV, SERVICE_KEY)
    monkeypatch.setenv(BASE_URL_ENV, f"{TARS_URL}/")
    monkeypatch.setenv(TICKET_SECRET_ENV, SECRET)
    tars_models.clear_model_list_cache()
    yield
    tars_models.clear_model_list_cache()
    # The test app's create_app registers TARS (and locks language models) while these are set.
    unregister_provider(PROVIDER)
    lock_language_models_to(None)


@pytest.fixture
def tars_provider(tars_env):  # noqa: ARG001
    unregister_provider(PROVIDER)
    assert register_tars_model_provider()


@pytest.fixture
def pwc_tars():
    with respx.mock(assert_all_called=False) as router:
        route = router.get(MODELS_URL).mock(
            return_value=httpx.Response(200, json={"success": True, "data": {"models": TARS_MODELS}})
        )
        yield route


@pytest.fixture
async def linked_user(active_user):
    """``active_user`` signed in through TARS as pwc_tars user ``amy``."""
    async with session_scope() as db:
        db.add(SSOUserProfile(user_id=active_user.id, sso_provider=TARS_PROVIDER, sso_user_id="amy"))
    return active_user


def _tars_only(user_id) -> ModelProviderPolicySnapshot:
    candidates = frozenset({resolve_provider_id(PROVIDER), "openai", "anthropic"})
    return ModelProviderPolicySnapshot(
        context=ModelProviderPolicyContext(user_id=user_id),
        purpose=ModelProviderPolicyPurpose.USE,
        candidate_provider_ids=candidates,
        allowed_provider_ids=frozenset({"tars"}),
    )


def test_not_registered_without_a_service_key(monkeypatch):
    monkeypatch.delenv(SERVICE_KEY_ENV, raising=False)
    registered = []

    assert register_tars_model_provider(registered.append) is False
    assert registered == []


@pytest.mark.usefixtures("tars_provider")
def test_registered_as_a_keyless_live_provider():
    assert is_registered(PROVIDER)
    assert resolve_provider_id(PROVIDER) == "tars"


@pytest.mark.usefixtures("tars_env")
async def test_discovery_lists_what_pwc_tars_allows_the_linked_user(linked_user, pwc_tars):
    rows = fetch_live_tars_models(linked_user.id, "llm")

    assert [row["name"] for row in rows] == ["gpt-5.4-mini", "gemma3"]
    assert all(row["provider"] == PROVIDER and row["default"] and row["tool_calling"] for row in rows)
    request = pwc_tars.calls.last.request
    assert request.headers["X-TARS-User-Id"] == "amy"
    assert request.headers["X-TARS-Service-Key"] == SERVICE_KEY


@pytest.mark.usefixtures("tars_env")
async def test_discovery_is_cached_per_person(linked_user, pwc_tars):
    fetch_live_tars_models(linked_user.id, "llm")
    fetch_live_tars_models(linked_user.id, "llm")

    assert pwc_tars.call_count == 1


@pytest.mark.usefixtures("tars_env")
async def test_discovery_offers_nothing_for_embeddings_or_unlinked_accounts(active_user, pwc_tars):
    assert fetch_live_tars_models(active_user.id, "embeddings") == []
    assert fetch_live_tars_models(active_user.id, "llm") == []
    assert fetch_live_tars_models(None, "llm") == []
    assert pwc_tars.call_count == 0


@pytest.mark.usefixtures("tars_env")
async def test_discovery_follows_the_operator_an_actor_cookie_names(active_user, pwc_tars):
    with bind_actor(encode_actor("bob", str(active_user.id), 3600, SECRET)):
        fetch_live_tars_models(active_user.id, "llm")

    assert pwc_tars.calls.last.request.headers["X-TARS-User-Id"] == "bob"


@pytest.mark.usefixtures("tars_env")
async def test_discovery_degrades_to_no_models_when_pwc_tars_fails(linked_user):
    with respx.mock() as router:
        router.get(MODELS_URL).mock(return_value=httpx.Response(503))
        assert fetch_live_tars_models(linked_user.id, "llm") == []


@pytest.mark.usefixtures("tars_env")
async def test_a_run_acts_for_the_chatter_over_the_executing_account(linked_user):
    executor = set_current_message_executor_id(linked_user.id)
    try:
        # The executing account is linked to amy...
        assert chatting_tars_id() == "amy"
        # ...but the run carries the person chatting (LibreChat's MCP header), who wins.
        variables = set_current_request_variables({"TARS_USER_ID": "carol"})
        try:
            assert chatting_tars_id() == "carol"
        finally:
            reset_current_request_variables(variables)
    finally:
        reset_current_message_executor_id(executor)


@pytest.mark.usefixtures("tars_provider")
def test_get_llm_builds_a_client_for_pwc_tars():
    variables = set_current_request_variables({"TARS_USER_ID": "carol"})
    try:
        llm = get_llm(
            [{"name": "gemma3", "provider": PROVIDER, "metadata": {}}],
            user_id=None,
            temperature=0.2,
            stream=True,
            provider_policy=_tars_only(None),
        )
    finally:
        reset_current_request_variables(variables)

    assert isinstance(llm, TarsChatModel)
    assert llm.model_name == "gemma3"
    assert llm.openai_api_base == f"{TARS_URL}/api/langflow-service/v1"
    assert llm.default_headers == {"X-TARS-Service-Key": SERVICE_KEY, "X-TARS-User-Id": "carol"}
    assert llm.streaming is True
    assert llm.stream_usage is True


@pytest.mark.usefixtures("tars_provider")
def test_a_tars_only_policy_refuses_other_providers():
    with pytest.raises(ModelProviderPolicyError):
        get_llm(
            [{"name": "gpt-4o", "provider": "OpenAI", "metadata": {}}],
            user_id=None,
            api_key="sk-test",  # pragma: allowlist secret
            provider_policy=_tars_only(None),
        )


def test_the_client_needs_the_service_key(monkeypatch):
    monkeypatch.delenv(SERVICE_KEY_ENV, raising=False)

    with pytest.raises(ValueError, match=SERVICE_KEY_ENV):
        TarsChatModel(model="gemma3", api_key="EMPTY")  # pragma: allowlist secret


@pytest.mark.usefixtures("tars_provider", "pwc_tars")
async def test_language_model_options_list_tars_models(client: AsyncClient, linked_user, logged_in_headers):
    assert linked_user is not None
    response = await client.get("api/v1/model_options/language", headers=logged_in_headers)

    assert response.status_code == status.HTTP_200_OK
    tars_options = [option["name"] for option in response.json() if option["provider"] == PROVIDER]
    assert tars_options == ["gpt-5.4-mini", "gemma3"]


@pytest.mark.usefixtures("tars_provider", "pwc_tars")
async def test_an_unlinked_account_sees_no_tars_models(client: AsyncClient, logged_in_headers):
    response = await client.get("api/v1/model_options/language", headers=logged_in_headers)

    assert response.status_code == status.HTTP_200_OK
    assert [option for option in response.json() if option["provider"] == PROVIDER] == []


class _ChatterProbe(ChatInput):
    """Reports who a model client built inside this component would act for."""

    name = "ChatterProbe"

    async def message_response(self) -> Message:
        return Message(text=chatting_tars_id() or "")


async def test_a_graph_run_exposes_its_chatter_to_components(client):  # noqa: ARG001
    probe = _ChatterProbe(_id="probe")
    chat_output = ChatOutput(_id="chat_output")
    chat_output.set(input_value=probe.message_response)
    graph = Graph(probe, chat_output, flow_id=str(uuid4()), user_id=str(uuid4()))
    graph.context = {"request_variables": {"TARS_USER_ID": "carol"}}

    async for _ in graph.async_start():
        pass

    assert probe.get_output_by_method(probe.message_response).value.text == "carol"
    # Outside the run nothing is bound.
    assert chatting_tars_id() is None


# --- Language models come from TARS alone; embeddings keep their providers ---


@pytest.mark.usefixtures("tars_provider")
def test_registration_locks_language_models_to_tars():
    assert language_model_provider_lock() == frozenset({"tars"})


def test_the_lock_can_be_lifted(tars_env, monkeypatch):  # noqa: ARG001
    monkeypatch.setenv(LANGUAGE_MODELS_ONLY_ENV, "false")
    unregister_provider(PROVIDER)

    assert register_tars_model_provider()
    assert language_model_provider_lock() is None


@pytest.mark.usefixtures("client", "tars_provider")
def test_runs_refuse_language_models_of_other_providers():
    with pytest.raises(ModelProviderPolicyError):
        get_llm(
            [{"name": "gpt-4o", "provider": "OpenAI", "metadata": {}}],
            user_id=None,
            api_key="sk-test",  # pragma: allowlist secret
        )
    snapshot = resolve_model_provider_policy(
        user_id=None,
        providers=["OpenAI", "Anthropic", PROVIDER],
        purpose=ModelProviderPolicyPurpose.USE,
    )
    assert snapshot.allows_model("OpenAI", "text-embedding-3-small", model_type="embeddings")
    assert not snapshot.allows("Anthropic")


@pytest.mark.usefixtures("tars_provider", "pwc_tars")
async def test_model_catalog_keeps_only_embeddings_of_other_providers(
    client: AsyncClient, linked_user, logged_in_headers
):
    assert linked_user is not None
    catalog = await client.get("api/v1/models", headers=logged_in_headers)
    providers = await client.get("api/v1/models/providers", headers=logged_in_headers)
    enabled = await client.get("api/v1/models/enabled_models", headers=logged_in_headers)

    assert catalog.status_code == status.HTTP_200_OK
    groups = {group["provider"]: group for group in catalog.json()}
    assert "Anthropic" not in groups
    assert {model["model_name"] for model in groups[PROVIDER]["models"]} == {"gpt-5.4-mini", "gemma3"}
    openai_types = {(model.get("metadata") or {}).get("model_type") or "llm" for model in groups["OpenAI"]["models"]}
    assert openai_types == {"embeddings"}

    assert providers.status_code == status.HTTP_200_OK
    assert "Anthropic" not in providers.json()
    assert {"OpenAI", PROVIDER} <= set(providers.json())

    assert enabled.status_code == status.HTTP_200_OK
    openai_enabled = enabled.json()["enabled_models_by_type"].get("OpenAI", {})
    assert "llm" not in openai_enabled
