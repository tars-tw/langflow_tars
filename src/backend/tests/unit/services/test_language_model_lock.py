"""Locking language models to some providers while embedding providers stay as they are."""

from __future__ import annotations

import pytest
from langflow.api.v1.endpoints import _filter_component_palette_by_provider_policy, _palette_model_types
from lfx.base.models.provider_registry import resolve_provider_id
from lfx.components.models_and_agents.agent import AgentComponent
from lfx.services.model_provider_policy import (
    ModelProviderPolicyContext,
    ModelProviderPolicyError,
    ModelProviderPolicyPurpose,
    ModelProviderPolicyService,
    ModelProviderPolicySnapshot,
    language_model_provider_lock,
    lock_language_models_to,
)

OPENAI_EMBEDDING = "text-embedding-3-small"


@pytest.fixture
def locked_to_ollama():
    """Ollama stands in for the one provider language models may come from."""
    lock_language_models_to(["Ollama"])
    yield
    lock_language_models_to(None)


def _snapshot(language_model_provider_ids=None) -> ModelProviderPolicySnapshot:
    candidates = frozenset({"openai", "anthropic", "ollama"})
    return ModelProviderPolicySnapshot(
        context=ModelProviderPolicyContext(user_id=None),
        purpose=ModelProviderPolicyPurpose.USE,
        candidate_provider_ids=candidates,
        allowed_provider_ids=candidates,
        language_model_provider_ids=language_model_provider_ids,
    )


def _resolve(providers=("OpenAI", "Anthropic", "Ollama")) -> ModelProviderPolicySnapshot:
    return ModelProviderPolicyService().resolve(
        context=ModelProviderPolicyContext(user_id=None),
        candidate_provider_ids=frozenset(resolve_provider_id(p) for p in providers),
        purpose=ModelProviderPolicyPurpose.USE,
    )


def test_unlocked_snapshots_behave_as_before():
    snapshot = _snapshot()

    assert snapshot.allows_model("OpenAI", "gpt-4o", model_type="llm")
    assert snapshot.allows_model_type("Anthropic", "llm")


def test_a_lock_blocks_language_models_of_other_providers_only():
    snapshot = _snapshot(frozenset({"ollama"}))

    assert snapshot.allows_model("Ollama", "llama3", model_type="llm")
    assert not snapshot.allows_model("OpenAI", "gpt-4o", model_type="llm")
    assert snapshot.allows_model("OpenAI", OPENAI_EMBEDDING, model_type="embeddings")
    # An untyped request may be for a language model.
    assert not snapshot.allows_model("OpenAI", "gpt-4o")
    with pytest.raises(ModelProviderPolicyError):
        snapshot.require_model("OpenAI", "gpt-4o", model_type="llm")
    with pytest.raises(ModelProviderPolicyError):
        snapshot.require_model_type("OpenAI", "llm")
    snapshot.require_model_type("OpenAI", "embeddings")


@pytest.mark.usefixtures("locked_to_ollama")
def test_resolved_snapshots_drop_providers_left_with_nothing_to_offer():
    snapshot = _resolve()

    assert snapshot.language_model_provider_ids == frozenset({"ollama"})
    # OpenAI still serves embeddings; Anthropic serves language models only.
    assert snapshot.allows("OpenAI")
    assert not snapshot.allows("Anthropic")
    assert snapshot.allows("Ollama")


def test_the_lock_is_part_of_the_snapshot_cache_key():
    service = ModelProviderPolicyService()
    context = ModelProviderPolicyContext(user_id=None)
    candidates = frozenset({"openai", "anthropic"})

    before = service.resolve(context=context, candidate_provider_ids=candidates, purpose=ModelProviderPolicyPurpose.USE)
    lock_language_models_to(["Ollama"])
    try:
        after = service.resolve(
            context=context, candidate_provider_ids=candidates, purpose=ModelProviderPolicyPurpose.USE
        )
    finally:
        lock_language_models_to(None)

    assert before.allows("Anthropic")
    assert not after.allows("Anthropic")
    assert language_model_provider_lock() is None


@pytest.mark.parametrize(
    ("outputs", "model_types"),
    [
        (["text_response", "build_model"], ("llm",)),
        (["build_embeddings"], ("embeddings",)),
        (["build_model", "build_embeddings"], ("embeddings", "llm")),
    ],
)
def test_palette_reads_a_components_model_type_from_its_outputs(outputs, model_types):
    component = {"outputs": [{"method": method} for method in outputs]}

    assert _palette_model_types(component) == model_types


def _palette_entry(provider_id: str | None, *methods: str) -> dict:
    metadata = {"model_provider_id": provider_id} if provider_id else {}
    return {"metadata": metadata, "outputs": [{"method": method} for method in methods]}


@pytest.mark.usefixtures("client", "locked_to_ollama")
async def test_palette_hides_language_model_components_of_locked_out_providers():
    palette = {
        "models": {
            "OpenAIModel": _palette_entry("openai", "text_response", "build_model"),
            "OpenAIEmbeddings": _palette_entry("openai", "build_embeddings"),
            "AnthropicModel": _palette_entry("anthropic", "text_response", "build_model"),
            "OllamaModel": _palette_entry("ollama", "text_response", "build_model"),
            "ChatInput": _palette_entry(None, "message_response"),
        }
    }

    filtered = await _filter_component_palette_by_provider_policy(palette, user_id=None)

    assert sorted(filtered["models"]) == ["ChatInput", "OllamaModel", "OpenAIEmbeddings"]


@pytest.mark.usefixtures("client", "locked_to_ollama")
async def test_a_legacy_agent_provider_dropdown_is_held_to_the_lock():
    agent = AgentComponent()

    with pytest.raises(ModelProviderPolicyError):
        await agent.arequire_model_provider_policy(
            ModelProviderPolicyPurpose.USE,
            parameters={"agent_llm": "OpenAI", "model_name": "gpt-4o"},
        )
    # Once a ModelInput selection drives the agent, a leftover legacy value does not matter.
    await agent.arequire_model_provider_policy(
        ModelProviderPolicyPurpose.USE,
        parameters={"agent_llm": "OpenAI", "model": [{"name": "llama3", "provider": "Ollama", "metadata": {}}]},
    )
