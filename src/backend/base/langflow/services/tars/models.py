"""pwc_tars as Langflow's model provider.

The unified model picker (Language Model, Agent) lists the chat models pwc_tars lets the
person use (``model_profile`` filtered by ``sys_model_permission``) and runs them through
pwc_tars's OpenAI-compatible ``/api/langflow-service/v1/chat/completions``, so the model
menu, the permission check and the provider keys stay in pwc_tars.

Who a call is for:

* the picker (editing a flow) lists the models of the pwc_tars user the editor's account is
  linked to, or the operator an actor cookie names;
* a run is checked against the person chatting: the run's ``TARS_USER_ID`` request
  variable (LibreChat's MCP header, or the canvas operator), else the pwc_tars user the
  executing account is linked to. pwc_tars refuses a call that names nobody.

Once registered, language models come from TARS alone (``TARS_LANGUAGE_MODELS_ONLY=false``
lifts that); embedding models keep their own providers.
"""

from __future__ import annotations

import os
import threading
import time
from typing import TYPE_CHECKING, Any
from uuid import UUID

import httpx
from langchain_openai import ChatOpenAI
from lfx.base.models.model_metadata import create_model_metadata
from lfx.base.models.provider_registry import ProviderDescriptor, register_provider
from lfx.log.logger import logger
from lfx.memory.flow_context import get_current_message_executor_id, get_current_request_variables
from lfx.services.model_provider_policy import lock_language_models_to
from lfx.utils.async_helpers import run_until_complete
from pydantic import model_validator

from langflow.services.tars.actor import TARS_USER_VARIABLE, current_actor

if TYPE_CHECKING:
    from collections.abc import Callable

PROVIDER = "TARS"
PROVIDER_ID = "tars"
ICON = "Bot"
BASE_URL_ENV = "TARS_BASE_URL"
SERVICE_KEY_ENV = "TARS_SERVICE_KEY"
LANGUAGE_MODELS_ONLY_ENV = "TARS_LANGUAGE_MODELS_ONLY"
DEFAULT_BASE_URL = "http://localhost:5000"
SERVICE_KEY_HEADER = "X-TARS-Service-Key"
USER_HEADER = "X-TARS-User-Id"
COMPLETIONS_PATH = "/api/langflow-service/v1"
MODELS_PATH = "/api/langflow-service/models"

MODEL_LIST_TTL_SECONDS = 30.0
_LIST_TIMEOUT_SECONDS = 10.0
_RUN_TIMEOUT_SECONDS = 300.0
# pwc_tars authenticates with the service key; the OpenAI client still insists on some key.
_PLACEHOLDER_API_KEY = "tars"  # pragma: allowlist secret

_model_lists: dict[str, tuple[float, list[dict]]] = {}
_model_lists_lock = threading.Lock()


def tars_base_url() -> str:
    return (os.getenv(BASE_URL_ENV, "").strip() or DEFAULT_BASE_URL).rstrip("/")


def tars_service_key() -> str:
    return os.getenv(SERVICE_KEY_ENV, "").strip()


def is_configured() -> bool:
    return bool(tars_service_key())


def language_models_only() -> bool:
    return os.getenv(LANGUAGE_MODELS_ONLY_ENV, "").strip().lower() not in {"false", "0", "no", "off"}


async def _linked_tars_id(user_id: UUID | str) -> str | None:
    from langflow.services.deps import session_scope
    from langflow.services.tars.accounts import find_tars_id

    async with session_scope() as db:
        return await find_tars_id(db, user_id if isinstance(user_id, UUID) else UUID(str(user_id)))


def tars_id_for_account(user_id: UUID | str | None) -> str | None:
    """The pwc_tars user behind a Langflow account: the operator an actor cookie names, else the linked one."""
    if not user_id or str(user_id).lower() in {"none", "null"}:
        return None
    actor = current_actor()
    if actor is not None and actor.langflow_user_id == str(user_id):
        return actor.tars_id
    try:
        return run_until_complete(_linked_tars_id(user_id))
    except (ValueError, TypeError):
        return None


def chatting_tars_id() -> str | None:
    """The pwc_tars user the executing run is for (see the module docstring)."""
    tars_id = str(get_current_request_variables().get(TARS_USER_VARIABLE) or "").strip()
    if tars_id:
        return tars_id
    return tars_id_for_account(get_current_message_executor_id())


def _fetch_models(tars_id: str) -> list[dict]:
    response = httpx.get(
        f"{tars_base_url()}{MODELS_PATH}",
        headers={SERVICE_KEY_HEADER: tars_service_key(), USER_HEADER: tars_id},
        timeout=_LIST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return list((response.json().get("data") or {}).get("models") or [])


def _model_rows(models: list[dict], model_type: str) -> list[dict]:
    return [
        create_model_metadata(
            provider=PROVIDER,
            name=model["name"],
            icon=ICON,
            tool_calling=bool(model.get("tool_calling", True)),
            # Everything pwc_tars allows is offered; the user's enabled/disabled toggles still apply.
            default=True,
            model_type=model_type,
        )
        for model in models
        if model.get("name")
    ]


def fetch_live_tars_models(user_id: UUID | str | None, model_type: str = "llm") -> list[dict]:
    """Live discovery: the chat models pwc_tars lets this account's person use; empty when that is unknown."""
    if model_type != "llm" or not is_configured():
        return []
    tars_id = tars_id_for_account(user_id)
    if not tars_id:
        return []
    now = time.monotonic()
    with _model_lists_lock:
        cached = _model_lists.get(tars_id)
    if cached and cached[0] > now:
        return _model_rows(cached[1], model_type)
    try:
        models = _fetch_models(tars_id)
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning(f"Could not list TARS models for {tars_id}: {exc}")
        return []
    with _model_lists_lock:
        _model_lists[tars_id] = (now + MODEL_LIST_TTL_SECONDS, models)
    return _model_rows(models, model_type)


def clear_model_list_cache() -> None:
    with _model_lists_lock:
        _model_lists.clear()


class TarsChatModel(ChatOpenAI):
    """``ChatOpenAI`` pointed at pwc_tars's completion route, acting for the person chatting."""

    @model_validator(mode="before")
    @classmethod
    def _connect_to_tars(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        service_key = tars_service_key()
        if not service_key:
            msg = f"TARS model provider is not configured: set {SERVICE_KEY_ENV} and restart Langflow."
            raise ValueError(msg)
        headers = dict(values.get("default_headers") or {})
        headers[SERVICE_KEY_HEADER] = service_key
        tars_id = chatting_tars_id()
        if tars_id:
            headers[USER_HEADER] = tars_id
        # The endpoint and credentials are pwc_tars's alone, whatever the caller passed.
        overridden = {"openai_api_base", "base_url", "openai_api_key", "api_key"}
        values = {key: value for key, value in values.items() if key not in overridden}
        values["base_url"] = f"{tars_base_url()}{COMPLETIONS_PATH}"
        values["default_headers"] = headers
        values["api_key"] = _PLACEHOLDER_API_KEY
        values.setdefault("timeout", _RUN_TIMEOUT_SECONDS)
        # pwc_tars counts tokens per user, so ask for usage on streamed calls too.
        values.setdefault("stream_usage", True)
        return values


TARS_PROVIDER_DESCRIPTOR = ProviderDescriptor(
    name=PROVIDER,
    provider_id=PROVIDER_ID,
    display_name=PROVIDER,
    metadata={
        "icon": ICON,
        "variables": [],
        "mapping": {"model_class": "TarsChatModel", "model_param": "model"},
        "max_tokens_field_name": "max_tokens",
    },
    model_class=(__name__, "TarsChatModel", None),
    api_key_required=False,
    live=True,
    live_discovery=f"{__name__}:fetch_live_tars_models",
)


def register_tars_model_provider(register: Callable[[ProviderDescriptor], bool] = register_provider) -> bool:
    """Add TARS to the unified model providers when this Langflow can reach pwc_tars.

    Also locks language models to TARS unless ``TARS_LANGUAGE_MODELS_ONLY`` says otherwise.
    """
    if not is_configured():
        logger.info(f"TARS model provider not registered: {SERVICE_KEY_ENV} is not set")
        return False
    registered = register(TARS_PROVIDER_DESCRIPTOR)
    if not registered:
        logger.warning("TARS model provider not registered: the name is already taken")
        return False
    if language_models_only():
        lock_language_models_to([PROVIDER_ID])
        logger.info("Language models are limited to the TARS provider; embedding providers are unchanged")
    return True
