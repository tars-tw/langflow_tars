"""Process-wide lock restricting language models to a set of providers.

Embedding models are untouched: a locked-out provider that also serves embeddings
stays allowed for them, while one serving language models only is not offered at
all. ``None`` (the default) means no lock. Set it at startup, before any policy
snapshot is resolved; snapshots are cached.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_lock = threading.Lock()
_language_model_provider_ids: frozenset[str] | None = None


def lock_language_models_to(provider_ids: Iterable[str] | None) -> None:
    """Allow language models from these providers only (stable IDs or names); ``None`` lifts the lock."""
    global _language_model_provider_ids  # noqa: PLW0603 - one process-wide setting
    if provider_ids is None:
        locked = None
    else:
        from lfx.base.models.provider_registry import resolve_provider_id

        locked = frozenset(resolve_provider_id(provider) for provider in provider_ids)
    with _lock:
        _language_model_provider_ids = locked


def language_model_provider_lock() -> frozenset[str] | None:
    """Return the providers language models are locked to, or ``None`` when unlocked."""
    with _lock:
        return _language_model_provider_ids
