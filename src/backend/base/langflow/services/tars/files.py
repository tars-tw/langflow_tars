"""The attachments LibreChat sends with a Workflow tool call, staged where the flow's Chat Input reads them.

LibreChat adds ``__librechat_files`` (``[{id, filename, path}]``) to every Workflow tool call; ``path`` is a
short-lived signed reference relative to LibreChat's origin, which is this deployment's
``TARS_LIBRECHAT_URL`` and never a host named by the call. A flow receives the files only once it opts in
by putting Chat Input's Files field on the node, which is also what puts ``files`` into its input schema;
every other flow runs exactly as before.

Files land in the flow's own storage directory under their original name: Langflow's storage keys are a
single ``<flow id>/<file name>`` segment pair, and the name is how a flow tells its documents apart (the
model reads "File <name> contents"). A name already holding other content gets " (2)", " (3)", ... so
concurrent calls never overwrite each other's files; identical content is shared. Each staged file has a
``.librechat-<file id>`` marker, and both are removed once unused for ``TARS_LIBRECHAT_FILE_TTL_SECONDS``.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
import httpx
from lfx.log.logger import logger

from langflow.services.deps import get_settings_service, get_storage_service

if TYPE_CHECKING:
    from uuid import UUID

WORKFLOW_FILES_ARGUMENT = "__librechat_files"
FILES_FIELD = "files"

LIBRECHAT_URL_ENV = "TARS_LIBRECHAT_URL"
MAX_BYTES_ENV = "TARS_LIBRECHAT_FILE_MAX_BYTES"
TTL_ENV = "TARS_LIBRECHAT_FILE_TTL_SECONDS"
DEFAULT_MAX_BYTES = 50 * 1024 * 1024
DEFAULT_TTL_SECONDS = 60 * 60
DOWNLOAD_TIMEOUT_SECONDS = 60.0

REF_PATH_PREFIX = "/api/tars/files/"
MARKER_PREFIX = ".librechat-"
MAX_NAME_LENGTH = 200
MAX_NAME_ATTEMPTS = 50
_FILE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


@dataclass(frozen=True)
class LibreChatFileRef:
    id: str
    filename: str
    path: str


def accepts_files(input_schema: dict) -> bool:
    """Whether a flow opted in: its Chat Input's Files field is on the node, so it is an exposed input."""
    return FILES_FIELD in input_schema.get("properties", {})


def without_files_field(input_schema: dict) -> dict:
    """The schema the model sees: ``files`` takes server paths, which only this module may name."""
    properties = {name: field for name, field in input_schema.get("properties", {}).items() if name != FILES_FIELD}
    required = [name for name in input_schema.get("required", []) if name != FILES_FIELD]
    return {**input_schema, "properties": properties, "required": required}


def parse_refs(value: Any) -> list[LibreChatFileRef]:
    """The well-formed references in ``__librechat_files``; anything else is dropped."""
    if not isinstance(value, list):
        return []
    refs: list[LibreChatFileRef] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        file_id, filename, path = item.get("id"), item.get("filename"), item.get("path")
        if not isinstance(file_id, str) or not _FILE_ID.match(file_id):
            continue
        if not isinstance(path, str) or not path.startswith(f"{REF_PATH_PREFIX}{file_id}?"):
            continue
        refs.append(LibreChatFileRef(id=file_id, filename=safe_file_name(filename, file_id), path=path))
    return refs


def safe_file_name(name: Any, fallback: str) -> str:
    """A single storage segment keeping the original name as far as Langflow's storage keys allow."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1] if isinstance(name, str) else ""
    base = re.sub(r"\.{2,}", ".", base.replace("\x00", "")).strip()
    if not base or base == "." or base.startswith(MARKER_PREFIX):
        return fallback
    if len(base) <= MAX_NAME_LENGTH:
        return base
    stem, dot, suffix = base.rpartition(".")
    return f"{stem[: MAX_NAME_LENGTH - len(suffix) - 1]}.{suffix}" if dot else base[:MAX_NAME_LENGTH]


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default
    return value if value > 0 else default


def _candidate_names(name: str):
    yield name
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        stem, suffix = name, ""
    for n in range(2, MAX_NAME_ATTEMPTS + 1):
        yield f"{stem} ({n}).{suffix}" if suffix else f"{stem} ({n})"


def _sha256_of(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _publish(flow_dir: Path, ref: LibreChatFileRef, data: bytes) -> str | None:
    """Writes ``data`` under the first free (or identical) name; returns that name.

    ``os.link`` publishes the finished file atomically and fails when the name is taken, so two
    processes staging the same name can never interleave or overwrite.
    """
    digest = hashlib.sha256(data).hexdigest()
    flow_dir.mkdir(parents=True, exist_ok=True)
    tmp = flow_dir / f"{MARKER_PREFIX}{ref.id}.{os.getpid()}.{time.monotonic_ns()}.part"
    tmp.write_bytes(data)
    try:
        for name in _candidate_names(ref.filename):
            target = flow_dir / name
            try:
                os.link(tmp, target)
            except FileExistsError:
                if _sha256_of(target) != digest:
                    continue
                os.utime(target)
            (flow_dir / f"{MARKER_PREFIX}{ref.id}").write_text(name, encoding="utf-8")
            return name
    finally:
        tmp.unlink(missing_ok=True)
    return None


def _purge(flow_dir: Path, ttl_seconds: int) -> None:
    """Removes staged files and their markers once unused for the TTL."""
    cutoff = time.time() - ttl_seconds
    try:
        markers = [entry for entry in flow_dir.iterdir() if entry.name.startswith(MARKER_PREFIX)]
    except OSError:
        return
    for marker in markers:
        try:
            if marker.stat().st_mtime > cutoff:
                continue
            if not marker.name.endswith(".part"):
                staged = flow_dir / safe_file_name(marker.read_text(encoding="utf-8"), "")
                if staged != flow_dir and staged.is_file() and staged.stat().st_mtime <= cutoff:
                    staged.unlink()
            marker.unlink()
        except OSError:
            continue


async def _download(client: httpx.AsyncClient, url: str, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    size = 0
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            size += len(chunk)
            if size > max_bytes:
                msg = f"larger than {max_bytes} bytes ({MAX_BYTES_ENV})"
                raise ValueError(msg)
            chunks.append(chunk)
    return b"".join(chunks)


def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=False)


async def stage_librechat_files(flow_id: UUID | str, refs: list[LibreChatFileRef]) -> list[str]:
    """Downloads the referenced files into the flow's storage; returns their ``<flow id>/<name>`` keys.

    Fail-soft per file: one that cannot be fetched is left out (and logged), so the flow still runs and
    says what it is missing rather than the whole call failing.
    """
    if not refs:
        return []
    base_url = os.getenv(LIBRECHAT_URL_ENV, "").strip().rstrip("/")
    if not base_url:
        await logger.awarning(f"LibreChat attachments ignored: {LIBRECHAT_URL_ENV} is not set")
        return []
    if str(get_settings_service().settings.storage_type).lower() == "s3":
        await logger.awarning("LibreChat attachments ignored: only local storage is supported")
        return []

    flow_dir = Path(str(get_storage_service().data_dir)) / str(flow_id)
    max_bytes = _env_int(MAX_BYTES_ENV, DEFAULT_MAX_BYTES)
    await anyio.to_thread.run_sync(_purge, flow_dir, _env_int(TTL_ENV, DEFAULT_TTL_SECONDS))

    keys: list[str] = []
    async with _http_client() as client:
        for ref in refs:
            try:
                data = await _download(client, f"{base_url}{ref.path}", max_bytes)
                name = await anyio.to_thread.run_sync(_publish, flow_dir, ref, data)
            except (httpx.HTTPError, OSError, ValueError) as e:
                await logger.awarning(f"LibreChat attachment {ref.id} not staged: {e!s}")
                continue
            if name is None:
                await logger.awarning(f"LibreChat attachment {ref.id} not staged: no free file name")
                continue
            keys.append(f"{flow_id}/{name}")
    return keys
