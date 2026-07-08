"""
PerseusVaultStore — LangGraph BaseStore implementation backed by Perseus Vault.

Drop-in persistent long-term memory for LangGraph agents.
Maps LangGraph's namespace/key/value model to Perseus Vault's entity model.

Usage:
    from perseus_vault_langgraph import PerseusVaultStore
    from langgraph.store.memory import InMemoryStore

    store = PerseusVaultStore()  # connects to local Perseus Vault via MCP stdio
    # Or with explicit config:
    store = PerseusVaultStore(
        binary="/usr/local/bin/perseus-vault",
        db_path="~/.perseus-vault/data/perseus-vault.db"
    )

    # Use as any BaseStore
    store.put(("users", "123"), "prefs", {"theme": "dark"})
    item = store.get(("users", "123"), "prefs")
    results = store.search(("users",), query="preferences")
"""

from __future__ import annotations

import json
import subprocess
import time
import threading
import logging
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

from langgraph.store.base import BaseStore, Item, SearchItem, Op, Result

# The "no TTL given" sentinel was renamed NOT_GIVEN -> NOT_PROVIDED in
# LangGraph 1.0. Support both so the adapter imports across versions.
try:
    from langgraph.store.base import NOT_PROVIDED as _NOT_GIVEN
except ImportError:  # langgraph < 1.0
    from langgraph.store.base import NOT_GIVEN as _NOT_GIVEN

logger = logging.getLogger(__name__)


class PerseusVaultStore(BaseStore):
    """LangGraph BaseStore backed by a local Perseus Vault MCP server.

    Perseus Vault is a local-first persistent memory engine with structured entities,
    journal events, and state management. This adapter maps LangGraph's
    namespace/key/value model onto Perseus Vault's entity model.

    Mapping:
        namespace tuple  → Perseus Vault category (joined with '/')
        key              → Perseus Vault key
        value dict       → Perseus Vault body_json
        search query     → Perseus Vault FTS5 recall
    """

    def __init__(
        self,
        binary: str = "perseus-vault",
        db_path: str = "~/.perseus-vault/data/perseus-vault.db",
        timeout: float = 30.0,
        connect_timeout: float = 10.0,
        encryption_key: Optional[str] = None,
        ollama_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """Initialize the Perseus Vault-backed store.

        Args:
            binary: Path to the perseus-vault binary (default: finds on PATH)
            db_path: Path to the Perseus Vault SQLite database
            timeout: Command timeout in seconds
            connect_timeout: MCP handshake timeout in seconds
            encryption_key: Optional path to AES-256-GCM key file
            ollama_url: Optional Ollama endpoint for hybrid search
            embedding_model: Optional embedding model name (requires ollama_url)
        """
        self.binary = binary
        self.db_path = str(Path(db_path).expanduser())
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        self.encryption_key = encryption_key
        self.ollama_url = ollama_url
        self.embedding_model = embedding_model
        self._proc: Optional[subprocess.Popen] = None
        self._req_id: int = 0
        self._lock = threading.Lock()

    def _ensure_session(self):
        """Spawn a persistent perseus-vault process if one isn't already running."""
        if self._proc is not None and self._proc.poll() is None:
            return

        args = [self.binary, "serve", "--db", self.db_path]
        if self.encryption_key:
            args.extend(["--encryption-key", self.encryption_key])
        # ollama_url maps to Perseus Vault's --llm-endpoint (there is no --ollama-url flag).
        if self.ollama_url:
            args.extend(["--llm-endpoint", self.ollama_url])
        # embedding_model maps to --llm-model for Ollama-backed embeddings.
        # (Perseus Vault's --embedding-model expects an ONNX model *path*, not a name.)
        if self.embedding_model:
            args.extend(["--llm-model", self.embedding_model])

        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._req_id = 0

        # Send initialize
        init_id = self._req_id + 1
        self._req_id = init_id
        init_req = json.dumps({
            "jsonrpc": "2.0", "id": init_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "langgraph-perseus-vault", "version": "1.0.0"},
            },
        })
        try:
            self._proc.stdin.write(init_req + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._proc = None
            raise RuntimeError("Failed to initialize perseus-vault process")

        # Read the initialize response
        self._read_response(init_id)

    def _read_response(self, expect_id: int) -> Optional[dict]:
        """Read a single JSON-RPC response matching *expect_id*."""
        assert self._proc is not None
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                return None
            line = self._proc.stdout.readline()
            if not line:
                time.sleep(0.01)
                continue
            try:
                msg = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            if isinstance(msg, dict) and msg.get("id") == expect_id:
                return msg
        return None

    def _close_session(self):
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None

    def __del__(self):
        self._close_session()

    def _namespace_to_category(self, namespace: tuple[str, ...]) -> str:
        """Convert LangGraph namespace tuple to Perseus Vault category string."""
        return "/".join(namespace) if namespace else "default"

    def _category_to_namespace(self, category: str) -> tuple[str, ...]:
        """Convert Perseus Vault category string back to namespace tuple."""
        return tuple(category.split("/")) if category != "default" else ()

    @staticmethod
    def _unwrap_result(result: dict) -> dict:
        """Unwrap an MCP ``tools/call`` result into Perseus Vault's payload dict.

        Perseus Vault returns the standard MCP envelope::

            {"content": [{"type": "text", "text": "<json>"}],
             "structuredContent": {...parsed json...}}

        The actual payload (``items``, ``by_category``, ``context`` ...) lives
        in ``structuredContent`` (preferred) or, failing that, in the JSON text
        of the first content block. Reading ``result["items"]`` directly always
        yields nothing, so callers must go through this helper.
        """
        structured = result.get("structuredContent")
        if isinstance(structured, dict):
            return structured
        content = result.get("content")
        if isinstance(content, list) and content:
            text = content[0].get("text", "") if isinstance(content[0], dict) else ""
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                return {}
            if isinstance(parsed, dict):
                return parsed
        return {}

    def _call_perseus_vault(self, method: str, params: dict) -> dict:
        """Call a Perseus Vault MCP tool via the persistent stdio session.

        The process is spawned once on first call and reused across all
        subsequent calls — no per-call cold-start overhead.
        """
        with self._lock:
            try:
                self._ensure_session()
            except RuntimeError as e:
                raise RuntimeError(f"Perseus Vault session failed: {e}")

            # No init response to consume — _ensure_session already read it

            req_id = self._req_id + 2
            call_req = json.dumps({
                "jsonrpc": "2.0", "id": req_id,
                "method": "tools/call",
                "params": {"name": method, "arguments": params},
            })

            try:
                self._proc.stdin.write(call_req + "\n")
                self._proc.stdin.flush()
                self._req_id = req_id
            except (BrokenPipeError, OSError):
                self._proc = None
                raise RuntimeError("Perseus Vault process died during call")

            response = self._read_response(req_id)
            if response is None:
                self._close_session()
                raise RuntimeError(f"No response from Perseus Vault for {method}")
            if response.get("error"):
                raise RuntimeError(f"Perseus Vault error ({method}): {response['error']}")
            return self._unwrap_result(response.get("result", {}))

    @staticmethod
    def _ms_to_dt(ms: Any) -> datetime:
        """Convert a Perseus Vault ``*_unix_ms`` timestamp to a UTC ``datetime``.

        ``Item.created_at`` / ``updated_at`` are typed ``datetime``; the epoch
        is used as a fallback when a record carries no usable timestamp.
        """
        epoch = datetime.fromtimestamp(0, tz=timezone.utc)
        if not ms:
            return epoch
        try:
            return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc)
        except (ValueError, TypeError, OSError):
            return epoch

    def put(
        self,
        namespace: tuple[str, ...],
        key: str,
        value: dict[str, Any],
        index: list[str] | Literal[False] | None = None,  # type: ignore[name-defined]
        *,
        ttl: float | None | Any = _NOT_GIVEN,
    ) -> None:
        """Store a value in Perseus Vault.

        Maps to perseus_vault_remember with category=namespace, key=key.
        The value dict becomes body_json. TTL is stored as a state entry.
        """
        category = self._namespace_to_category(namespace)

        result = self._call_perseus_vault("perseus_vault_remember", {
            "category": category,
            "key": key,
            "body_json": json.dumps(value),
            "type": "langgraph_item",
        })

        # Handle TTL via Perseus Vault state
        if ttl is not _NOT_GIVEN and ttl is not None:
            self._call_perseus_vault("perseus_vault_state_set", {
                "key": f"{category}/{key}__ttl",
                "value": str(time.time() + float(ttl)),
                "ttl": float(ttl),
            })

    async def aput(self, *args, **kwargs) -> None:
        """Async variant — delegates to sync put."""
        self.put(*args, **kwargs)

    def get(
        self,
        namespace: tuple[str, ...],
        key: str,
        *,
        refresh_ttl: bool | None = None,
    ) -> Item | None:
        """Retrieve a value from Perseus Vault.

        Maps to perseus_vault_recall filtered by category + key.
        """
        category = self._namespace_to_category(namespace)

        result = self._call_perseus_vault("perseus_vault_recall", {
            "query": key,
            "category": category,
            "limit": 5,
        })

        items = result.get("items", [])
        for item in items:
            if item.get("key") == key:
                try:
                    value = json.loads(item.get("body_json", "{}"))
                except (json.JSONDecodeError, TypeError):
                    value = {}

                return Item(
                    namespace=namespace,
                    key=key,
                    value=value,
                    created_at=self._ms_to_dt(item.get("created_at_unix_ms")),
                    updated_at=self._ms_to_dt(
                        item.get("last_accessed_unix_ms")
                        or item.get("created_at_unix_ms")
                    ),
                )

        return None

    async def aget(self, *args, **kwargs) -> Item | None:
        """Async variant — delegates to sync get."""
        return self.get(*args, **kwargs)

    def search(
        self,
        namespace_prefix: tuple[str, ...],
        /,
        *,
        query: str | None = None,
        filter: dict[str, Any] | None = None,
        limit: int = 10,
        offset: int = 0,
        refresh_ttl: bool | None = None,
    ) -> list[SearchItem]:
        """Search for items in Perseus Vault.

        Uses Perseus Vault's FTS5 keyword search. The namespace_prefix becomes
        a category filter.
        """
        category = self._namespace_to_category(namespace_prefix)
        search_query = query or ""

        params = {
            "query": search_query,
            "limit": limit,
            "offset": offset,
        }
        if category and category != "default":
            params["category"] = category

        result = self._call_perseus_vault("perseus_vault_recall", params)
        items = result.get("items", [])

        results = []
        for item in items:
            try:
                value = json.loads(item.get("body_json", "{}"))
            except (json.JSONDecodeError, TypeError):
                value = {}

            results.append(SearchItem(
                namespace=namespace_prefix,
                key=item.get("key", ""),
                value=value,
                created_at=self._ms_to_dt(item.get("created_at_unix_ms")),
                updated_at=self._ms_to_dt(
                    item.get("last_accessed_unix_ms")
                    or item.get("created_at_unix_ms")
                ),
                score=item.get("decay_score"),
            ))

        return results

    async def asearch(self, *args, **kwargs) -> list[SearchItem]:
        """Async variant — delegates to sync search."""
        return self.search(*args, **kwargs)

    def delete(self, namespace: tuple[str, ...], key: str) -> None:
        """Delete an item from Perseus Vault.

        Maps to perseus_vault_forget (soft-delete with archived=1).
        """
        category = self._namespace_to_category(namespace)
        self._call_perseus_vault("perseus_vault_forget", {
            "category": category,
            "key": key,
            "reason": "LangGraph delete",
        })

    async def adelete(self, *args, **kwargs) -> None:
        """Async variant — delegates to sync delete."""
        self.delete(*args, **kwargs)

    def list_namespaces(
        self,
        *,
        prefix: Any | None = None,
        suffix: Any | None = None,
        max_depth: int | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[tuple[str, ...]]:
        """List all namespaces (categories) in Perseus Vault."""
        result = self._call_perseus_vault("perseus_vault_stats", {})
        # perseus_vault_stats returns category counts under "by_category" (a mapping of
        # category name -> count), not a "categories" list.
        by_category = result.get("by_category", {})

        namespaces = []
        for cat in by_category:
            ns = self._category_to_namespace(cat)
            namespaces.append(ns)

        return namespaces[offset:offset + limit]

    async def alist_namespaces(self, *args, **kwargs) -> list[tuple[str, ...]]:
        """Async variant — delegates to sync list_namespaces."""
        return self.list_namespaces(*args, **kwargs)

    def batch(self, ops: Iterable[Op]) -> list[Result]:  # type: ignore[name-defined]
        """Execute a batch of operations."""
        results = []
        for op in ops:
            try:
                if op[0] == "put":
                    self.put(*op[1], **op[2] if len(op) > 2 else {})
                    results.append(None)
                elif op[0] == "delete":
                    self.delete(*op[1])
                    results.append(None)
                elif op[0] == "get":
                    results.append(self.get(*op[1], **op[2] if len(op) > 2 else {}))
                elif op[0] == "search":
                    results.append(self.search(*op[1], **op[2] if len(op) > 2 else {}))
                else:
                    results.append(None)
            except Exception as e:
                logger.error(f"Batch op {op[0]} failed: {e}")
                results.append(None)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:  # type: ignore[name-defined]
        """Async variant — delegates to sync batch."""
        return self.batch(ops)


# Convenience helper
def create_perseus_vault_store(
    db_path: str = "~/.perseus-vault/data/perseus-vault.db",
    **kwargs,
) -> PerseusVaultStore:
    """Create a PerseusVaultStore with sensible defaults.

    Args:
        db_path: Path to the Perseus Vault database
        **kwargs: Additional PerseusVaultStore arguments
    """
    return PerseusVaultStore(db_path=db_path, **kwargs)
