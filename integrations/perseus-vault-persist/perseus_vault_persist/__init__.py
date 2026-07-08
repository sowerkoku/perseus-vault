"""perseus-vault-persist — a persistent FastMCP ``EventStore`` backed by SQLite.

MCP infrastructure: gives any FastMCP server SSE stream resumability across
restarts. See :class:`perseus_vault_persist.store.PerseusVaultEventStore`.
"""

from perseus_vault_persist.store import PerseusVaultEventStore

__all__ = ["PerseusVaultEventStore"]
__version__ = "0.1.0"
