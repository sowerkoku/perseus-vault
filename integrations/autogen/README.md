# Perseus Vault for AutoGen

Persistent long-term memory for [AutoGen](https://github.com/microsoft/autogen)
(AG2 / `autogen-core` v0.4+) agents, backed by [Perseus Vault](https://github.com/Perseus-Computing-LLC/perseus-vault).

`PerseusVaultMemory` implements the `autogen_core.memory.Memory` protocol, so it drops
straight into an `AssistantAgent(memory=[...])`. Stored knowledge is injected
into the model context before each inference, giving your agents memory that
survives across sessions, processes, and crews.

## Install

```bash
# Install Perseus Vault (the binary)
curl -sSL https://raw.githubusercontent.com/Perseus-Computing-LLC/perseus-vault/main/scripts/bootstrap.sh | bash

# Install the adapter
pip install -e integrations/autogen
```

## Usage

```python
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from perseus_vault_autogen import PerseusVaultMemory


async def main():
    memory = PerseusVaultMemory(db_path="~/.perseus-vault/data/agent.db")

    # Seed a fact
    from autogen_core.memory import MemoryContent, MemoryMimeType
    await memory.add(MemoryContent(
        content="The user prefers TypeScript over JavaScript.",
        mime_type=MemoryMimeType.TEXT,
        metadata={"category": "preferences", "key": "language"},
    ))

    agent = AssistantAgent(
        name="assistant",
        model_client=OpenAIChatCompletionClient(model="gpt-4o"),
        memory=[memory],
    )

    result = await agent.run(task="What language should I use for this project?")
    print(result.messages[-1].content)

    await memory.close()


asyncio.run(main())
```

## How it maps to Perseus Vault

| AutoGen `Memory` method | Perseus Vault tool | Behavior |
|---|---|---|
| `add(MemoryContent)` | `perseus_vault_remember` | Content → `body_json`; `metadata.category`/`metadata.key` route the entity |
| `query(text)` | `perseus_vault_recall` | FTS5 keyword search → list of `MemoryContent` |
| `update_context(ctx)` | `perseus_vault_context` | Prepends the rendered memory block as a `SystemMessage` |
| `clear()` | `perseus_vault_prune` | Soft-deletes (archives) this memory's category |
| `close()` | — | Shuts down the persistent Perseus Vault stdio process |

## Configuration

```python
PerseusVaultMemory(
    binary="perseus-vault",               # or absolute path: /usr/local/bin/perseus-vault
    db_path="~/.perseus-vault/data/perseus-vault.db",
    category="autogen",                   # default category for add()
    context_limit=10,                     # entities injected by update_context()
    encryption_key="~/.perseus-vault/secret.key", # optional AES-256-GCM at rest
    llm_endpoint="http://localhost:11434/api/generate",  # optional, for hybrid search
    llm_model="nomic-embed-text",         # optional embedding/RAG model
)
```

## Notes

- The adapter keeps a **persistent** Perseus Vault stdio session — the process is
  spawned once and reused across all calls (no per-call cold start). Call
  `await memory.close()` when done, or let `__del__` reap it.
- `add()` accepts `metadata={"category": ..., "key": ...}` to control where the
  entity lands. Without an explicit key, a timestamped key is generated so
  repeated `add()` calls never collide.
- Use a **project-specific** `db_path` to isolate memories per agent or per
  workspace.
