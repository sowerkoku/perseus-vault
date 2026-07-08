# Perseus Vault CrewAI Integration

Persistent memory for CrewAI agents via Perseus Vault.

## Install

Install from source (not yet published to PyPI):

```bash
pip install crewai
pip install -e integrations/crewai/
```

## Quick Start

```python
from crewai import Agent, Task, Crew
from perseus_vault_crewai import PerseusVaultMemoryTool

# Create the memory tool
memory = PerseusVaultMemoryTool(
    db_path="~/.perseus-vault/data/crew.db"
)

# Give it to your agents
researcher = Agent(
    role="Senior Researcher",
    goal="Find and analyze information",
    backstory="Expert at gathering and synthesizing data",
    tools=[memory],
    verbose=True,
)

# Agents use it naturally
task = Task(
    description=(
        "Research the competitor's pricing strategy. "
        "Use Perseus Vault Memory to recall any previous findings on this topic, "
        "then remember your new conclusions."
    ),
    agent=researcher,
    expected_output="A report with pricing analysis",
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

## Available Actions

| Action | Description | Parameters |
|---|---|---|
| `remember` | Store a fact or decision | `category`, `key`, `content`, `entity_type?` |
| `recall` | Search stored memories | `query`, `category?`, `limit?` |
| `journal` | Record a significant event | `event_type`, `description`, `context?` |
| `context` | Get session context summary | (none) |

## How It Works

The `PerseusVaultMemoryTool` wraps Perseus Vault's MCP tools as a CrewAI tool:

- `remember` → `perseus_vault_remember`
- `recall` → `perseus_vault_recall`
- `journal` → `perseus_vault_journal`
- `context` → `perseus_vault_context`

All memories persist across sessions and crews. Agents can build up
a shared knowledge base over time.

## Requirements

- Perseus Vault v1.0.0+ (`curl -sSL https://raw.githubusercontent.com/Perseus-Computing-LLC/perseus-vault/main/scripts/bootstrap.sh | bash`)
- CrewAI >= 0.30.0
- Python 3.10+
