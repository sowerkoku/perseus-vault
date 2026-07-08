"""Tests for the CrewAI PerseusVaultMemoryTool.

CrewAI (and its heavy dependency tree) need not be installed to exercise the
Perseus Vault wiring: we stub ``crewai.tools.BaseTool`` with a minimal base class, then
drive the tool against Perseus Vault's real MCP JSON-RPC envelope via a fake subprocess.
"""

from __future__ import annotations

import json
import sys
import types

import pytest


@pytest.fixture(scope="module")
def PerseusVaultMemoryTool():
    """Import PerseusVaultMemoryTool with a stubbed ``crewai.tools.BaseTool``."""
    if "crewai" not in sys.modules:
        crewai = types.ModuleType("crewai")
        tools = types.ModuleType("crewai.tools")

        class BaseTool:
            name: str = ""
            description: str = ""

            def __init__(self, *args, **kwargs):
                pass

        tools.BaseTool = BaseTool
        crewai.tools = tools
        sys.modules["crewai"] = crewai
        sys.modules["crewai.tools"] = tools

    from perseus_vault_crewai import PerseusVaultMemoryTool as tool_cls

    return tool_cls


def _fake_popen(routes):
    """Build a FakePopen whose stdout replays JSON-RPC responses over the
    persistent stdio session the tool uses (write/readline, not communicate).

    ``routes`` maps a tool name → callable(args) -> payload dict. The fake
    answers the initialize handshake with ``{}`` and each ``tools/call`` with
    the MCP envelope wrapping the routed payload.
    """
    class FakeStdout:
        def __init__(self):
            self._lines = []

        def push(self, line):
            self._lines.append(line)

        def readline(self):
            if self._lines:
                return self._lines.pop(0)
            return ""

    class FakePopen:
        last_input = None

        def __init__(self, *args, **kwargs):
            self._stdout = FakeStdout()
            self.stderr = None

            class _Stdin:
                def __init__(self, outer):
                    self.outer = outer

                def write(self, data):
                    FakePopen.last_input = data
                    for line in data.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        msg = json.loads(line)
                        mid = msg.get("id")
                        if msg.get("method") == "initialize":
                            self.outer._stdout.push(
                                json.dumps({"jsonrpc": "2.0", "id": mid, "result": {}}) + "\n"
                            )
                        elif msg.get("method") == "tools/call":
                            name = msg["params"]["name"]
                            payload = routes[name](msg["params"]["arguments"])
                            env = {
                                "jsonrpc": "2.0",
                                "id": mid,
                                "result": {
                                    "content": [{"type": "text", "text": json.dumps(payload)}],
                                    "structuredContent": payload,
                                },
                            }
                            self.outer._stdout.push(json.dumps(env) + "\n")

                def flush(self):
                    pass

                def close(self):
                    pass

            self.stdin = _Stdin(self)

        @property
        def stdout(self):
            return self._stdout

        def poll(self):
            return None

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    return FakePopen


def test_remember_sends_type(monkeypatch, PerseusVaultMemoryTool):
    captured = {}

    def remember(args):
        captured.update(args)
        return {"id": "mem-1", "status": "ok"}

    monkeypatch.setattr(
        "perseus_vault_crewai.subprocess.Popen", _fake_popen({"perseus_vault_remember": remember})
    )
    tool = PerseusVaultMemoryTool()
    out = tool._remember(category="crewai", key="k1", content="hello world")

    assert captured.get("type") == "fact"  # regression: was the dropped "entity_type"
    assert "entity_type" not in captured
    assert json.loads(captured["body_json"]) == {"content": "hello world"}
    assert "Remembered" in out


def test_recall_parses_structured_items(monkeypatch, PerseusVaultMemoryTool):
    def recall(args):
        return {
            "items": [
                {
                    "category": "crewai",
                    "key": "k1",
                    "body_json": json.dumps({"content": "the answer is 42"}),
                }
            ],
            "total": 1,
        }

    monkeypatch.setattr(
        "perseus_vault_crewai.subprocess.Popen", _fake_popen({"perseus_vault_recall": recall})
    )
    tool = PerseusVaultMemoryTool()
    out = tool._recall(query="answer")

    # Before the envelope-unwrap fix this returned "No memories found".
    assert "Found 1 memory" in out
    assert "the answer is 42" in out


def test_unwrap_handles_text_only_envelope(PerseusVaultMemoryTool):
    assert PerseusVaultMemoryTool._unwrap_result(
        {"content": [{"type": "text", "text": json.dumps({"items": [1, 2]})}]}
    ) == {"items": [1, 2]}
