"""A scripted stand-in for the Anthropic client, used in tests and the offline end-to-end script.

It mimics the shape of `client.messages.create(...)` responses (content blocks with type text/tool_use and a
stop_reason). Each script step is a function(messages) -> list of blocks, so a step can read ids produced by
earlier tool results (e.g. forecast_id) exactly like the real model would.
"""
from __future__ import annotations

import itertools
import json
from types import SimpleNamespace

_ids = itertools.count(1)


def tool(_tool_name: str, **inp):
    return {"type": "tool_use", "id": f"toolu_{next(_ids)}", "name": _tool_name, "input": inp}


def last_results(messages) -> list[dict]:
    for m in reversed(messages):
        if m["role"] == "user" and isinstance(m["content"], list):
            out = []
            for b in m["content"]:
                if b.get("type") == "tool_result":
                    try:
                        out.append(json.loads(b["content"].split("... [truncated")[0]))
                    except Exception:
                        out.append({"raw": b["content"]})
            return out
    return []


class ScriptedClient:
    def __init__(self, steps):
        self.steps = list(steps)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        step = self.steps.pop(0) if self.steps else (lambda m: [tool("final_answer", answer="done", next_questions=["a", "b"])])
        blocks = step(kw["messages"]) if callable(step) else step
        content = [SimpleNamespace(**b) for b in blocks]
        return SimpleNamespace(content=content, stop_reason="tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn")
