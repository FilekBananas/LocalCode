"""End-to-end tests of the coding agent with real local models.

A throwaway git repository with a deliberately broken ``calc.py`` is created in
a temp directory, the agent is asked — in Polish, like a user would —

    "Fix the failing test in this project. Run the tests to verify."

and the test passes only when ``pytest`` really is green afterwards. Nothing is
mocked: a real ``llama-server`` (or MLX worker) is started by ``RuntimeManager``,
the real tools read, edit and run commands in the repository, and the real
context manager fits everything into the window.

Three runs, one model in memory at a time:

* GGUF ``Qwen3-4B-Instruct-2507-Q4_K_M`` at ``n_ctx`` 8192,
* the same model at ``n_ctx`` 4096, which forces compaction mid-task,
* MLX ``Qwen3-4B-Instruct-2507-4bit`` at ``n_ctx`` 8192.

These are ``@pytest.mark.real_model`` (skipped without the models) and slow —
budget a few minutes each. On failure the whole transcript is printed.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from lc_testing import model_path
from localcode.agent.loop import Agent
from localcode.agent.session import SessionStore
from localcode.config import Settings
from localcode.hub.metadata import read_gguf_meta, read_mlx_meta
from localcode.hub.types import LocalModel
from localcode.runtime.manager import RuntimeManager

pytestmark = pytest.mark.real_model

VENV_BIN = str(Path(sys.executable).parent)
TASK = "Fix the failing test in this project. Run the tests to verify."

BUGGY_CALC = '''"""Tiny calculator used by the test suite."""


def add(a, b):
    """Return the sum of a and b."""
    return a - b


def multiply(a, b):
    """Return the product of a and b."""
    return a * b
'''

TEST_CALC = '''from calc import add, multiply


def test_add():
    assert add(2, 3) == 5


def test_multiply():
    assert multiply(2, 3) == 6
'''


# --------------------------------------------------------------------------- #
# Project under test
# --------------------------------------------------------------------------- #


def make_project(root: Path) -> Path:
    """A git repository with one bug: ``add`` subtracts instead of adding."""
    project = root / "kalkulator"
    project.mkdir(parents=True, exist_ok=True)
    (project / "calc.py").write_text(BUGGY_CALC, encoding="utf-8")
    (project / "test_calc.py").write_text(TEST_CALC, encoding="utf-8")
    (project / "README.md").write_text("# Kalkulator\n\nUruchom testy: `pytest -q`\n", encoding="utf-8")
    env = {**os.environ, "GIT_AUTHOR_NAME": "LocalCode", "GIT_AUTHOR_EMAIL": "test@localcode",
           "GIT_COMMITTER_NAME": "LocalCode", "GIT_COMMITTER_EMAIL": "test@localcode"}
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "initial"]):
        subprocess.run(["git", *args], cwd=project, env=env, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return project


def run_pytest(project: Path) -> subprocess.CompletedProcess[str]:
    """Run the project's tests exactly as the assertion needs them."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=project, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )


def local_model(key: str, path: Path, fmt: str) -> LocalModel:
    meta = read_gguf_meta(path) if fmt == "gguf" else read_mlx_meta(path)
    size = path.stat().st_size if fmt == "gguf" else sum(
        f.stat().st_size for f in path.glob("*.safetensors"))
    return LocalModel(key=key, format=fmt, path=str(path), display_name=key.split("/")[-1],
                      size_bytes=size, quant=meta.quant, meta=meta, capabilities=["tools"])


class OneModelLibrary:
    """``ModelLibrary`` surface the runtime manager needs, for exactly one model."""

    def __init__(self, model: LocalModel) -> None:
        self.model = model

    def get(self, key: str) -> LocalModel | None:
        return self.model if key == self.model.key else None

    def load_defaults(self, key: str) -> dict[str, Any] | None:
        return None


EARLIER_TOPICS = [
    ("Opisz, co robi ten projekt i jakie ma zależności.", "README.md",
     "# Kalkulator\n\nUruchom testy: `pytest -q`\n" + "\n".join(
         f"Sekcja {i}: ten projekt jest minimalnym przykładem używanym w testach LocalCode, "
         f"zawiera moduł calc.py z podstawowymi operacjami arytmetycznymi oraz test_calc.py." for i in range(12))),
    ("A jakie funkcje są w calc.py?", "calc.py",
     "\n".join(f"{i:4d}  def helper_{i}(value):  # pomocnicza funkcja numer {i}, "
               f"zwraca wartość przemnożoną przez {i} i zapisuje ją w historii obliczeń"
               for i in range(1, 40))),
    ("Sprawdź jeszcze historię gita.", "git log --stat -20",
     "\n".join(f"commit {i:040x}\nAuthor: LocalCode <test@localcode>\n\n    zmiana numer {i} "
               f"w module obliczeń, poprawki formatowania i dokumentacji" for i in range(1, 16))),
]


def seed_earlier_work(store: SessionStore) -> None:
    """Store a long earlier conversation about this project (≈4k tokens).

    Resuming it makes the next turn start over the context budget, so the
    compaction pipeline runs for real before the first model step.
    """
    for index, (question, target, output) in enumerate(EARLIER_TOPICS):
        call_id = f"seed{index:04d}"
        tool = "bash" if " " in target else "read_file"
        args = {"command": target} if tool == "bash" else {"path": target}
        store.append_messages([
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"Sprawdzam {target}.",
             "tool_calls": [{"id": call_id, "type": "function",
                             "function": {"name": tool, "arguments": args}}]},
            {"role": "tool", "tool_call_id": call_id, "name": tool, "content": output},
            {"role": "assistant", "content":
                f"Podsumowanie {target}: " + " ".join(
                    f"punkt {i} opisuje szczegóły implementacji i konwencje projektu," for i in range(1, 18))},
        ])


# --------------------------------------------------------------------------- #
# Transcript
# --------------------------------------------------------------------------- #


@dataclass
class Transcript:
    events: list[Any] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def __call__(self, event: Any) -> None:
        self.events.append(event)
        kind = event.type
        if kind == "tool_call":
            self.lines.append(f"→ {event.title}  {event.arguments}")
        elif kind == "tool_result":
            head = " ".join(event.output.split())[:200]
            self.lines.append(f"  {'ok' if event.ok else 'FAIL'} [{event.summary}] {head}")
        elif kind == "text_delta":
            if self.lines and self.lines[-1].startswith("💬"):
                self.lines[-1] += event.text
            else:
                self.lines.append("💬 " + event.text)
        elif kind == "compaction":
            self.lines.append(f"⇲ compaction {event.strategy}: {event.before} → {event.after}")
        elif kind == "context":
            self.lines.append(f"· context {event.used}/{event.limit}")
        elif kind == "status":
            self.lines.append(f"· {event.message}")
        elif kind == "error":
            self.lines.append(f"✗ {event.message}")
        elif kind == "turn_end":
            self.lines.append(f"■ turn_end {event.reason} {event.stats}")

    def render(self) -> str:
        return "\n".join(self.lines)

    @property
    def steps(self) -> int:
        end = [e for e in self.events if e.type == "turn_end"]
        return end[-1].stats.get("steps", 0) if end else 0

    @property
    def compactions(self) -> int:
        return sum(1 for e in self.events if e.type == "compaction")

    @property
    def strategies(self) -> list[str]:
        return [e.strategy for e in self.events if e.type == "compaction"]

    @property
    def peak_context(self) -> int:
        return max((e.used for e in self.events if e.type == "context"), default=0)

    @property
    def tool_calls(self) -> list[str]:
        return [e.name for e in self.events if e.type == "tool_call"]


@dataclass
class Outcome:
    label: str
    passed: bool
    steps: int
    seconds: float
    reason: str
    transcript: Transcript

    def summary(self) -> str:
        return (f"{self.label}: {'PASS' if self.passed else 'FAIL'} "
                f"({self.reason}, {self.steps} kroków, {self.seconds:.1f}s, "
                f"kompaktowań: {self.transcript.compactions}{self.transcript.strategies or ''}, "
                f"szczyt kontekstu: {self.transcript.peak_context}, "
                f"narzędzia: {', '.join(self.transcript.tool_calls)})")


# --------------------------------------------------------------------------- #
# The run
# --------------------------------------------------------------------------- #


async def run_agent(project: Path, model: LocalModel, n_ctx: int, *, label: str,
                    max_steps: int = 16, timeout: float = 900.0,
                    session: SessionStore | None = None) -> Outcome:
    settings = Settings()
    settings.runtime.default_ctx = n_ctx
    settings.runtime.max_loaded_models = 1
    settings.runtime.idle_ttl_minutes = 0
    settings.agent.permission_mode = "yolo"
    settings.agent.max_steps = max_steps
    settings.agent.temperature = 0.2
    settings.agent.thinking = False

    transcript = Transcript()
    runtime = RuntimeManager(settings, OneModelLibrary(model))
    store = session or SessionStore.create(project, model_key=model.key, mode="yolo")
    agent = Agent(runtime, model.key, project, settings, transcript, None, session=store)
    started = time.monotonic()
    try:
        await asyncio.wait_for(agent.run_turn(TASK), timeout=timeout)
    except asyncio.TimeoutError:
        agent.interrupt()
        await asyncio.sleep(0.5)
    finally:
        await runtime.shutdown()
    elapsed = time.monotonic() - started
    result = run_pytest(project)
    ends = [e for e in transcript.events if e.type == "turn_end"]
    return Outcome(
        label=label,
        passed=result.returncode == 0,
        steps=transcript.steps,
        seconds=elapsed,
        reason=(ends[-1].reason if ends else "no turn_end") + (
            "" if result.returncode == 0 else f" / pytest:\n{result.stdout[-1500:]}"),
        transcript=transcript,
    )


def check(outcome: Outcome) -> None:
    print(f"\n=== {outcome.summary()} ===")
    if not outcome.passed or not outcome.reason.startswith("done"):
        print(outcome.transcript.render())
    assert outcome.passed, outcome.summary()


@pytest.fixture(autouse=True)
def tools_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """The agent's shell must find the same ``python``/``pytest`` the test uses."""
    monkeypatch.setenv("PATH", VENV_BIN + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")


@pytest.fixture
def project(tmp_path: Path, isolated_home: Path) -> Path:
    """A fresh broken project; sessions land in the isolated LOCALCODE_HOME."""
    project = make_project(tmp_path)
    result = run_pytest(project)
    assert result.returncode != 0, "the fixture project must start with a failing test"
    return project


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


async def test_gguf_small_fixes_the_failing_test(project: Path) -> None:
    """The main end-to-end run: GGUF Qwen3-4B with a normal 8k window."""
    model = local_model("unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M",
                        model_path("gguf_small"), "gguf")
    check(await run_agent(project, model, 8192, label="gguf_small @ 8192"))


async def test_gguf_small_fixes_the_failing_test_in_a_4k_window(project: Path) -> None:
    """Half the window *and* a session resumed from an earlier long conversation.

    The stored history alone is bigger than the 4k window, so the very first
    step has to compact it for real (mask → shrink → the model summarising its
    own transcript) before the agent can even start working — and it still has
    to finish the task.
    """
    model = local_model("unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M",
                        model_path("gguf_small"), "gguf")
    store = SessionStore.create(project, model_key=model.key, mode="yolo")
    seed_earlier_work(store)
    outcome = await run_agent(project, model, 4096, label="gguf_small @ 4096 (resumed)",
                              max_steps=20, session=store)
    check(outcome)
    assert outcome.transcript.compactions, "a resumed long session in a 4k window must compact"


async def test_mlx_small_fixes_the_failing_test(project: Path) -> None:
    """Same task, MLX engine — proves the loop is engine-agnostic."""
    pytest.importorskip("mlx_lm", reason="mlx-lm is not installed")
    pytest.importorskip("localcode.runtime.mlx_engine", reason="the MLX engine does not exist yet")
    model = local_model("mlx-community/Qwen3-4B-Instruct-2507-4bit", model_path("mlx_small"), "mlx")
    check(await run_agent(project, model, 8192, label="mlx_small @ 8192"))
