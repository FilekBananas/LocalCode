"""Tests for ``localcode.runtime.mlx_engine``.

Unit tests cover everything that does not need a GPU: the worker command line
(per-slot context, KV quantisation, memory limit, interpreter, extra args),
the child environment, configuration validation, the Polish failure hints and
memory reporting.  ``real_model`` tests run the real worker with the tiny
Qwen3-0.6B MLX model (≈0.35 GB): worker extras in ``/props``, the ``/memory``
endpoint, a quantised KV cache and the orphan watchdog (a ``SIGKILL``-ed
parent must not leave a worker behind).

Behaviour shared with the GGUF engine (streaming, stats, prefix cache,
cancellation, overflow…) is tested once for both engines in
``test_engine_parity.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import psutil
import pytest
import pytest_asyncio

from lc_testing import model_path
from localcode.config import RuntimeSettings, Settings
from localcode.hub.types import LocalModel, ModelMeta
from localcode.runtime.base import EngineError, LoadConfig, SamplingParams
from localcode.runtime.http_engine import EngineStartError, HttpEngine
from localcode.runtime.mlx_engine import (
    WORKER_MODULE,
    MlxEngine,
    kv_bits_for,
    mlx_source_root,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN_META = ModelMeta(arch="qwen3", n_layers=28, n_heads=16, n_kv_heads=8, head_dim=128, context_length=40960)

try:  # the worker needs mlx-lm; unit tests do not
    import mlx_lm  # noqa: F401

    HAVE_MLX = True
except Exception:  # noqa: BLE001 - not on Apple Silicon / not installed
    HAVE_MLX = False

requires_mlx = pytest.mark.skipif(not HAVE_MLX, reason="mlx-lm not installed")
# real-model tests share one event loop with the module-scoped engine fixture
module_loop = pytest.mark.asyncio(loop_scope="module")


def _model_dir(tmp_path: Path, *, config: bool = True) -> Path:
    d = tmp_path / "mlx-model"
    d.mkdir(exist_ok=True)
    if config:
        (d / "config.json").write_text('{"model_type": "qwen3"}')
        (d / "model.safetensors").write_bytes(b"0" * 2048)
    return d


def _model(path: Path | str, *, fmt: str = "mlx", size: int = 1000) -> LocalModel:
    return LocalModel(key="mlx-community/Test", format=fmt, path=str(path), display_name="Test MLX",
                      size_bytes=size, meta=QWEN_META)


def _engine(tmp_path: Path, config: LoadConfig | None = None, *, settings: Settings | None = None,
            model: LocalModel | None = None, **kw) -> MlxEngine:
    return MlxEngine(
        "inst-1",
        model or _model(_model_dir(tmp_path)),
        config or LoadConfig(n_ctx=4096),
        settings=settings if settings is not None else Settings(),
        **kw,
    )


def _arg(cmd: list[str], flag: str) -> str | None:
    return cmd[cmd.index(flag) + 1] if flag in cmd else None


# --------------------------------------------------------------------------- #
# KV cache mapping
# --------------------------------------------------------------------------- #


def test_kv_bits_for_accepts_both_vocabularies():
    assert kv_bits_for("8bit") == 8
    assert kv_bits_for("q8_0") == 8
    assert kv_bits_for("4bit") == 4
    assert kv_bits_for("Q4_0") == 4
    assert kv_bits_for("f16") is None
    assert kv_bits_for("bf16") is None
    assert kv_bits_for(None) is None
    assert kv_bits_for("nonsense") is None


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_command_defaults(tmp_path):
    e = _engine(tmp_path, LoadConfig(n_ctx=8192, n_parallel=3))
    cmd = e._build_command(51234)
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", WORKER_MODULE]
    assert _arg(cmd, "--model") == str(_model_dir(tmp_path))
    assert _arg(cmd, "--host") == "127.0.0.1"
    assert _arg(cmd, "--port") == "51234"
    # --ctx is per slot (unlike llama.cpp's -c, which is the total)
    assert _arg(cmd, "--ctx") == "8192" and _arg(cmd, "--parallel") == "3"
    assert "--kv-bits" not in cmd and "--memory-limit-mb" not in cmd
    assert _arg(cmd, "--parent-pid") == str(os.getpid())
    assert e.command == cmd and e.format == "mlx"


def test_command_expands_user_path_and_clamps_parallel(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    d = _model_dir(tmp_path)
    e = _engine(tmp_path, LoadConfig(n_ctx=512, n_parallel=0), model=_model("~/mlx-model"))
    cmd = e._build_command(1234)
    assert _arg(cmd, "--model") == str(d)
    assert _arg(cmd, "--parallel") == "1"


def test_command_caps_context_at_the_trained_window(tmp_path):
    # the model was trained for 40960 tokens (QWEN_META)
    e = _engine(tmp_path, LoadConfig(n_ctx=131072))
    assert _arg(e._build_command(1), "--ctx") == "40960"
    e = _engine(tmp_path, LoadConfig(n_ctx=2048))
    assert _arg(e._build_command(1), "--ctx") == "2048"
    # an unknown training context is not second-guessed
    e = _engine(tmp_path, LoadConfig(n_ctx=131072), model=_model(_model_dir(tmp_path)))
    e.model.meta.context_length = None
    assert _arg(e._build_command(1), "--ctx") == "131072"


def test_command_kv_bits(tmp_path):
    for kv, expected in (("8bit", "8"), ("q8_0", "8"), ("4bit", "4"), ("q4_0", "4")):
        e = _engine(tmp_path, LoadConfig(n_ctx=1024, kv_cache_type=kv))
        assert _arg(e._build_command(1), "--kv-bits") == expected
        assert e.kv_bits == int(expected)
    e = _engine(tmp_path, LoadConfig(n_ctx=1024, kv_cache_type="f16"))
    assert "--kv-bits" not in e._build_command(1) and e.kv_bits is None


def test_command_memory_limit_from_settings_and_argument(tmp_path):
    e = _engine(tmp_path)
    assert e.memory_limit_mb is None and "--memory-limit-mb" not in e._build_command(1)

    s = Settings(runtime=RuntimeSettings(ram_limit_gb=12))
    e = _engine(tmp_path, settings=s)
    assert _arg(e._build_command(1), "--memory-limit-mb") == str(12 * 1024)

    e = _engine(tmp_path, settings=s, memory_limit_mb=4096)   # explicit wins
    assert _arg(e._build_command(1), "--memory-limit-mb") == "4096"

    e = _engine(tmp_path, memory_limit_mb=1)                  # worker's minimum
    assert _arg(e._build_command(1), "--memory-limit-mb") == "256"


def test_command_uses_configured_python_and_extra_args_last(tmp_path):
    other = tmp_path / "python3"
    other.write_text("#!/bin/sh\n")
    s = Settings(runtime=RuntimeSettings(python_path=str(other)))
    e = _engine(tmp_path, LoadConfig(n_ctx=1024, extra_args=["--ctx", "2048", "--trust-remote-code"]), settings=s)
    cmd = e._build_command(1)
    assert cmd[0] == str(other) and e.python_path == str(other)
    # extra args come last, so argparse lets the user override our defaults
    assert cmd[-3:] == ["--ctx", "2048", "--trust-remote-code"]

    e = _engine(tmp_path, python_path="/usr/bin/python3")      # constructor wins over settings
    assert e._build_command(1)[0] == "/usr/bin/python3"


def test_env_exposes_localcode_and_unbuffered_output(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    env = _engine(tmp_path)._env()
    assert env["PYTHONUNBUFFERED"] == "1"
    assert env["PYTHONPATH"].split(os.pathsep) == [str(REPO_ROOT), "/somewhere/else"]
    assert mlx_source_root() == str(REPO_ROOT)
    monkeypatch.delenv("PYTHONPATH")
    assert _engine(tmp_path)._env()["PYTHONPATH"] == str(REPO_ROOT)


def test_log_name_and_startup_timeout(tmp_path):
    e = _engine(tmp_path)
    assert e._log_name() == "mlx-inst-1"
    # size_bytes is authoritative; without it the weight files are measured
    e.model.size_bytes = 4_000_000_000
    assert e._startup_timeout() == pytest.approx(e.startup_timeout_base_s + 4 * e.startup_timeout_per_gb_s)
    e.model.size_bytes = 0
    (Path(e.model.path) / "model-00002.safetensors").write_bytes(b"0" * 1_000_000)
    assert e._startup_timeout() == pytest.approx(
        e.startup_timeout_base_s + e.startup_timeout_per_gb_s * (1_002_048 / 1e9)
    )
    e.model.path = "/does/not/exist"
    assert e._startup_timeout() == e.startup_timeout_base_s


# --------------------------------------------------------------------------- #
# Validation and failure explanations
# --------------------------------------------------------------------------- #


async def test_before_spawn_validation(tmp_path):
    e = _engine(tmp_path, model=_model(_model_dir(tmp_path), fmt="gguf"))
    with pytest.raises(EngineStartError, match="nie jest w formacie MLX"):
        await e._before_spawn()

    e = _engine(tmp_path, model=_model(tmp_path / "missing"))
    with pytest.raises(EngineStartError, match="Katalog modelu MLX nie istnieje"):
        await e._before_spawn()

    empty = tmp_path / "empty"
    empty.mkdir()
    e = _engine(tmp_path, model=_model(empty))
    with pytest.raises(EngineStartError, match="brakuje pliku config.json"):
        await e._before_spawn()

    s = Settings(runtime=RuntimeSettings(python_path=str(tmp_path / "no-python")))
    e = _engine(tmp_path, settings=s)
    with pytest.raises(EngineStartError, match="Nie znaleziono interpretera Pythona"):
        await e._before_spawn()

    e = _engine(tmp_path, LoadConfig(n_ctx=8))
    with pytest.raises(EngineStartError, match="co najmniej 16 tokenów"):
        await e._before_spawn()

    e = _engine(tmp_path, LoadConfig(n_ctx=2048, embedding=True))
    with pytest.raises(EngineStartError, match="nie obsługuje trybu embeddingów"):
        await e._before_spawn()

    await _engine(tmp_path)._before_spawn()  # a valid setup raises nothing


def test_failure_hints(tmp_path):
    e = _engine(tmp_path)
    hint = e._failure_hint("Traceback ...\nModuleNotFoundError: No module named 'mlx_lm'", 1)
    assert hint and "mlx-lm" in hint and sys.executable in hint
    hint = e._failure_hint("ModuleNotFoundError: No module named 'localcode.runtime'", 1)
    assert hint and "localcode" in hint
    hint = e._failure_hint("No module named 'sentencepiece'", 1)
    assert hint and "sentencepiece" in hint
    hint = e._failure_hint("ERROR mlx_worker: Błąd konfiguracji: Nieprawidłowy port: 99999", 2)
    assert hint == "Worker MLX odrzucił parametry uruchomienia: Nieprawidłowy port: 99999"
    hint = e._failure_hint("ERROR mlx_worker: Błąd ładowania modelu: Za mało pamięci, aby załadować model MLX", 1)
    assert hint == "Za mało pamięci, aby załadować model MLX"
    hint = e._failure_hint("ValueError: Model type mamba2 not supported.", 1)
    assert hint and "mlx-lm" in hint
    assert e._failure_hint("some unrelated log line", 1) is None
    # the generic hints of the base class still apply
    assert "pamięci" in e._failure_message("failed to allocate buffer: out of memory", 1)


async def test_embeddings_are_rejected_without_a_round_trip(tmp_path):
    e = _engine(tmp_path)
    with pytest.raises(EngineError, match="nie obsługuje embeddingów"):
        await e.embed(["hello"])


def _rss(value: int | None):
    async def memory_bytes(self):
        return value

    return memory_bytes


async def test_memory_bytes_combines_worker_stats_and_rss(tmp_path, monkeypatch):
    e = _engine(tmp_path)
    stats: object = {"active": 5_000_000_000, "cache": 1_000_000_000, "peak": 6_000_000_000}

    async def fake_request(method, path, body=None, *, timeout=None):
        assert (method, path) == ("GET", "/memory")
        if isinstance(stats, Exception):
            raise stats
        return stats

    monkeypatch.setattr(e, "_request_json", fake_request)
    monkeypatch.setattr(HttpEngine, "memory_bytes", _rss(2_000_000_000))
    assert await e.memory_bytes() == 6_000_000_000        # MLX knows more than RSS
    monkeypatch.setattr(HttpEngine, "memory_bytes", _rss(9_000_000_000))
    assert await e.memory_bytes() == 9_000_000_000        # RSS knows more than MLX
    assert await e.worker_memory() == stats

    stats = EngineError("nie działa")                     # /memory unavailable -> RSS
    assert await e.memory_bytes() == 9_000_000_000
    monkeypatch.setattr(HttpEngine, "memory_bytes", _rss(None))
    assert await e.memory_bytes() is None


# --------------------------------------------------------------------------- #
# Real worker with the tiny MLX model
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def lc_home(tmp_path_factory):
    home = tmp_path_factory.mktemp("lc-home")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("LOCALCODE_HOME", str(home))
        yield home


def _tiny_model() -> LocalModel:
    p = model_path("mlx_tiny")
    size = sum(f.stat().st_size for f in p.iterdir() if f.is_file())
    return LocalModel(key="mlx-community/Qwen3-0.6B-4bit", format="mlx", path=str(p),
                      display_name="Qwen3 0.6B MLX", size_bytes=size, quant="4bit", meta=QWEN_META)


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def tiny(lc_home):
    if not HAVE_MLX:
        pytest.skip("mlx-lm not installed")
    engine = MlxEngine("mlx-tiny-1", _tiny_model(), LoadConfig(n_ctx=1024, n_parallel=2), settings=Settings())
    await engine.start()
    try:
        yield engine
    finally:
        await engine.stop()


@pytest.mark.real_model
@requires_mlx
@module_loop
async def test_real_worker_props_and_log(tiny, lc_home):
    assert tiny.running and tiny.pid and tiny.port
    props = tiny.last_props
    assert props["engine"] == "mlx" and props["batching"] is True and props["kv_bits"] is None
    assert props["model_path"] == str(model_path("mlx_tiny").resolve())
    assert props["total_slots"] == 2 and props["n_ctx"] == 1024
    assert "mlx_lm-" in props["build_info"]
    assert tiny.batching is True and tiny.worker_kv_bits is None
    assert (lc_home / "logs" / "mlx-mlx-tiny-1.log").exists()
    assert "Nasłuchuję na http" in tiny.log_tail(200)


@pytest.mark.real_model
@requires_mlx
@module_loop
async def test_real_memory_endpoint(tiny):
    stats = await tiny.worker_memory()
    assert stats["active"] > 200_000_000                 # the 4-bit weights are resident
    assert stats["peak"] >= stats["active"]
    assert stats["memory_limit"] is None
    assert stats["kv_bytes_per_token"] > 0
    assert stats["prompt_cache"]["budget"] > 0
    total = await tiny.memory_bytes()
    assert total is not None and total >= stats["active"]


@pytest.mark.real_model
@requires_mlx
@module_loop
async def test_real_thinking_switch_and_tool_format(tiny):
    info = tiny.template_info
    assert info.supports_tools and info.supports_tool_role and info.supports_system
    assert info.supports_thinking and info.thinking_kwarg == "enable_thinking"
    assert info.tool_format == "hermes" and info.stop_strings == ["<|im_end|>"]
    on = await tiny.apply_template([{"role": "user", "content": "hi"}], template_kwargs={"enable_thinking": True})
    off = await tiny.apply_template([{"role": "user", "content": "hi"}], template_kwargs={"enable_thinking": False})
    assert off.endswith("<think>\n\n</think>\n\n") and not on.endswith("</think>\n\n")


@pytest.mark.real_model
@requires_mlx
@module_loop
async def test_real_quantized_kv_cache(lc_home):
    engine = MlxEngine("mlx-kv8", _tiny_model(), LoadConfig(n_ctx=512, n_parallel=2, kv_cache_type="8bit"),
                       settings=Settings())
    await engine.start()
    pid = engine.pid
    try:
        assert _arg(engine.command, "--kv-bits") == "8"
        assert engine.worker_kv_bits == 8
        # mlx-lm cannot batch a quantised cache: the worker serves slot by slot
        assert engine.batching is False and engine.n_ctx == 512
        prompt = await engine.apply_template([{"role": "user", "content": "Say OK."}],
                                             template_kwargs={"enable_thinking": False})
        chunks = [c async for c in engine.complete(prompt, SamplingParams(max_tokens=8, temperature=0.0))]
        assert chunks[-1].done and chunks[-1].stats.completion_tokens > 0
    finally:
        await engine.stop()
    assert pid is not None and not _alive(pid)


@pytest.mark.real_model
@requires_mlx
def test_real_worker_exits_when_its_parent_is_killed(tmp_path):
    """A SIGKILL-ed core cannot run any handler — ``--parent-pid`` is what
    keeps a multi-GB worker from surviving it."""
    script = tmp_path / "parent.py"
    script.write_text(textwrap.dedent(f"""
        import asyncio, sys
        sys.path.insert(0, {str(REPO_ROOT)!r})
        from localcode.config import Settings
        from localcode.hub.types import LocalModel
        from localcode.runtime.base import LoadConfig
        from localcode.runtime.mlx_engine import MlxEngine

        async def main():
            model = LocalModel(key="k", format="mlx", path={str(model_path("mlx_tiny"))!r},
                               display_name="Qwen3 0.6B MLX", size_bytes=350_000_000)
            engine = MlxEngine("orphan-1", model, LoadConfig(n_ctx=256), settings=Settings())
            await engine.start()
            print(engine.pid, flush=True)
            await asyncio.sleep(300)

        asyncio.run(main())
    """))
    env = {**os.environ, "LOCALCODE_HOME": str(tmp_path / "home")}
    parent = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True, env=env)
    worker_pid: int | None = None
    try:
        line = parent.stdout.readline()
        assert line.strip(), f"parent did not start the engine (exit {parent.poll()})"
        worker_pid = int(line)
        assert _alive(worker_pid)
        parent.kill()
        parent.wait(timeout=10)
        deadline = time.monotonic() + 30
        while _alive(worker_pid) and time.monotonic() < deadline:
            time.sleep(0.2)
        assert not _alive(worker_pid), "worker survived its parent"
    finally:
        if worker_pid is not None and _alive(worker_pid):
            os.kill(worker_pid, 9)
        if parent.poll() is None:
            parent.kill()
        parent.wait(timeout=10)
        if parent.stdout is not None:
            parent.stdout.close()


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
