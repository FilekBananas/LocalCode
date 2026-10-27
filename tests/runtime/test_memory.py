"""Tests for runtime/memory.py: KV math, estimates, budgets, system probing.

Real-model tests start llama-server / an MLX process on the small test models
and check that the estimate is close to the measured memory.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from lc_testing import model_path
from localcode.config import RuntimeSettings, Settings
from localcode.hub.types import LocalModel, MemoryEstimate, ModelMeta
from localcode.runtime import memory as mem
from localcode.runtime.base import LoadConfig

MiB = mem.MiB
GiB = mem.GiB

QWEN3_06B = ModelMeta(arch="qwen3", context_length=40960, n_layers=28, n_heads=16, n_kv_heads=8,
                      head_dim=128, hidden_size=1024, vocab_size=151936)
QWEN3_4B = ModelMeta(arch="qwen3", context_length=262144, n_layers=36, n_heads=32, n_kv_heads=8,
                     head_dim=128, hidden_size=2560, vocab_size=151936)


def make_model(meta: ModelMeta = QWEN3_06B, *, fmt: str = "gguf", size: int = 400 * MiB,
               path: str = "/nonexistent/model.gguf", key: str = "org/repo/model",
               capabilities: list[str] | None = None, mmproj: str | None = None) -> LocalModel:
    return LocalModel(key=key, format=fmt, path=path, display_name=key.rsplit("/", 1)[-1],
                      size_bytes=size, meta=meta, capabilities=capabilities or [],
                      mmproj_path=mmproj)


@pytest.fixture
def fake_system(monkeypatch):
    """Deterministic 32 GiB machine with 20 GiB available."""
    state = {"total": 32 * GiB, "available": 20 * GiB}

    def fake() -> dict[str, int]:
        return {"total": state["total"], "available": state["available"],
                "used": state["total"] - state["available"]}

    monkeypatch.setattr(mem, "system_memory", fake)
    return state


# --------------------------------------------------------------------------- #
# KV math
# --------------------------------------------------------------------------- #


def test_kv_bytes_per_token_qwen3():
    assert mem.kv_bytes_per_token(QWEN3_06B, "f16", "gguf") == 2 * 28 * 8 * 128 * 2
    assert mem.kv_bytes_per_token(QWEN3_06B, "bf16", "gguf") == 114688
    assert mem.kv_bytes_per_token(QWEN3_06B, "q8_0", "gguf") == pytest.approx(114688 * 1.0625 / 2)
    assert mem.kv_bytes_per_token(QWEN3_06B, "q4_0", "gguf") == pytest.approx(114688 * 0.5625 / 2)
    assert mem.kv_bytes_per_token(QWEN3_06B, "8bit", "mlx") == pytest.approx(114688 * 1.0625 / 2)
    assert mem.kv_bytes_per_token(QWEN3_06B, "4bit", "mlx") == pytest.approx(114688 * 0.5625 / 2)
    # cross-format spellings are translated, unknown types count as f16
    assert mem.kv_bytes_per_token(QWEN3_06B, "q8_0", "mlx") == mem.kv_bytes_per_token(QWEN3_06B, "8bit", "mlx")
    assert mem.kv_bytes_per_token(QWEN3_06B, "weird", "gguf") == 114688


def test_head_dim_derivation():
    derived = ModelMeta(n_layers=32, n_heads=32, n_kv_heads=8, hidden_size=4096)
    assert mem.attention_layout(derived).head_dim == 128
    # an explicit head_dim wins over hidden/heads (Qwen3-0.6B: 1024/16 = 64, real 128)
    assert mem.attention_layout(QWEN3_06B).head_dim == 128
    # no kv heads -> assume MHA (conservative)
    mha = ModelMeta(n_layers=32, n_heads=32, hidden_size=4096)
    assert mem.attention_layout(mha).n_kv_heads == 32


def test_kv_cache_matches_llama_cpp_buffers():
    # values printed by llama-server b10964 for Qwen3-0.6B ("llama_kv_cache: size = …")
    assert mem.kv_cache_bytes(QWEN3_06B, 4096, 1, "f16", "gguf") == 448 * MiB
    assert mem.kv_cache_bytes(QWEN3_06B, 8192, 4, "f16", "gguf") == 3584 * MiB
    assert mem.kv_cache_bytes(QWEN3_06B, 16384, 1, "q8_0", "gguf") == 952 * MiB
    assert mem.kv_cache_bytes(QWEN3_4B, 4096, 2, "f16", "gguf") == 1152 * MiB


def test_gguf_kv_cells_are_padded_mlx_are_not():
    assert mem.kv_cache_bytes(QWEN3_06B, 1000, 1, "f16", "gguf") == 1024 * 114688
    assert mem.kv_cache_bytes(QWEN3_06B, 1000, 1, "f16", "mlx") == 1000 * 114688


def test_swa_known_architecture_pattern():
    gemma3 = ModelMeta(arch="gemma3", context_length=131072, n_layers=34, n_heads=8,
                       n_kv_heads=4, head_dim=256, sliding_window=1024)
    layout = mem.attention_layout(gemma3)
    assert (layout.full_layers, layout.swa_layers, layout.linear_layers) == (5, 29, 0)
    per_layer = 2 * 4 * 256 * 2
    assert mem.kv_bytes_per_token(gemma3) == 5 * per_layer
    ctx = 32768
    assert mem.kv_cache_bytes(gemma3, ctx) == per_layer * (5 * ctx + 29 * (1024 + 512))
    assert mem.kv_cache_bytes(gemma3, ctx, fmt="mlx") == per_layer * (5 * ctx + 29 * 1024)
    assert mem.kv_cache_bytes(gemma3, ctx, swa_full=True) == per_layer * 34 * ctx
    assert mem.kv_bytes_per_token(gemma3, swa_full=True) == 34 * per_layer
    # a context shorter than the window is not inflated
    assert mem.kv_cache_bytes(gemma3, 512, fmt="mlx") == per_layer * 34 * 512
    # MLX config spelling of the architecture
    assert mem.attention_layout(ModelMeta(arch="gemma3_text", n_layers=26, sliding_window=512)).full_layers == 4
    # --swa-full in extra args reaches the estimate
    model = make_model(gemma3)
    capped = mem.breakdown(model, LoadConfig(n_ctx=ctx)).kv_cache
    full = mem.breakdown(model, LoadConfig(n_ctx=ctx, extra_args=["--swa-full"])).kv_cache
    assert full > 4 * capped


def test_swa_with_explicit_full_layer_count():
    meta = ModelMeta(arch="whatever", n_layers=24, n_kv_heads=8, head_dim=64, sliding_window=4096,
                     n_full_attn_layers=6)
    layout = mem.attention_layout(meta)
    assert (layout.full_layers, layout.swa_layers) == (6, 18)
    per_layer = 2 * 8 * 64 * 2
    assert mem.kv_cache_bytes(meta, 65536, 2, fmt="mlx") == per_layer * (6 * 65536 * 2 + 18 * 4096 * 2)


def test_swa_unknown_architecture_is_conservative():
    mistral = ModelMeta(arch="llama", n_layers=32, n_kv_heads=8, head_dim=128, sliding_window=4096)
    layout = mem.attention_layout(mistral)
    assert layout.full_layers == 32 and layout.swa_layers == 0


def test_hybrid_linear_attention_layers_excluded():
    qwen_next = ModelMeta(arch="qwen3next", n_layers=48, n_heads=16, n_kv_heads=2, head_dim=256,
                          n_full_attn_layers=12, vocab_size=151936, context_length=262144)
    layout = mem.attention_layout(qwen_next)
    assert (layout.full_layers, layout.swa_layers, layout.linear_layers) == (12, 0, 36)
    assert mem.kv_bytes_per_token(qwen_next) == 2 * 12 * 2 * 256 * 2
    model = make_model(qwen_next, size=40 * GiB)
    dense = make_model(ModelMeta(**{**qwen_next.__dict__, "n_full_attn_layers": None}), size=40 * GiB)
    hybrid_parts = mem.breakdown(model, LoadConfig(n_ctx=32768))
    dense_parts = mem.breakdown(dense, LoadConfig(n_ctx=32768))
    assert hybrid_parts.kv_cache * 4 == dense_parts.kv_cache
    # constant recurrent state allowance for the 36 linear layers
    assert hybrid_parts.overhead - dense_parts.overhead == 36 * mem.RECURRENT_STATE_PER_LAYER


def test_encoder_embedding_models_have_no_kv_cache():
    nomic = ModelMeta(arch="nomic-bert", context_length=2048, n_layers=12, n_heads=12,
                      hidden_size=768, vocab_size=30522)
    model = make_model(nomic, size=139 * MiB, capabilities=["embedding"])
    parts = mem.breakdown(model, LoadConfig(n_ctx=2048, embedding=True))
    assert parts.kv_cache == 0
    assert 250 * MiB < parts.total < 350 * MiB
    # a decoder-based embedding model keeps its KV cache
    qwen_embed = make_model(QWEN3_06B, capabilities=["embedding"])
    assert mem.breakdown(qwen_embed, LoadConfig(n_ctx=2048, embedding=True)).kv_cache > 0


def test_unknown_metadata_uses_conservative_defaults():
    empty = ModelMeta()
    layout = mem.attention_layout(empty)
    assert (layout.n_layers, layout.n_kv_heads, layout.head_dim) == (36, 8, 128)
    assert mem.attention_layout(ModelMeta(params_b=70)).n_layers == 80
    assert mem.attention_layout(ModelMeta(), weights_bytes=int(0.4e9)).n_layers == 28
    model = make_model(empty)
    assert mem.native_context(model) == mem.DEFAULT_NATIVE_CTX
    est = mem.estimate(model, LoadConfig(n_ctx=4096))
    assert est.total_bytes > est.weights_bytes > 0


def test_normalize_kv_type():
    assert mem.normalize_kv_type("q8_0", "mlx") == "8bit"
    assert mem.normalize_kv_type("q4_0", "mlx") == "4bit"
    assert mem.normalize_kv_type("bf16", "mlx") == "f16"
    assert mem.normalize_kv_type("8bit", "gguf") == "q8_0"
    assert mem.normalize_kv_type("4bit", "gguf") == "q4_0"
    assert mem.normalize_kv_type("BF16", "gguf") == "bf16"
    assert mem.normalize_kv_type(None, "gguf") == "f16"
    assert mem.normalize_kv_type("nonsense", "mlx") == "f16"


# --------------------------------------------------------------------------- #
# Weights and overhead
# --------------------------------------------------------------------------- #


def _write(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.truncate(size)


def test_weights_from_disk(tmp_path):
    single = tmp_path / "a" / "model-Q4_K_M.gguf"
    _write(single, 12345)
    assert mem.model_weights_bytes(make_model(path=str(single), size=1)) == 12345

    shard1 = tmp_path / "b" / "big-Q4_K_M-00001-of-00003.gguf"
    _write(shard1, 1000)
    _write(tmp_path / "b" / "big-Q4_K_M-00002-of-00003.gguf", 2000)
    _write(tmp_path / "b" / "big-Q4_K_M-00003-of-00003.gguf", 3000)
    _write(tmp_path / "b" / "mmproj-F16.gguf", 999)
    assert mem.model_weights_bytes(make_model(path=str(shard1), size=1)) == 6000

    mlx_dir = tmp_path / "mlx"
    _write(mlx_dir / "model-00001-of-00002.safetensors", 700)
    _write(mlx_dir / "model-00002-of-00002.safetensors", 300)
    _write(mlx_dir / "tokenizer.json", 5000)
    assert mem.model_weights_bytes(make_model(fmt="mlx", path=str(mlx_dir), size=1)) == 1000

    assert mem.model_weights_bytes(make_model(path=str(tmp_path / "missing.gguf"), size=4242)) == 4242


def test_mmproj_counts_for_gguf(tmp_path):
    proj = tmp_path / "mmproj.gguf"
    _write(proj, 600 * MiB)
    plain = make_model()
    vision = make_model(mmproj=str(proj))
    delta = mem.breakdown(vision, LoadConfig(n_ctx=4096)).total - mem.breakdown(plain, LoadConfig(n_ctx=4096)).total
    assert delta == 600 * MiB + mem.VISION_COMPUTE_BYTES


def test_overhead_depends_on_flash_attention_and_extra_args():
    model = make_model()
    fa = mem.breakdown(model, LoadConfig(n_ctx=32768, flash_attn="auto"))
    no_fa = mem.breakdown(model, LoadConfig(n_ctx=32768, flash_attn="off"))
    # measured: ~226 MiB of compute buffers with FA vs ~1180 MiB without
    assert 150 * MiB < fa.overhead - mem.GGUF_PROCESS_BASE < 300 * MiB
    assert 1000 * MiB < no_fa.overhead - mem.GGUF_PROCESS_BASE < 1300 * MiB
    cram = mem.breakdown(model, LoadConfig(n_ctx=32768, extra_args=["--cache-ram", "512"]))
    assert cram.overhead - fa.overhead == 512 * MiB


def test_mlx_quantized_kv_adds_attention_scores():
    model = make_model(fmt="mlx", path="/nonexistent/mlx", size=320 * MiB)
    f16 = mem.breakdown(model, LoadConfig(n_ctx=16384, kv_cache_type="f16"))
    q8 = mem.breakdown(model, LoadConfig(n_ctx=16384, kv_cache_type="8bit"))
    assert q8.kv_cache < f16.kv_cache
    assert q8.overhead - f16.overhead == pytest.approx(16 * 2048 * 16384 * mem.MLX_SCORE_BYTES, rel=1e-6)
    short = mem.breakdown(model, LoadConfig(n_ctx=1024, kv_cache_type="8bit"))
    assert short.total < mem.breakdown(model, LoadConfig(n_ctx=1024)).total + 100 * MiB


# --------------------------------------------------------------------------- #
# Estimate verdicts and messages
# --------------------------------------------------------------------------- #


def test_estimate_fits_yes_with_polish_message(fake_system):
    model = make_model(QWEN3_4B, size=int(2.4 * GiB))
    est = mem.estimate(model, LoadConfig(n_ctx=8192), budget=16 * GiB)
    assert isinstance(est, MemoryEstimate)
    assert est.fits == "yes"
    assert est.total_bytes == est.weights_bytes + est.kv_cache_bytes + est.overhead_bytes
    assert (est.n_ctx, est.n_parallel) == (8192, 1)
    assert re.fullmatch(r"Zmieści się: ~\d+,\d GB \(wagi \d+,\d GB \+ kontekst \d+,\d GB \+ bufory \d+,\d GB\)\.",
                        est.message), est.message
    assert est.to_dict()["fits"] == "yes"


def test_estimate_tight_and_no(fake_system):
    model = make_model(QWEN3_4B, size=int(2.4 * GiB))
    cfg = LoadConfig(n_ctx=8192)
    total = mem.breakdown(model, cfg).total
    # nearly the whole remaining budget
    tight = mem.estimate(model, cfg, budget=int(total / 0.95))
    assert tight.fits == "tight" and tight.message.startswith("Na styk")
    # fits the budget but not the memory available right now
    fake_system["available"] = total - 100 * MiB
    busy = mem.estimate(model, cfg, budget=16 * GiB)
    assert busy.fits == "tight" and "Wolne jest teraz tylko" in busy.message
    assert mem.estimate(model, cfg, budget=16 * GiB, reclaimable=GiB).fits == "yes"
    fake_system["available"] = 20 * GiB
    # other instances take the budget
    no = mem.estimate(model, cfg, budget=8 * GiB, in_use=7 * GiB)
    assert no.fits == "no"
    assert no.message.startswith("Nie zmieści się")
    assert "inne modele" in no.message
    # bigger than the machine
    huge = mem.estimate(make_model(QWEN3_4B, size=40 * GiB), cfg, budget=64 * GiB)
    assert huge.fits == "no" and "ten Mac ma 32,0 GB" in huge.message


def test_estimate_no_suggests_smaller_context(fake_system):
    model = make_model(QWEN3_4B, size=int(2.4 * GiB))
    est = mem.estimate(model, LoadConfig(n_ctx=131072, n_parallel=2), budget=8 * GiB)
    assert est.fits == "no"
    match = re.search(r"Zmniejsz kontekst do ([\d ]+) tokenów", est.message)
    assert match, est.message
    suggested = int(match.group(1).replace(" ", ""))
    assert suggested % 1024 == 0
    assert mem.breakdown(model, LoadConfig(n_ctx=suggested, n_parallel=2)).total <= 8 * GiB
    assert "równoległych" in est.message
    # the weights alone do not fit -> different advice
    small_budget = mem.estimate(model, LoadConfig(n_ctx=4096), budget=2 * GiB)
    assert "mniejszy model" in small_budget.message


def test_format_helpers():
    assert mem.format_gb(5.23 * GiB) == "5,2 GB"
    assert mem.format_gb(0.6 * GiB) == "0,6 GB"
    assert mem.format_gb(0.01 * GiB) == "<0,1 GB"
    assert mem.format_gb(0) == "0,0 GB"
    assert mem.format_gb(150 * GiB) == "150 GB"
    assert mem.format_tokens(16384) == "16 384"
    assert mem.format_tokens(512) == "512"


# --------------------------------------------------------------------------- #
# Context sizing
# --------------------------------------------------------------------------- #


def test_max_ctx_that_fits_properties():
    model = make_model(QWEN3_4B, size=int(2.4 * GiB))
    budget = 5 * GiB
    ctx = mem.max_ctx_that_fits(model, budget, 1, "f16")
    assert ctx > 0 and ctx % 1024 == 0
    assert mem.breakdown(model, LoadConfig(n_ctx=ctx)).total <= budget
    assert mem.breakdown(model, LoadConfig(n_ctx=ctx + 1024)).total > budget
    # more parallel slots or a bigger KV type -> less context per slot
    assert mem.max_ctx_that_fits(model, budget, 2, "f16") < ctx
    assert mem.max_ctx_that_fits(model, budget, 1, "q8_0") > ctx
    # capped by the native context, returned as-is when everything fits
    assert mem.max_ctx_that_fits(model, 1000 * GiB, 1, "f16") == 262144
    odd = make_model(ModelMeta(**{**QWEN3_06B.__dict__, "context_length": 32000}))
    assert mem.max_ctx_that_fits(odd, 1000 * GiB) == 32000
    # nothing fits
    assert mem.max_ctx_that_fits(model, GiB, 1, "f16") == 0


def test_auto_ctx_rules():
    tiny = make_model(QWEN3_06B)
    big_budget = 100 * GiB
    assert mem.auto_ctx(tiny, Settings(), big_budget) == 32768                    # default cap
    assert mem.auto_ctx(tiny, RuntimeSettings(default_ctx=8192), big_budget) == 8192
    assert mem.auto_ctx(tiny, RuntimeSettings(default_ctx=100_000), big_budget) == 40960   # native
    limited = mem.auto_ctx(tiny, Settings(), GiB)
    assert 2048 <= limited < 32768 and limited % 1024 == 0
    assert mem.breakdown(tiny, LoadConfig(n_ctx=limited)).total <= GiB
    assert mem.auto_ctx(tiny, Settings(), 10 * MiB) == 2048                       # floor
    embed = make_model(ModelMeta(arch="nomic-bert", context_length=512, n_layers=12, n_heads=12,
                                 hidden_size=768), capabilities=["embedding"])
    assert mem.auto_ctx(embed, Settings(), big_budget, embedding=True) == 512
    unknown = make_model(ModelMeta())
    assert mem.auto_ctx(unknown, Settings(), big_budget) == 8192
    # parallel slots share the budget
    two = mem.auto_ctx(tiny, RuntimeSettings(default_parallel=2), 2 * GiB)
    one = mem.auto_ctx(tiny, RuntimeSettings(default_parallel=1), 2 * GiB)
    assert two < one


def test_default_ram_budget(fake_system):
    assert mem.default_ram_budget(Settings()) == int(32 * GiB * 0.70)
    assert mem.default_ram_budget(RuntimeSettings(ram_limit_gb=8)) == 8 * GiB
    assert mem.default_ram_budget(RuntimeSettings(ram_limit_gb=12.5)) == int(12.5 * GiB)
    assert mem.default_ram_budget(RuntimeSettings(ram_limit_gb=500)) == 32 * GiB
    assert mem.default_ram_budget(RuntimeSettings(ram_limit_gb=0)) == int(32 * GiB * 0.70)


# --------------------------------------------------------------------------- #
# System / process memory
# --------------------------------------------------------------------------- #


def test_system_memory_is_consistent():
    import psutil

    info = mem.system_memory()
    assert set(info) == {"total", "available", "used"}
    assert info["total"] == psutil.virtual_memory().total
    assert info["used"] + info["available"] == info["total"]
    assert 0 < info["available"] < info["total"]


def _vm_stat_available() -> int:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page = int(re.search(r"page size of (\d+) bytes", out).group(1))
    val = {m.group(1): int(m.group(2)) for m in re.finditer(r'^"?([^:"]+)"?:\s+(\d+)\.', out, re.M)}
    app = val["Anonymous pages"] - val["Pages purgeable"]
    used = (app + val["Pages wired down"] + val["Pages occupied by compressor"]) * page
    return mem.system_memory()["total"] - used


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
def test_system_memory_matches_vm_stat():
    before = _vm_stat_available()
    ours = mem.system_memory()["available"]
    after = _vm_stat_available()
    slack = 512 * MiB
    assert min(before, after) - slack <= ours <= max(before, after) + slack


def test_process_memory_bytes():
    own = mem.process_memory_bytes(os.getpid())
    assert own is not None and own > 10 * MiB
    assert mem.process_memory_bytes(None) is None
    assert mem.process_memory_bytes(0) is None
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    assert mem.process_memory_bytes(proc.pid) is None


# --------------------------------------------------------------------------- #
# Real models: estimates vs. measurements
# --------------------------------------------------------------------------- #

LLAMA_SERVER = shutil.which("llama-server") or "/opt/homebrew/bin/llama-server"


def gguf_local_model(path: Path, *, embedding: bool = False) -> LocalModel:
    import gguf

    reader = gguf.GGUFReader(str(path))
    fields = reader.fields
    arch = fields["general.architecture"].contents()

    def get(name: str):
        f = fields.get(f"{arch}.{name}")
        return f.contents() if f is not None else None

    meta = ModelMeta(arch=arch, context_length=get("context_length"), n_layers=get("block_count"),
                     n_heads=get("attention.head_count"), n_kv_heads=get("attention.head_count_kv"),
                     head_dim=get("attention.key_length"), hidden_size=get("embedding_length"),
                     sliding_window=get("attention.sliding_window"),
                     vocab_size=len(fields["tokenizer.ggml.tokens"].data), is_embedding=embedding)
    return LocalModel(key=f"test/{path.stem}", format="gguf", path=str(path), display_name=path.stem,
                      size_bytes=path.stat().st_size, meta=meta,
                      capabilities=["embedding"] if embedding else [])


def mlx_local_model(path: Path) -> LocalModel:
    cfg = json.loads((path / "config.json").read_text())
    meta = ModelMeta(arch=cfg.get("model_type"), context_length=cfg.get("max_position_embeddings"),
                     n_layers=cfg.get("num_hidden_layers"), n_heads=cfg.get("num_attention_heads"),
                     n_kv_heads=cfg.get("num_key_value_heads"), head_dim=cfg.get("head_dim"),
                     hidden_size=cfg.get("hidden_size"), vocab_size=cfg.get("vocab_size"))
    size = sum(f.stat().st_size for f in path.glob("*.safetensors"))
    return LocalModel(key=f"test/{path.name}", format="mlx", path=str(path), display_name=path.name,
                      size_bytes=size, meta=meta)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(url: str, body: dict | None = None, timeout: float = 5.0) -> tuple[int, bytes]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, resp.read()


def _run_llama(model: LocalModel, cfg: LoadConfig, log_path: Path) -> tuple[int, str]:
    """Start llama-server, touch the buffers with one request, return (measured bytes, log)."""
    if not os.path.exists(LLAMA_SERVER):
        pytest.skip("llama-server not installed")
    port = _free_port()
    args = [LLAMA_SERVER, "-m", model.path, "--host", "127.0.0.1", "--port", str(port),
            "-c", str(cfg.n_ctx * cfg.n_parallel), "-np", str(cfg.n_parallel), "-ngl", "99",
            "-fa", cfg.flash_attn, "--no-webui", "-lv", "4"]
    if cfg.embedding:
        args.append("--embeddings")
    with open(log_path, "w") as fh:
        proc = subprocess.Popen(args, stdout=fh, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 120
        while True:
            if proc.poll() is not None:
                pytest.fail(f"llama-server exited: {log_path.read_text()[-2000:]}")
            try:
                if _http(f"http://127.0.0.1:{port}/health")[0] == 200:
                    break
            except OSError:
                pass
            if time.time() > deadline:
                pytest.fail("llama-server did not become healthy")
            time.sleep(0.2)
        if cfg.embedding:
            _http(f"http://127.0.0.1:{port}/embedding", {"content": "hello world " * 200}, timeout=120)
        else:
            _http(f"http://127.0.0.1:{port}/completion",
                  {"prompt": "hello world " * 700, "n_predict": 8, "cache_prompt": True}, timeout=300)
        measured = mem.process_memory_bytes(proc.pid)
        assert measured is not None
        return measured, log_path.read_text()
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _logged_kv_mib(log: str) -> float | None:
    match = re.search(r"llama_kv_cache: size =\s+([\d.]+) MiB", log)
    return float(match.group(1)) if match else None


@pytest.mark.real_model
@pytest.mark.parametrize("name,n_ctx,n_parallel", [
    ("gguf_tiny", 4096, 1),
    ("gguf_tiny", 8192, 2),
    ("gguf_small", 4096, 1),
])
def test_gguf_estimate_close_to_llama_server(tmp_path, name, n_ctx, n_parallel):
    model = gguf_local_model(model_path(name))
    cfg = LoadConfig(n_ctx=n_ctx, n_parallel=n_parallel, flash_attn="auto")
    parts = mem.breakdown(model, cfg)
    measured, log = _run_llama(model, cfg, tmp_path / "llama.log")
    kv_mib = _logged_kv_mib(log)
    assert kv_mib is not None, "llama-server did not report its KV cache size"
    assert parts.kv_cache / MiB == pytest.approx(kv_mib, abs=0.5)
    ratio = parts.total / measured
    assert 0.9 <= ratio <= 1.25, f"estimate {parts.total / MiB:.0f} MiB vs measured {measured / MiB:.0f} MiB"


@pytest.mark.real_model
def test_gguf_embedding_estimate_close_to_llama_server(tmp_path):
    model = gguf_local_model(model_path("gguf_embed"), embedding=True)
    cfg = LoadConfig(n_ctx=2048, embedding=True)
    parts = mem.breakdown(model, cfg)
    measured, log = _run_llama(model, cfg, tmp_path / "embed.log")
    assert parts.kv_cache == 0 and _logged_kv_mib(log) is None
    ratio = parts.total / measured
    assert 0.8 <= ratio <= 1.4, f"estimate {parts.total / MiB:.0f} MiB vs measured {measured / MiB:.0f} MiB"


MLX_PROBE = r"""
import ctypes, json, os, sys
import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
sys.path.insert(0, sys.argv[3])
from localcode.runtime.memory import process_memory_bytes
model, tok = load(sys.argv[1])
mx.eval(model.parameters())
active_loaded = mx.get_active_memory()
proc_loaded = process_memory_bytes(os.getpid())
n = int(sys.argv[2])
ids = tok.encode("hello world " * n)[:n]
mx.reset_peak_memory()
cache = make_prompt_cache(model)
for _, _ in zip(range(4), generate_step(mx.array(ids), model, prompt_cache=cache, max_tokens=4)):
    pass
print(json.dumps({"active_loaded": active_loaded, "proc_loaded": proc_loaded,
                  "peak": mx.get_peak_memory(), "proc_after": process_memory_bytes(os.getpid())}))
"""


@pytest.mark.real_model
def test_mlx_estimate_close_to_measured(tmp_path):
    path = model_path("mlx_tiny")
    model = mlx_local_model(path)
    n_tokens = 2000
    cfg = LoadConfig(n_ctx=n_tokens)
    parts = mem.breakdown(model, cfg)
    repo = str(Path(__file__).resolve().parents[2])
    proc = subprocess.run([sys.executable, "-c", MLX_PROBE, str(path), str(n_tokens), repo],
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    # weights ≈ safetensors size
    assert parts.weights == pytest.approx(data["active_loaded"], rel=0.05)
    process_base = max(0, data["proc_loaded"] - data["active_loaded"])
    measured = data["peak"] + process_base
    ratio = parts.total / measured
    assert 0.85 <= ratio <= 1.3, f"estimate {parts.total / MiB:.0f} MiB vs measured {measured / MiB:.0f} MiB"
