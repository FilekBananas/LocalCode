"""Tests for hub/metadata.py: file-name parsing, GGUF headers, MLX configs, capabilities.

The synthetic fixtures write real GGUF/safetensors bytes (see ``hub_fixtures.py``).
The ``real_model`` tests read the five downloaded test models and cross-check the
result against an authority: ``llama-server``'s own ``/props`` + ``/v1/models``
for GGUF, the checkpoint's ``config.json`` for MLX.
"""

from __future__ import annotations

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from hub_fixtures import (MLX_QWEN3_CONFIG, QWEN3_06B_KV, make_mlx_dir, write_embedding_gguf, write_gguf,
                          write_split_gguf)
from lc_testing import model_path
from localcode.hub import metadata as md
from localcode.hub.metadata import (MetadataError, base_quant, detect_capabilities, ftype_to_quant, gguf_stem,
                                    is_mmproj, meta_from_hf_config, mlx_quant_label, parse_quant,
                                    parse_size_label, read_gguf_header, read_gguf_meta, read_hf_chat_template,
                                    read_mlx_meta, split_shard, template_supports_thinking,
                                    template_supports_tools)

LLAMA_SERVER = "/opt/homebrew/bin/llama-server"
FTYPE_NAMES = {"Q4_K - Medium": "Q4_K_M", "Q8_0": "Q8_0", "Q4_K - Small": "Q4_K_S", "F16": "F16", "BF16": "BF16"}


# --------------------------------------------------------------------------- #
# File-name helpers
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name,expected", [
    ("Qwen3-0.6B-Q4_K_M.gguf", "Q4_K_M"),
    ("Qwen3-4B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
    ("model.q8_0.gguf", "Q8_0"),
    ("gpt-oss-20b-mxfp4.gguf", "MXFP4"),
    ("gpt-oss-20b-MXFP4_MOE.gguf", "MXFP4"),
    ("x-BF16-00001-of-00002.gguf", "BF16"),
    ("x-fp16.gguf", "F16"),
    ("Llama-3.2-1B-IQ4_XS.gguf", "IQ4_XS"),
    ("model-TQ1_0.gguf", "TQ1_0"),
    ("Qwen3-30B-A3B-i1-Q3_K_S.gguf", "I1-Q3_K_S"),   # imatrix prefix kept, base_quant() drops it
    ("no-quant-here.gguf", None),
    ("Qwen3-235B-128k.gguf", None),          # a context size is not a quantisation
])
def test_parse_quant(name: str, expected: str | None):
    assert parse_quant(name) == expected


def test_base_quant_drops_vendor_prefixes():
    assert base_quant("UD-Q4_K_XL") == "Q4_K_XL"
    assert base_quant("I1-Q4_K_M") == "Q4_K_M"
    assert base_quant("Q8_0") == "Q8_0"
    assert base_quant(None) is None


def test_split_shard_and_stem():
    assert split_shard("model-00002-of-00003.gguf") == ("model", 2, 3)
    assert split_shard("model.gguf") is None
    assert gguf_stem("dir/model-00002-of-00003.gguf") == "model"
    assert gguf_stem("dir/model.gguf") == "model"


def test_is_mmproj():
    assert is_mmproj("mmproj-F16.gguf")
    assert is_mmproj("Q4/mmproj-model-f16.gguf")
    assert not is_mmproj("model-Q4_K_M.gguf")
    assert not is_mmproj("mmproj-F16.txt")


@pytest.mark.parametrize("text,total,active", [
    ("Qwen3-Coder-30B-A3B-Instruct", 30.0, 3.0),
    ("Mixtral-8x7B", 56.0, None),
    ("Qwen3-0.6B", 0.6, None),
    ("granite-4.1-350M", 0.35, None),
    ("Qwen3-235B-A22B-128k", 235.0, 22.0),
    ("no-size-here", None, None),
])
def test_parse_size_label(text: str, total: float | None, active: float | None):
    assert parse_size_label(text) == (pytest.approx(total) if total else total,
                                      pytest.approx(active) if active else active)


def test_ftype_to_quant_uses_the_llama_cpp_enum():
    assert ftype_to_quant(15) == "Q4_K_M"
    assert ftype_to_quant(7) == "Q8_0"
    assert ftype_to_quant(32) == "BF16"
    assert ftype_to_quant(None) is None
    assert ftype_to_quant(9999) is None


# --------------------------------------------------------------------------- #
# GGUF header parsing
# --------------------------------------------------------------------------- #


def test_read_gguf_meta_of_a_synthetic_model(tmp_path: Path):
    path = write_gguf(tmp_path / "Qwen3-0.6B-Q4_K_M.gguf", tokens=["<a>", "<b>", "<|im_end|>"],
                      eos_id=2, chat_template="{% if tools %}{{ tools }}{% endif %}",
                      general={"parameter_count": 596_049_920})
    meta = read_gguf_meta(path)
    assert meta.arch == "qwen3"
    assert meta.context_length == 40960
    assert meta.n_layers == 28 and meta.n_full_attn_layers == 28
    assert meta.n_heads == 16 and meta.n_kv_heads == 8 and meta.head_dim == 128
    assert meta.hidden_size == 1024 and meta.vocab_size == 3
    assert meta.params_b == 0.596
    assert meta.quant == "Q4_K_M"
    assert meta.eos_token == "<|im_end|>"
    assert meta.is_embedding is False
    assert "tools" in (meta.chat_template or "")


def test_quantisation_falls_back_to_the_file_name(tmp_path: Path):
    path = write_gguf(tmp_path / "model-Q6_K.gguf", file_type=None)
    assert read_gguf_meta(path).quant == "Q6_K"


def test_big_arrays_are_skipped_not_decoded(tmp_path: Path):
    path = write_gguf(tmp_path / "m.gguf", tokens=[f"t{i}" for i in range(5000)])
    header = read_gguf_header(path, want_tensors=False)
    assert isinstance(header.kv["tokenizer.ggml.tokens"], md.BigArray)
    assert header.kv["tokenizer.ggml.tokens"].count == 5000
    assert read_gguf_meta(path).vocab_size == 5000


def test_a_truncated_header_still_yields_the_architecture(tmp_path: Path):
    path = write_gguf(tmp_path / "m.gguf", tokens=[f"token-{i}" for i in range(3000)])
    blob = path.read_bytes()[:400]
    header = read_gguf_header(blob, want_tensors=False, allow_truncated=True)
    assert header.truncated is True
    assert header.kv["general.architecture"] == "qwen3"
    assert header.kv["qwen3.block_count"] == 28
    with pytest.raises(MetadataError):
        read_gguf_header(blob, want_tensors=False)


def test_not_a_gguf_file(tmp_path: Path):
    bad = tmp_path / "x.gguf"
    bad.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(MetadataError):
        read_gguf_meta(bad)


def test_sliding_window_layers_are_excluded_from_the_full_attention_count(tmp_path: Path):
    params = dict(QWEN3_06B_KV, block_count=12, **{"attention.sliding_window": 1024,
                                                   "attention.sliding_window_pattern": 4})
    meta = read_gguf_meta(write_gguf(tmp_path / "gemma.gguf", arch="gemma3", params=params))
    assert meta.sliding_window == 1024
    assert meta.n_layers == 12
    assert meta.n_full_attn_layers == 3          # every 4th layer keeps a full KV cache


def test_mla_models_are_reduced_to_one_latent_per_token(tmp_path: Path):
    params = dict(QWEN3_06B_KV, **{"attention.kv_lora_rank": 512, "rope.dimension_count": 64})
    meta = read_gguf_meta(write_gguf(tmp_path / "glm.gguf", arch="glm4moe", params=params))
    assert meta.n_kv_heads == 1
    assert meta.head_dim == 288                  # ceil((512 + 64) / 2)


def test_embedding_models_are_detected(tmp_path: Path):
    meta = read_gguf_meta(write_embedding_gguf(tmp_path / "nomic-embed-text-v1.5.Q8_0.gguf"))
    assert meta.is_embedding is True
    assert meta.arch == "nomic-bert"
    assert detect_capabilities(meta, ["nomic-embed-text-v1.5.Q8_0.gguf"]) == ["embedding"]


def test_split_model_metadata_comes_from_the_first_shard(tmp_path: Path):
    shards = write_split_gguf(tmp_path, "big-Q4_K_M", 3, general={"parameter_count": 1_000_000_000})
    assert read_gguf_meta(shards[0]).arch == "qwen3"
    assert read_gguf_meta(shards[2]).arch == "qwen3"     # any shard resolves to shard 1
    (tmp_path / shards[0].name).unlink()
    with pytest.raises(MetadataError):
        read_gguf_meta(shards[2])


def test_moe_active_parameters(tmp_path: Path):
    params = dict(QWEN3_06B_KV, expert_count=128, expert_used_count=8)
    path = write_gguf(tmp_path / "moe.gguf", params=params,
                      general={"size_label": "30B-A3B", "parameter_count": 30_000_000_000})
    meta = read_gguf_meta(path)
    assert meta.params_b == 30.0
    assert meta.active_params_b == 3.0


# --------------------------------------------------------------------------- #
# MLX / HF config.json
# --------------------------------------------------------------------------- #


def test_read_mlx_meta_of_a_synthetic_checkpoint(tmp_path: Path):
    d = make_mlx_dir(tmp_path / "Qwen3-0.6B-4bit",
                     tokenizer_config={"eos_token": {"content": "<|im_end|>"}, "bos_token": None},
                     chat_template="{% if tools %}{{ tools }}{% endif %}")
    meta = read_mlx_meta(d)
    assert meta.arch == "qwen3"
    assert meta.context_length == 40960
    assert meta.n_layers == 28 and meta.n_full_attn_layers == 28
    assert meta.n_kv_heads == 8 and meta.head_dim == 128
    assert meta.quant == "4bit"
    assert meta.eos_token == "<|im_end|>"
    assert meta.is_embedding is False
    assert detect_capabilities(meta, []) == ["tools"]


def test_missing_config_json_is_an_error(tmp_path: Path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(MetadataError):
        read_mlx_meta(tmp_path / "empty")


@pytest.mark.parametrize("config,expected", [
    ({"quantization": {"bits": 4, "group_size": 64}}, "4bit"),
    ({"quantization": {"bits": 8}}, "8bit"),
    ({"quantization": {"mode": "mxfp4"}}, "MXFP4"),
    ({"torch_dtype": "bfloat16"}, "BF16"),
    ({"dtype": "float16"}, "F16"),
    ({}, None),
])
def test_mlx_quant_label(config: dict[str, Any], expected: str | None):
    assert mlx_quant_label(config) == expected


def test_hybrid_layer_types_are_understood():
    config = dict(MLX_QWEN3_CONFIG, num_hidden_layers=6, sliding_window=512,
                  layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2)
    meta = meta_from_hf_config(config, name="gemma-4-12b-it")
    assert meta.n_full_attn_layers == 2
    assert meta.sliding_window == 512


def test_recurrent_layers_do_not_count_as_full_attention():
    config = dict(MLX_QWEN3_CONFIG, num_hidden_layers=4,
                  layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"])
    meta = meta_from_hf_config(config, name="qwen3.5-4b")
    assert meta.n_full_attn_layers == 1


def test_parameter_count_of_a_quantised_checkpoint(tmp_path: Path):
    weights = {"model.safetensors": {
        "model.layers.0.self_attn.q_proj.weight": ("U32", [1024, 128]),   # 4-bit packed → ×8
        "model.layers.0.self_attn.q_proj.scales": ("F16", [1024, 8]),
        "model.layers.0.self_attn.q_proj.biases": ("F16", [1024, 8]),
    }}
    d = make_mlx_dir(tmp_path / "Model-4bit", weights=weights)
    meta = read_mlx_meta(d)
    assert meta.params_b == round(1024 * 128 * 8 / 1e9, 3)


def test_chat_template_precedence(tmp_path: Path):
    d = make_mlx_dir(tmp_path / "m", chat_template=None,
                     tokenizer_config={"chat_template": "from-tokenizer-config"})
    assert read_hf_chat_template(d) == "from-tokenizer-config"
    (d / "chat_template.jinja").write_text("from-jinja-file")
    assert read_hf_chat_template(d) == "from-jinja-file"


def test_embedding_checkpoints_are_detected(tmp_path: Path):
    config = {"model_type": "nomic_bert", "hidden_size": 768, "num_hidden_layers": 12,
              "num_attention_heads": 12, "max_position_embeddings": 2048,
              "architectures": ["NomicBertModel"]}
    d = make_mlx_dir(tmp_path / "nomic-embed-text-v1.5", config=config, chat_template=None)
    meta = read_mlx_meta(d)
    assert meta.is_embedding is True
    assert detect_capabilities(meta, []) == ["embedding"]


# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #


def test_template_capability_detection():
    assert template_supports_tools("{%- if tools %}{{ tools }}{% endif %}")
    assert not template_supports_tools("{{ messages }}")
    assert template_supports_thinking("{% if enable_thinking %}<think>{% endif %}")
    assert template_supports_thinking("{{ add_generation_prompt }}<think>\n")
    assert not template_supports_thinking("{% if message.content.startswith('<think>') %}{% endif %}")
    assert template_supports_thinking(None, "DeepSeek-R1-Distill-Qwen-7B")
    assert not template_supports_thinking(None, "Qwen3-4B-Instruct-2507")


def test_detect_capabilities_orders_and_adds_vision(tmp_path: Path):
    meta = read_gguf_meta(write_gguf(tmp_path / "m.gguf",
                                     chat_template="{% if tools %}{% if enable_thinking %}{% endif %}{% endif %}"))
    caps = detect_capabilities(meta, ["m.gguf", "mmproj-F16.gguf"])
    assert caps == ["tools", "thinking", "vision"]
    caps_cfg = detect_capabilities(meta, ["m.gguf"], config={"vision_config": {"depth": 1}})
    assert caps_cfg == ["tools", "thinking", "vision"]


# --------------------------------------------------------------------------- #
# Real models
# --------------------------------------------------------------------------- #


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _get_json(url: str, timeout: float = 15.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def llama_server_facts(gguf: Path, extra: list[str], tmp_path: Path) -> dict[str, Any]:
    """Start llama-server on ``gguf`` and return ``/props`` + ``/v1/models`` facts."""
    if not Path(LLAMA_SERVER).exists():  # pragma: no cover - depends on the machine
        pytest.skip("llama-server not installed")
    port = _free_port()
    log = open(tmp_path / f"llama-{port}.log", "w")
    proc = subprocess.Popen(
        [LLAMA_SERVER, "-m", str(gguf), "--host", "127.0.0.1", "--port", str(port),
         "-c", "512", "-np", "1", "-ngl", "99", "--no-webui", *extra],
        stdout=log, stderr=log,
    )
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if proc.poll() is not None:  # pragma: no cover - server refused to start
                pytest.fail(f"llama-server exited with {proc.returncode}: {Path(log.name).read_text()[-800:]}")
            try:
                _get_json(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except (urllib.error.URLError, OSError, TimeoutError):
                time.sleep(0.2)
        else:  # pragma: no cover
            pytest.fail("llama-server did not become healthy in time")
        props = _get_json(f"http://127.0.0.1:{port}/props")
        models = _get_json(f"http://127.0.0.1:{port}/v1/models")
        return {"props": props, "meta": models["data"][0]["meta"]}
    finally:
        proc.terminate()
        try:
            proc.wait(20)
        except subprocess.TimeoutExpired:  # pragma: no cover
            proc.kill()
            proc.wait(10)
        log.close()


@pytest.mark.real_model
@pytest.mark.parametrize("key,extra,expected", [
    ("gguf_tiny", ["--jinja"], {"arch": "qwen3", "context_length": 40960, "n_layers": 28, "n_heads": 16,
                                "n_kv_heads": 8, "head_dim": 128, "hidden_size": 1024, "vocab_size": 151936,
                                "params_b": 0.596, "quant": "Q4_K_M", "is_embedding": False,
                                "n_full_attn_layers": 28, "sliding_window": None}),
    ("gguf_small", ["--jinja"], {"arch": "qwen3", "context_length": 262144, "n_layers": 36, "n_heads": 32,
                                 "n_kv_heads": 8, "head_dim": 128, "hidden_size": 2560, "vocab_size": 151936,
                                 "params_b": 4.02, "quant": "Q4_K_M", "is_embedding": False,
                                 "n_full_attn_layers": 36, "sliding_window": None}),
    ("gguf_embed", ["--embeddings", "--pooling", "mean"],
     {"arch": "nomic-bert", "context_length": 2048, "n_layers": 12, "n_heads": 12, "n_kv_heads": 12,
      "head_dim": 64, "hidden_size": 768, "vocab_size": 30522, "params_b": 0.137, "quant": "Q8_0",
      "is_embedding": True, "n_full_attn_layers": 12, "sliding_window": None}),
])
def test_real_gguf_metadata_matches_llama_server(key: str, extra: list[str], expected: dict[str, Any],
                                                 tmp_path: Path):
    path = model_path(key)
    meta = read_gguf_meta(path)
    for field, value in expected.items():
        assert getattr(meta, field) == value, f"{key}.{field}"

    facts = llama_server_facts(path, extra, tmp_path)
    server = facts["meta"]
    assert meta.context_length == server["n_ctx_train"]
    assert meta.hidden_size == server["n_embd"]
    assert meta.vocab_size == server["n_vocab"]
    assert meta.params_b == pytest.approx(server["n_params"] / 1e9, rel=0.01)
    assert FTYPE_NAMES[facts["props"]["model_ftype"]] == meta.quant
    assert meta.eos_token == facts["props"]["eos_token"]
    if meta.is_embedding:
        assert not meta.chat_template          # llama.cpp substitutes a default template for BERT models
    else:
        assert (meta.chat_template or "").strip() == (facts["props"]["chat_template"] or "").strip()
        assert "tools" in detect_capabilities(meta, [path.name], name=path.name)


@pytest.mark.real_model
@pytest.mark.parametrize("key,expected", [
    ("mlx_tiny", {"arch": "qwen3", "context_length": 40960, "n_layers": 28, "n_heads": 16, "n_kv_heads": 8,
                  "head_dim": 128, "hidden_size": 1024, "vocab_size": 151936, "params_b": 0.596,
                  "quant": "4bit", "is_embedding": False, "n_full_attn_layers": 28}),
    ("mlx_small", {"arch": "qwen3", "context_length": 262144, "n_layers": 36, "n_heads": 32, "n_kv_heads": 8,
                   "head_dim": 128, "hidden_size": 2560, "vocab_size": 151936, "params_b": 4.02,
                   "quant": "4bit", "is_embedding": False, "n_full_attn_layers": 36}),
])
def test_real_mlx_metadata_matches_config_json(key: str, expected: dict[str, Any]):
    directory = model_path(key)
    meta = read_mlx_meta(directory)
    for field, value in expected.items():
        assert getattr(meta, field) == value, f"{key}.{field}"

    config = json.loads((directory / "config.json").read_text())
    assert meta.arch == config["model_type"]
    assert meta.context_length == config["max_position_embeddings"]
    assert meta.n_layers == config["num_hidden_layers"]
    assert meta.n_heads == config["num_attention_heads"]
    assert meta.n_kv_heads == config["num_key_value_heads"]
    assert meta.head_dim == config["head_dim"]
    assert meta.hidden_size == config["hidden_size"]
    assert meta.vocab_size == config["vocab_size"]
    assert meta.quant == f"{config['quantization']['bits']}bit"
    assert meta.chat_template and "im_start" in meta.chat_template


@pytest.mark.real_model
def test_gguf_and_mlx_of_the_same_model_agree():
    gguf = read_gguf_meta(model_path("gguf_tiny"))
    mlx = read_mlx_meta(model_path("mlx_tiny"))
    for field in ("arch", "context_length", "n_layers", "n_heads", "n_kv_heads", "head_dim", "hidden_size",
                  "vocab_size", "params_b", "n_full_attn_layers"):
        assert getattr(gguf, field) == getattr(mlx, field), field


# --------------------------------------------------------------------------- #
# Telling the user why the answers are nonsense
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("quant", ["IQ1_S", "I1-IQ1_S", "Q2_K", "IQ2_XXS", "TQ1_0", "Q1_0"])
def test_a_one_or_two_bit_build_is_flagged(quant: str) -> None:
    """The reported failure: a 1-bit 30B model answered with looping markup."""
    from localcode.hub.metadata import quant_warning

    warning = quant_warning(quant)
    assert warning and "bez sensu" in warning
    assert "Q4" in warning                  # says what to pick instead


@pytest.mark.parametrize("quant", ["Q3_K_M", "Q4_K_M", "UD-Q4_K_XL", "Q5_K_M", "Q8_0",
                                   "BF16", "F16", "MXFP4", "4bit", "8bit", None, ""])
def test_an_ordinary_quantisation_is_not_flagged(quant: str | None) -> None:
    from localcode.hub.metadata import quant_warning

    assert quant_warning(quant) is None


def test_the_warning_names_the_bit_depth_it_found() -> None:
    from localcode.hub.metadata import quant_warning

    assert "1-bitowa" in (quant_warning("IQ1_M") or "")
    assert "2-bitowa" in (quant_warning("Q2_K") or "")


@pytest.mark.parametrize("params, flagged", [
    (0.6, True), (1.2, True), (2.9, True), (3.0, False), (4.0, False), (30.5, False),
    (None, False), (0, False),
])
def test_a_model_too_small_for_the_agent_is_flagged(params: float | None, flagged: bool) -> None:
    from localcode.hub.metadata import agent_model_warning

    assert bool(agent_model_warning(params)) is flagged


def test_the_size_warning_is_written_in_polish_numbers() -> None:
    from localcode.hub.metadata import agent_model_warning

    warning = agent_model_warning(0.6) or ""
    assert "0,6 mld" in warning              # comma, not a full stop
    assert "Qwen3 4B" in warning              # and a concrete alternative


# --------------------------------------------------------------------------- #
# Video checkpoints must not land on the image shelf
# --------------------------------------------------------------------------- #
# A LTX-Video GGUF carries ``transformer_blocks.*`` exactly like FLUX does, so
# the diffusion probe used to call it an image model with family "unknown". The
# Mac app then offered it under „Obrazy", and picking it killed the image
# backend with "ValueError: NULL pointer access". These models belong to
# localcode.video (``classify_video_model``).


@pytest.mark.parametrize("filename", [
    "ltx-video-2b-v0.9-Q4_K_M.gguf",
    "ltxv-2b-0.9.6-distilled-04-25-BF16.gguf",
    "Wan2.1-T2V-1.3B-Q4_K_M.gguf",
    "hunyuan-video-t2v-720p-Q4_K_M.gguf",
    "CogVideoX-5b-Q8_0.gguf",
])
def test_a_video_checkpoint_is_not_an_image_model(filename: str) -> None:
    from localcode.hub.metadata import gguf_image_family, is_video_checkpoint

    assert is_video_checkpoint(filename=filename) is True
    # The tensor names below are what the LTX GGUF actually ships (first ones
    # read by the probe) — they match _DIFFUSION_TENSOR_MARKERS.
    names = ["adaln_single.emb.timestep_embedder.linear_1.weight",
             "transformer_blocks.0.attn1.to_q.weight"]
    assert gguf_image_family({"general.architecture": "ltxv"}, names, filename) is None


@pytest.mark.parametrize("filename", [
    "stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf",
    "flux1-schnell-Q4_0.gguf",
    "sd3.5_medium-Q4_0.gguf",
])
def test_an_image_checkpoint_is_still_an_image_model(filename: str) -> None:
    from localcode.hub.metadata import gguf_image_family, is_video_checkpoint

    assert is_video_checkpoint(filename=filename) is False
    names = ["model.diffusion_model.input_blocks.0.0.weight", "time_embed.0.weight"]
    assert gguf_image_family({"general.architecture": "sd1"}, names, filename) is not None


def test_a_video_directory_is_not_an_image_directory(tmp_path: Path) -> None:
    """The same trap for diffusers/MLX layouts, where only the name can tell."""
    from localcode.hub.metadata import is_image_model_dir

    for name, expected in (("LTX-Video-2B-diffusers", False), ("FLUX.1-schnell-mflux-4bit", True)):
        d = tmp_path / name
        (d / "transformer").mkdir(parents=True)
        (d / "vae").mkdir()
        (d / "transformer" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
        (d / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
        assert is_image_model_dir(d) is expected
