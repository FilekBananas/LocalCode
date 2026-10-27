"""Builders for synthetic model directories used by the hub tests.

Everything here writes *real* file formats (a valid GGUF header via
``gguf.GGUFWriter``, a valid safetensors header, HF-style ``config.json``),
because the code under test parses those bytes — a mock would only test the
mock.  Files stay tiny: the GGUF fixtures contain no tensor data and the
safetensors fixtures only the requested number of zero bytes.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Iterable, Sequence

from gguf import GGUFValueType, GGUFWriter

# --------------------------------------------------------------------------- #
# GGUF
# --------------------------------------------------------------------------- #

QWEN3_06B_KV: dict[str, Any] = {
    "context_length": 40960,
    "block_count": 28,
    "embedding_length": 1024,
    "attention.head_count": 16,
    "attention.head_count_kv": 8,
    "attention.key_length": 128,
    "attention.value_length": 128,
}


def write_gguf(
    path: str | os.PathLike[str],
    *,
    arch: str = "qwen3",
    params: dict[str, Any] | None = None,
    general: dict[str, Any] | None = None,
    file_type: int | None = 15,             # MOSTLY_Q4_K_M
    chat_template: str | None = None,
    tokens: Sequence[str] | None = None,
    bos_id: int | None = None,
    eos_id: int | None = None,
    extra: dict[str, tuple[Any, GGUFValueType]] | None = None,
    pad_to: int = 0,
) -> Path:
    """Write a tensor-less but structurally valid GGUF file.

    ``params`` keys are architecture-scoped (``"block_count"`` → ``"qwen3.block_count"``),
    ``general`` keys are written under ``general.*``, ``extra`` holds fully
    qualified keys with an explicit value type.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(p, arch)
    for key, value in (general or {}).items():
        if isinstance(value, str):
            writer.add_string(f"general.{key}", value)
        elif isinstance(value, int):
            writer.add_uint64(f"general.{key}", value)
    for key, value in (params if params is not None else QWEN3_06B_KV).items():
        full = f"{arch}.{key}"
        if isinstance(value, bool):
            writer.add_bool(full, value)
        elif isinstance(value, int):
            writer.add_uint32(full, value)
        elif isinstance(value, float):
            writer.add_float32(full, value)
        elif isinstance(value, str):
            writer.add_string(full, value)
        elif isinstance(value, list):
            writer.add_array(full, value)
    if file_type is not None:
        writer.add_file_type(file_type)
    if tokens is not None:
        writer.add_token_list(list(tokens))
    if bos_id is not None:
        writer.add_bos_token_id(bos_id)
    if eos_id is not None:
        writer.add_eos_token_id(eos_id)
    if chat_template is not None:
        writer.add_chat_template(chat_template)
    for key, (value, vtype) in (extra or {}).items():
        writer.add_key_value(key, value, vtype)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    if pad_to and p.stat().st_size < pad_to:
        with open(p, "ab") as fh:                      # size only: the parser stops at the header
            fh.write(b"\0" * (pad_to - p.stat().st_size))
    return p


def write_split_gguf(directory: str | os.PathLike[str], stem: str, count: int, **kwargs: Any) -> list[Path]:
    """``stem-00001-of-000NN.gguf`` … — shard 1 carries the metadata."""
    out: list[Path] = []
    for idx in range(1, count + 1):
        name = f"{stem}-{idx:05d}-of-{count:05d}.gguf"
        extra = dict(kwargs.pop("extra", {})) if idx == 1 else {}
        if idx == 1:
            extra["split.count"] = (count, GGUFValueType.UINT16)
        out.append(write_gguf(Path(directory) / name, extra=extra, **kwargs))
    return out


def write_mmproj(path: str | os.PathLike[str], *, file_type: int = 1) -> Path:
    """A vision projector file (``mmproj-*.gguf``)."""
    return write_gguf(
        path, arch="clip", params={"block_count": 2}, file_type=file_type,
        extra={"clip.has_vision_encoder": (True, GGUFValueType.BOOL)},
    )


def write_embedding_gguf(path: str | os.PathLike[str], *, arch: str = "nomic-bert") -> Path:
    return write_gguf(
        path,
        arch=arch,
        params={"context_length": 2048, "block_count": 12, "embedding_length": 768,
                "attention.head_count": 12, "attention.head_count_kv": 12,
                "attention.causal": False, "pooling_type": 1},
        general={"name": "nomic-embed-text-v1.5"},
        file_type=7,
    )


# --------------------------------------------------------------------------- #
# MLX / safetensors
# --------------------------------------------------------------------------- #


def write_safetensors(path: str | os.PathLike[str], tensors: dict[str, tuple[str, list[int]]] | None = None) -> Path:
    """Minimal but valid safetensors file (header + zeroed data)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tensors = tensors or {"model.embed_tokens.weight": ("F16", [512, 64])}
    width = {"F16": 2, "BF16": 2, "F32": 4, "U32": 4, "U8": 1}
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        n = 1
        for dim in shape:
            n *= dim
        size = n * width.get(dtype, 2)
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    with open(p, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        fh.write(b"\0" * offset)
    return p


MLX_QWEN3_CONFIG: dict[str, Any] = {
    "model_type": "qwen3",
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 40960,
    "vocab_size": 151936,
    "quantization": {"group_size": 64, "bits": 4},
}


def make_mlx_dir(
    directory: str | os.PathLike[str],
    *,
    config: dict[str, Any] | None = None,
    weights: dict[str, dict[str, tuple[str, list[int]]]] | None = None,
    index: dict[str, str] | None = None,
    chat_template: str | None = "{% for m in messages %}{{ m.content }}{% endfor %}",
    tokenizer_config: dict[str, Any] | None = None,
    readme: str | None = None,
) -> Path:
    """An MLX checkpoint directory (config.json + safetensors + tokenizer files)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps(config if config is not None else MLX_QWEN3_CONFIG))
    for name, tensors in (weights or {"model.safetensors": None}).items():  # type: ignore[union-attr]
        write_safetensors(d / name, tensors)
    if index is not None:
        (d / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    tok = dict(tokenizer_config or {"eos_token": "<|im_end|>", "bos_token": None})
    if chat_template is not None:
        (d / "chat_template.jinja").write_text(chat_template)
    (d / "tokenizer_config.json").write_text(json.dumps(tok))
    if readme is not None:
        (d / "README.md").write_text(readme)
    return d


# --------------------------------------------------------------------------- #
# Hugging Face cache layout
# --------------------------------------------------------------------------- #


def make_hf_cache_repo(cache_dir: str | os.PathLike[str], repo_id: str, files: dict[str, Path]) -> Path:
    """Build ``models--org--repo/{refs,blobs,snapshots}`` like huggingface_hub does.

    ``files`` maps the repo-relative name to a file that is *moved* into the blob
    store and symlinked from the snapshot (exactly the on-disk shape
    ``scan_cache_dir`` expects).  Returns the snapshot directory.
    """
    cache = Path(cache_dir)
    sha = f"{abs(hash(repo_id)):040x}"[:40]
    root = cache / ("models--" + repo_id.replace("/", "--"))
    (root / "refs").mkdir(parents=True, exist_ok=True)
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    snapshot = root / "snapshots" / sha
    snapshot.mkdir(parents=True, exist_ok=True)
    (root / "refs" / "main").write_text(sha)
    for i, (name, source) in enumerate(files.items()):
        blob = root / "blobs" / f"{abs(hash(name)):064x}"[:64]
        os.replace(source, blob)
        link = snapshot / name
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(blob, link.parent), link)
    return snapshot


def file_names(models: Iterable[Any]) -> list[str]:
    return sorted(m.key for m in models)
