"""Model metadata readers: GGUF headers, MLX/HF ``config.json`` and capabilities.

Design
------
* **GGUF** — ``gguf.GGUFReader`` (gguf 0.19) memory-maps the file and eagerly
  materialises *every* tokenizer string (150k–260k tokens plus as many merges)
  and all tensor infos as numpy views; on the test machine that takes 4–8 s per
  file.  A library scan must stay interactive, so this module contains a small
  streaming parser of the GGUF header (same on-disk format the ``gguf`` package
  documents: magic, version, KV section, tensor-info section) that *skips* big
  arrays by walking their length prefixes.  It reads a 20 GB model's metadata in
  ~0.2–0.5 s and only touches the first few MB of the file.  The parser also
  works on a truncated byte buffer (``allow_truncated=True``) so ``hub/search``
  can read the architecture keys of a remote GGUF from an HTTP Range request of
  its first few hundred KB.  Enum tables (file types, value types) come from the
  ``gguf`` package when available.
* **MLX** — ``config.json`` (with nested ``text_config`` for multimodal
  checkpoints), ``model.safetensors.index.json`` / safetensors headers for the
  parameter count, and the chat template from ``chat_template.jinja`` /
  ``chat_template.json`` / ``tokenizer_config.json``.

KV-cache facts (``ModelMeta`` conventions, relied on by ``runtime.memory``):

* ``n_full_attn_layers`` — layers that keep a KV cache growing with the full
  context.  Always set when the layer count is known (equals ``n_layers`` for a
  plain transformer).
* ``sliding_window`` — window of the *other* attention layers when those are
  sliding-window layers (gemma 2/3/4, gpt-oss, cohere2…).  When it is ``None``
  and ``n_full_attn_layers < n_layers``, the remaining layers hold no growing
  KV (recurrent / SSM / linear-attention / shared-KV layers: qwen3.5+, qwen3-next,
  nemotron-h, lfm2, granite-hybrid…).
* ``n_kv_heads`` / ``head_dim`` describe the *full-attention* layers (the ones
  that dominate KV memory at long context).  For MLA models (DeepSeek-V2/V3,
  GLM-4.7-Flash, Kimi…) the cache holds one compressed latent per token of width
  ``kv_lora_rank + rope_dim``; we encode that as ``n_kv_heads = 1`` and
  ``head_dim = ceil(width / 2)`` so that ``2 * n_kv_heads * head_dim`` equals the
  real per-token, per-layer cache width.
"""

from __future__ import annotations

import json
import logging
import math
import mmap
import os
import re
import struct
from array import array
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from localcode.hub.types import ModelMeta

log = logging.getLogger(__name__)

__all__ = [
    "MetadataError",
    "GGUFHeader",
    "read_gguf_header",
    "read_gguf_meta",
    "meta_from_gguf_kv",
    "read_mlx_meta",
    "read_mlx_config",
    "meta_from_hf_config",
    "read_hf_chat_template",
    "detect_capabilities",
    "template_supports_tools",
    "template_supports_thinking",
    "parse_quant",
    "base_quant",
    "ftype_to_quant",
    "mlx_quant_label",
    "split_shard",
    "gguf_stem",
    "is_mmproj",
    "parse_size_label",
    "EMBEDDING_ARCHS",
    # image models (text-to-image)
    "IMAGE_FAMILIES",
    "detect_image_family",
    "gguf_image_family",
    "image_family_from_name",
    "is_video_checkpoint",
    "is_image_companion",
    "is_image_model_dir",
    "read_image_dir_meta",
]


class MetadataError(ValueError):
    """Metadata could not be read (message is Polish, safe to show)."""


# --------------------------------------------------------------------------- #
# Filename helpers: quantisation labels, split shards, projectors
# --------------------------------------------------------------------------- #

_SHARD_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<count>\d{5})\.gguf$", re.IGNORECASE)

# Quant token at a separator boundary. Order inside the alternation matters only
# for readability; the regex engine takes the longest `(_[0-9A-Z]+)*` tail.
_QUANT_RE = re.compile(
    r"(?:^|[-._])(?P<prefix>(?:UD|I1)-)?"
    r"(?P<q>I?Q\d(?:_[0-9A-Z]+)*|TQ\d_\d|BF16|FP16|F16|FP32|F32|MXFP4(?:_MOE)?|NVFP4)"
    r"(?=$|[-._])",
    re.IGNORECASE,
)


def split_shard(filename: str) -> tuple[str, int, int] | None:
    """``"x-00001-of-00003.gguf"`` → ``("x", 1, 3)``; ``None`` for unsplit files."""
    m = _SHARD_RE.match(os.path.basename(filename))
    if not m:
        return None
    return m.group("stem"), int(m.group("idx")), int(m.group("count"))


def gguf_stem(filename: str) -> str:
    """File name without directory, ``.gguf`` suffix and split-shard suffix."""
    base = os.path.basename(filename)
    shard = split_shard(base)
    if shard:
        return shard[0]
    return base[:-5] if base.lower().endswith(".gguf") else base


def is_mmproj(filename: str) -> bool:
    """Vision/audio projector files (``mmproj-*.gguf``) are not standalone models."""
    base = os.path.basename(filename).lower()
    return base.endswith(".gguf") and "mmproj" in base


#: Files that live next to a model in the same repository but are not the model:
#: speculative-decoding draft heads (EAGLE3, Medusa, "draft"), LoRA adapters and
#: control vectors. A repo like ``ggml-org/gpt-oss-20b-GGUF`` ships an EAGLE3
#: head of ~0.9 GB beside the 12 GB weights; without this filter it wins the
#: "smallest variant that fits" contest and the store offers to install it.
_AUXILIARY_GGUF = re.compile(
    r"(?i)(^|[-._/])(eagle\d*|medusa|draft|speculat\w*|lora|adapter|control[-_]?vector|"
    r"vocab|tokenizer|clip[-_]?[lg]|t5xxl|umt5|vae)([-._]|$)")


def is_auxiliary_gguf(filename: str) -> bool:
    """True for a GGUF that is a companion file, not a model to run on its own."""
    base = os.path.basename(filename)
    if is_mmproj(base):
        return True
    return bool(_AUXILIARY_GGUF.search(base))


def parse_quant(filename: str) -> str | None:
    """Quantisation label from a GGUF file name.

    ``Qwen3-0.6B-UD-Q4_K_XL.gguf`` → ``UD-Q4_K_XL``, ``model.q8_0.gguf`` → ``Q8_0``,
    ``BF16/x-BF16-00001-of-00002.gguf`` → ``BF16``, ``gpt-oss-20b-MXFP4.gguf`` → ``MXFP4``.
    The last match wins because the quant is conventionally at the end.
    """
    stem = gguf_stem(filename)
    matches = list(_QUANT_RE.finditer(stem))
    if not matches:
        return None
    m = matches[-1]
    q = m.group("q").upper()
    q = {"FP16": "F16", "FP32": "F32", "MXFP4_MOE": "MXFP4"}.get(q, q)
    prefix = (m.group("prefix") or "").upper()
    return f"{prefix}{q}" if prefix else q


#: Quantisations that cost so much quality the model often stops making sense.
#: 1-bit and 2-bit GGUF weights fit a 30B model into a laptop and then answer
#: with word salad or one token repeated forever — the reason the agent needed a
#: loop detector at all (``localcode.runtime.degenerate``). They are still
#: offered, because for some very large models they are the only way in, but
#: never silently.
FRAGILE_QUANTS_RE = re.compile(r"^(?:I?Q1|IQ2|Q2|TQ1|TQ2)(?:_|$)")


#: Below this many billion parameters a model can follow a chat, but cannot
#: reliably drive the coding agent: it calls the right tool and then writes
#: nonsense into the file. Measured on Qwen3 0.6B, which happily created
#: ``banana.html`` containing a sentence that means nothing.
AGENT_MIN_PARAMS_B = 3.0


def agent_model_warning(params_b: float | None) -> str | None:
    """One Polish sentence when a model is too small to be an agent, else ``None``.

    The chat is a fair use for a 0,6B model. Editing code is not: the tool calls
    come out right and the content does not, which looks like a broken app rather
    than a model at its limit.
    """
    if not params_b or params_b <= 0 or params_b >= AGENT_MIN_PARAMS_B:
        return None
    size = f"{params_b:.1f}".rstrip("0").rstrip(".").replace(".", ",")
    return (f"Ten model ma {size} mld parametrów — do agenta to za mało: "
            f"wywoła narzędzie poprawnie, ale napisze bezsensowną treść. "
            f"Do pracy z kodem wybierz model od {AGENT_MIN_PARAMS_B:.0f} mld w górę "
            f"(np. Qwen3 4B Instruct).")


def quant_warning(label: str | None) -> str | None:
    """One Polish sentence about a quantisation this aggressive, or ``None``.

    Shown in the store and when such a model is loaded: the user who asked "is
    this correct behaviour?" after a 1-bit 30B model produced looping nonsense
    had no way to know the file itself was the problem.
    """
    base = base_quant(label)
    if not base or not FRAGILE_QUANTS_RE.match(base):
        return None
    bits = "1-bitowa" if base.startswith(("Q1", "IQ1", "TQ1")) else "2-bitowa"
    return (f"To {bits} wersja modelu — najmniejsza, ale jakość spada tak mocno, "
            f"że model może zacząć pisać bez sensu albo się zapętlać. "
            f"Jeśli masz pamięć, wybierz Q4 albo mniejszy model w pełniejszej jakości.")


def base_quant(label: str | None) -> str | None:
    """``UD-Q4_K_M`` → ``Q4_K_M`` (drops Unsloth-dynamic / imatrix prefixes)."""
    if not label:
        return label
    up = label.upper()
    for p in ("UD-", "I1-"):
        if up.startswith(p):
            return up[len(p):]
    return up


# llama.cpp `llama_ftype` values (stable ABI) → names; extended from gguf.LlamaFileType.
_FTYPE_FALLBACK = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
    10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
    16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S",
    22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M",
    28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0", 38: "MXFP4", 39: "NVFP4", 40: "Q1_0",
}


@lru_cache(maxsize=1)
def _ftype_table() -> dict[int, str]:
    table = dict(_FTYPE_FALLBACK)
    try:  # newer gguf releases may know more file types
        import gguf

        for member in gguf.LlamaFileType:
            if member.name == "GUESSED":
                continue
            name = member.name.removeprefix("MOSTLY_").removeprefix("ALL_")
            name = {"MXFP4_MOE": "MXFP4"}.get(name, name)
            table.setdefault(int(member.value), name)
    except Exception:  # pragma: no cover - gguf is a hard dependency, but stay robust
        pass
    return table


def ftype_to_quant(file_type: Any) -> str | None:
    """``general.file_type`` (llama.cpp ftype enum) → ``"Q4_K_M"`` etc."""
    try:
        return _ftype_table().get(int(file_type))
    except (TypeError, ValueError):
        return None


# "K" is deliberately not a parameter unit: "128k"/"262k" in names are context sizes.
_SIZE_LABEL_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9.])(?:(?P<experts>\d+)x)?(?P<num>\d+(?:\.\d+)?)(?P<unit>[MBT])(?![A-Za-z0-9])"
)
_ACTIVE_RE = re.compile(r"(?i)(?<![A-Za-z0-9.])A(?P<num>\d+(?:\.\d+)?)(?P<unit>[MB])(?![A-Za-z0-9])")
_UNIT = {"M": 1e-3, "B": 1.0, "T": 1e3}


def parse_size_label(text: str | None) -> tuple[float | None, float | None]:
    """``"30B-A3B"`` → ``(30.0, 3.0)``; ``"8x7B"`` → ``(56.0, None)``; ``"0.6B"`` → ``(0.6, None)``.

    Works on GGUF ``general.size_label`` values and on repo/file names
    (``Qwen3-Coder-30B-A3B-Instruct``).  Returns billions.
    """
    if not text:
        return None, None
    active = None
    m_active = _ACTIVE_RE.search(text)
    if m_active:
        active = float(m_active.group("num")) * _UNIT[m_active.group("unit").upper()]
        text = text[: m_active.start()] + " " + text[m_active.end():]
    m = _SIZE_LABEL_RE.search(text)
    total = None
    if m:
        total = float(m.group("num")) * _UNIT[m.group("unit").upper()]
        if m.group("experts"):
            total *= int(m.group("experts"))
    return total, active


def _round_b(n_params: float | int | None) -> float | None:
    if not n_params:
        return None
    b = float(n_params) / 1e9
    return round(b, 2) if b >= 1 else round(b, 3)


# --------------------------------------------------------------------------- #
# GGUF header parsing
# --------------------------------------------------------------------------- #

GGUF_MAGIC = b"GGUF"
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16, _T_UINT32, _T_INT32 = 0, 1, 2, 3, 4, 5
_T_FLOAT32, _T_BOOL, _T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = 6, 7, 8, 9, 10, 11, 12
_SCALAR_FMT = {
    _T_UINT8: ("B", 1), _T_INT8: ("b", 1), _T_UINT16: ("H", 2), _T_INT16: ("h", 2),
    _T_UINT32: ("I", 4), _T_INT32: ("i", 4), _T_FLOAT32: ("f", 4), _T_BOOL: ("?", 1),
    _T_UINT64: ("Q", 8), _T_INT64: ("q", 8), _T_FLOAT64: ("d", 8),
}
_U32 = struct.Struct("<I")
_U64 = struct.Struct("<Q")

# arrays larger than this are skipped (recorded as BigArray) instead of decoded
_MAX_DECODED_ARRAY = 4096


@dataclass(frozen=True)
class BigArray:
    """Placeholder for a large array value that was skipped while parsing."""

    item_type: int
    count: int


class _Truncated(Exception):
    pass


@dataclass
class GGUFHeader:
    version: int
    tensor_count: int
    kv: dict[str, Any]
    n_params: int | None = None          # sum of tensor element counts (None if not parsed)
    n_expert_params: int | None = None   # elements of MoE expert tensors (``*_exps``)
    tokens: dict[int, str] = field(default_factory=dict)  # requested token id -> string
    truncated: bool = False              # the buffer ended before the header did
    tensor_names: list[str] = field(default_factory=list)  # first N names (want_tensor_names)


class _Cursor:
    """Bounds-checked little-endian reader over ``bytes`` / ``mmap``."""

    __slots__ = ("buf", "pos", "end")

    def __init__(self, buf: Any) -> None:
        self.buf = buf
        self.pos = 0
        self.end = len(buf)

    def need(self, n: int) -> None:
        if self.pos + n > self.end:
            raise _Truncated()

    def u32(self) -> int:
        self.need(4)
        v = _U32.unpack_from(self.buf, self.pos)[0]
        self.pos += 4
        return v

    def u64(self) -> int:
        self.need(8)
        v = _U64.unpack_from(self.buf, self.pos)[0]
        self.pos += 8
        return v

    def string(self) -> str:
        n = self.u64()
        self.need(n)
        s = bytes(self.buf[self.pos:self.pos + n]).decode("utf-8", errors="replace")
        self.pos += n
        return s

    def skip_strings(self, count: int, offsets: array | None = None) -> None:
        """Walk ``count`` length-prefixed strings (optionally recording offsets)."""
        buf, pos, end, unpack = self.buf, self.pos, self.end, _U64.unpack_from
        if offsets is not None:
            append = offsets.append
            for _ in range(count):
                if pos + 8 > end:
                    self.pos = pos
                    raise _Truncated()
                append(pos)
                pos += 8 + unpack(buf, pos)[0]
        else:
            for _ in range(count):
                if pos + 8 > end:
                    self.pos = pos
                    raise _Truncated()
                pos += 8 + unpack(buf, pos)[0]
        if pos > end:
            self.pos = end
            raise _Truncated()
        self.pos = pos

    def string_at(self, offset: int) -> str | None:
        if offset + 8 > self.end:
            return None
        n = _U64.unpack_from(self.buf, offset)[0]
        if offset + 8 + n > self.end:
            return None
        return bytes(self.buf[offset + 8:offset + 8 + n]).decode("utf-8", errors="replace")


def _read_value(cur: _Cursor, vtype: int, key: str, token_offsets: array | None) -> Any:
    if vtype == _T_STRING:
        return cur.string()
    scalar = _SCALAR_FMT.get(vtype)
    if scalar is not None:
        fmt, size = scalar
        cur.need(size)
        v = struct.unpack_from("<" + fmt, cur.buf, cur.pos)[0]
        cur.pos += size
        return v
    if vtype == _T_ARRAY:
        item_type = cur.u32()
        count = cur.u64()
        if item_type == _T_STRING:
            if key == "tokenizer.ggml.tokens" or count > _MAX_DECODED_ARRAY:
                cur.skip_strings(count, token_offsets if key == "tokenizer.ggml.tokens" else None)
                return BigArray(item_type, count)
            return [cur.string() for _ in range(count)]
        scalar = _SCALAR_FMT.get(item_type)
        if scalar is not None:
            fmt, size = scalar
            nbytes = size * count
            if count > _MAX_DECODED_ARRAY:
                cur.need(nbytes)
                cur.pos += nbytes
                return BigArray(item_type, count)
            cur.need(nbytes)
            values = list(struct.unpack_from(f"<{count}{fmt}", cur.buf, cur.pos))
            cur.pos += nbytes
            return values
        if item_type == _T_ARRAY:  # nested arrays: rare, decode small ones recursively
            return [_read_value(cur, _T_ARRAY, key + "[]", None) for _ in range(count)]
        raise MetadataError(f"Nieznany typ elementu tablicy GGUF ({item_type}) w kluczu {key}.")
    raise MetadataError(f"Nieznany typ wartości GGUF ({vtype}) w kluczu {key}.")


def _parse_header(
    buf: Any,
    *,
    want_tensors: bool,
    allow_truncated: bool,
    source: str,
    want_tensor_names: int = 0,
) -> GGUFHeader:
    cur = _Cursor(buf)
    try:
        cur.need(4)
    except _Truncated:
        raise MetadataError(f"Plik {source} jest pusty lub uszkodzony.") from None
    if bytes(buf[0:4]) != GGUF_MAGIC:
        raise MetadataError(f"Plik {source} nie jest poprawnym plikiem GGUF.")
    cur.pos = 4
    header = GGUFHeader(version=0, tensor_count=0, kv={})
    token_offsets = array("Q")
    try:
        header.version = cur.u32()
        if header.version not in (2, 3):
            raise MetadataError(f"Nieobsługiwana wersja GGUF ({header.version}) w pliku {source}.")
        header.tensor_count = cur.u64()
        kv_count = cur.u64()
        for _ in range(kv_count):
            key = cur.string()
            vtype = cur.u32()
            header.kv[key] = _read_value(cur, vtype, key, token_offsets)
        if want_tensors:
            n_params = 0
            n_expert = 0
            for _ in range(header.tensor_count):
                name = cur.string()
                n_dims = cur.u32()
                cur.need(8 * n_dims + 12)
                dims = struct.unpack_from(f"<{n_dims}Q", buf, cur.pos)
                cur.pos += 8 * n_dims + 12  # dims + ggml type (u32) + data offset (u64)
                n = math.prod(dims) if dims else 1
                n_params += n
                if "_exps" in name:
                    n_expert += n
                if len(header.tensor_names) < want_tensor_names:
                    header.tensor_names.append(name)
            header.n_params = n_params
            header.n_expert_params = n_expert
    except _Truncated:
        if not allow_truncated:
            raise MetadataError(f"Nagłówek pliku GGUF {source} jest ucięty (plik niekompletny?).") from None
        header.truncated = True
    # resolve special token strings (ids are stored after the token list)
    for id_key in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"):
        tid = header.kv.get(id_key)
        if isinstance(tid, int) and 0 <= tid < len(token_offsets):
            s = cur.string_at(token_offsets[tid])
            if s is not None:
                header.tokens[tid] = s
    return header


def read_gguf_header(
    source: str | os.PathLike[str] | bytes | bytearray | memoryview,
    *,
    want_tensors: bool = True,
    allow_truncated: bool = False,
    want_tensor_names: int = 0,
) -> GGUFHeader:
    """Parse the KV section (and optionally tensor infos) of a GGUF file or buffer.

    ``source`` is a path (memory-mapped, only the header pages are touched) or
    an in-memory prefix of a GGUF file.  Large arrays (tokenizer vocab/merges,
    scores) are skipped and returned as :class:`BigArray`.

    ``want_tensor_names`` keeps the first N tensor names in
    ``GGUFHeader.tensor_names`` (they are decoded by the walk anyway); diffusion
    checkpoints are recognised by those names because they carry no KV metadata
    at all.
    """
    if isinstance(source, (bytes, bytearray, memoryview)):
        return _parse_header(source, want_tensors=want_tensors, allow_truncated=allow_truncated,
                             source="(bufor)", want_tensor_names=want_tensor_names)
    path = Path(source)
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise MetadataError(f"Nie można odczytać pliku {path.name}: {exc.strerror or exc}.") from exc
    if size < 16:
        raise MetadataError(f"Plik {path.name} jest za mały, aby był modelem GGUF.")
    with open(path, "rb") as fh:
        try:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        except (OSError, ValueError) as exc:
            raise MetadataError(f"Nie można odczytać pliku {path.name}: {exc}.") from exc
        try:
            return _parse_header(mm, want_tensors=want_tensors, allow_truncated=allow_truncated,
                                 source=path.name, want_tensor_names=want_tensor_names)
        finally:
            mm.close()


# --------------------------------------------------------------------------- #
# GGUF → ModelMeta
# --------------------------------------------------------------------------- #

# Architectures that are encoders / embedding models in llama.cpp.
EMBEDDING_ARCHS = frozenset({
    "bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert-v3", "gemma-embedding",
    "t5encoder", "modern-bert", "neo-bert", "eurobert", "roberta", "xlm-roberta",
})

# llama.cpp hard-codes the SWA layer pattern for some architectures that only
# store `attention.sliding_window` (no per-layer pattern key): every N-th layer
# is a full-attention layer (`set_swa_pattern(N)`: layer i is SWA iff i % N < N - 1).
_SWA_PERIOD_BY_ARCH = {
    "gemma2": 2, "gemma3": 6, "gemma-embedding": 6, "cohere2": 4, "gpt-oss": 2,
    "exaone4": 4, "gemma3n": 5, "smallthinker": 4, "afmoe": 4,
}

_PURE_RECURRENT_ARCHS = frozenset({"mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7"})


def _as_int(v: Any) -> int | None:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return None


def _per_layer(value: Any, n_layers: int | None) -> list[int] | None:
    """Scalar-or-array hyperparameter → per-layer list (None if unknown)."""
    if value is None or n_layers is None:
        return None
    if isinstance(value, list):
        vals = [(_as_int(v) or 0) for v in value]
        if len(vals) < n_layers:
            vals += [vals[-1] if vals else 0] * (n_layers - len(vals))
        return vals[:n_layers]
    iv = _as_int(value)
    return [iv] * n_layers if iv is not None else None


def _kv_layout(
    *,
    n_layers: int | None,
    kv_heads: list[int] | None,
    is_swa: list[bool] | None,
    is_recurrent: list[bool] | None,
    shared_kv_layers: int,
) -> tuple[int | None, int | None]:
    """Return (n_full_attn_layers, max kv heads over full-attention layers)."""
    if not n_layers:
        return None, (max(kv_heads) if kv_heads else None)
    owners = range(max(0, n_layers - max(0, shared_kv_layers)))
    full: list[int] = []
    for i in owners:
        if is_recurrent and is_recurrent[i]:
            continue
        if kv_heads is not None and kv_heads[i] <= 0:
            continue
        if is_swa and is_swa[i]:
            continue
        full.append(i)
    if kv_heads:
        pool = [kv_heads[i] for i in full] or [h for h in kv_heads if h > 0]
        heads = max(pool) if pool else None
    else:
        heads = None
    return len(full), heads


def meta_from_gguf_kv(
    kv: dict[str, Any],
    *,
    n_params: int | None = None,
    n_expert_params: int | None = None,
    tokens: dict[int, str] | None = None,
    filename: str | None = None,
) -> ModelMeta:
    """Build :class:`ModelMeta` from parsed GGUF key/values (local or remote header)."""
    arch = kv.get("general.architecture")
    arch = arch if isinstance(arch, str) and arch else None
    p = f"{arch}." if arch else "\0"

    def g(key: str) -> Any:
        return kv.get(p + key)

    meta = ModelMeta(arch=arch)
    meta.context_length = _as_int(g("context_length"))
    n_layers = _as_int(g("block_count"))
    meta.n_layers = n_layers

    heads_raw = g("attention.head_count")
    heads = _per_layer(heads_raw, n_layers) if isinstance(heads_raw, list) else None
    meta.n_heads = max(heads) if heads else _as_int(heads_raw)
    meta.hidden_size = _as_int(g("embedding_length"))

    kv_raw = g("attention.head_count_kv")
    kv_heads = _per_layer(kv_raw if kv_raw is not None else heads_raw, n_layers)

    key_len = _as_int(g("attention.key_length"))
    if key_len is None and meta.hidden_size and meta.n_heads:
        key_len = meta.hidden_size // meta.n_heads
    meta.head_dim = key_len

    # sliding-window / hybrid layout -------------------------------------------------
    window = _as_int(g("attention.sliding_window"))
    meta.sliding_window = window if window and window > 0 else None
    is_swa: list[bool] | None = None
    pattern = g("attention.sliding_window_pattern")
    if n_layers and meta.sliding_window:
        if isinstance(pattern, list) and pattern:
            vals = [bool(v) for v in pattern]
            vals += [vals[-1]] * max(0, n_layers - len(vals))
            is_swa = vals[:n_layers]
        else:
            period = _as_int(pattern) if pattern is not None else _SWA_PERIOD_BY_ARCH.get(arch or "")
            if period and period > 1:
                is_swa = [(i % period) < (period - 1) for i in range(n_layers)]
            # unknown pattern → conservatively treat every layer as full attention

    is_recurrent: list[bool] | None = None
    interval = _as_int(g("full_attention_interval"))
    if n_layers and interval and interval > 0:
        is_recurrent = [((i + 1) % interval) != 0 for i in range(n_layers)]
    if arch in _PURE_RECURRENT_ARCHS and n_layers:
        is_recurrent = [True] * n_layers

    shared = _as_int(g("attention.shared_kv_layers")) or 0
    n_full, full_heads = _kv_layout(
        n_layers=n_layers, kv_heads=kv_heads, is_swa=is_swa, is_recurrent=is_recurrent, shared_kv_layers=shared
    )
    meta.n_full_attn_layers = n_full
    meta.n_kv_heads = full_heads if full_heads is not None else _as_int(kv_raw)

    # MLA: one compressed latent per token and layer
    kv_lora = _as_int(g("attention.kv_lora_rank"))
    if kv_lora:
        rope = _as_int(g("rope.dimension_count")) or 64
        meta.n_kv_heads = 1
        meta.head_dim = math.ceil((kv_lora + rope) / 2)

    # parameters -----------------------------------------------------------------------
    label_total, label_active = parse_size_label(kv.get("general.size_label") if isinstance(kv.get("general.size_label"), str) else None)
    count = _as_int(kv.get("general.parameter_count"))
    total = count or n_params
    meta.params_b = _round_b(total) if total else (round(label_total, 3) if label_total else None)
    expert_count = _as_int(g("expert_count"))
    used = _as_int(g("expert_used_count"))
    if label_active:
        meta.active_params_b = round(label_active, 3)
    elif total and n_expert_params and expert_count and used:
        active = total - n_expert_params + n_expert_params * used / expert_count
        meta.active_params_b = _round_b(active)

    # tokenizer ------------------------------------------------------------------------
    vocab = _as_int(g("vocab_size"))
    toks = kv.get("tokenizer.ggml.tokens")
    if vocab is None:
        if isinstance(toks, BigArray):
            vocab = toks.count
        elif isinstance(toks, list):
            vocab = len(toks)
    meta.vocab_size = vocab
    tmpl = kv.get("tokenizer.chat_template")
    meta.chat_template = tmpl if isinstance(tmpl, str) and tmpl.strip() else None
    tokens = tokens or {}
    bos_id = kv.get("tokenizer.ggml.bos_token_id")
    eos_id = kv.get("tokenizer.ggml.eos_token_id")
    meta.bos_token = tokens.get(bos_id) if isinstance(bos_id, int) else None
    meta.eos_token = tokens.get(eos_id) if isinstance(eos_id, int) else None

    # quantisation ---------------------------------------------------------------------
    meta.quant = ftype_to_quant(kv.get("general.file_type"))
    if meta.quant is None and filename:
        meta.quant = parse_quant(filename)

    # embedding models -----------------------------------------------------------------
    pooling = _as_int(g("pooling_type"))
    causal = g("attention.causal")
    name = " ".join(
        str(kv.get(k) or "") for k in ("general.name", "general.basename", "general.finetune")
    ).lower()
    meta.is_embedding = bool(
        (arch in EMBEDDING_ARCHS)
        or (pooling is not None and pooling > 0)
        or (causal is False and arch not in {None, "clip"})
        or ("embed" in name and not meta.chat_template)
    )
    return meta


def read_gguf_meta(path: str | os.PathLike[str]) -> ModelMeta:
    """Metadata of a local GGUF model (any shard of a split model may be given).

    Metadata lives in shard 1; the parameter count sums tensor infos of all
    shards when they are all present.
    """
    p = Path(path)
    shard = split_shard(p.name)
    first = p
    if shard and shard[1] != 1:
        first = p.with_name(f"{shard[0]}-00001-of-{shard[2]:05d}.gguf")
        if not first.exists():
            raise MetadataError(f"Brak pierwszej części modelu dzielonego ({first.name}).")
    header = read_gguf_header(first, want_tensors=True, want_tensor_names=IMAGE_TENSOR_PROBE)
    image_family = gguf_image_family(header.kv, header.tensor_names, p.name)
    if image_family:
        # A diffusion checkpoint: it has no tokenizer, no context and no chat
        # template, so none of the language metadata below applies.
        meta = ModelMeta(
            arch=str(header.kv.get("general.architecture") or "") or None,
            quant=parse_quant(p.name) or ftype_to_quant(header.kv.get("general.file_type")),
            image_family=image_family,
        )
        return meta
    n_params = header.n_params
    n_expert = header.n_expert_params
    split_count = _as_int(header.kv.get("split.count")) or (shard[2] if shard else 1)
    if split_count and split_count > 1 and shard:
        for idx in range(2, split_count + 1):
            other = first.with_name(f"{shard[0]}-{idx:05d}-of-{split_count:05d}.gguf")
            if not other.exists():
                n_params = None
                n_expert = None
                break
            try:
                h = read_gguf_header(other, want_tensors=True)
            except MetadataError:
                n_params = None
                n_expert = None
                break
            n_params = (n_params or 0) + (h.n_params or 0)
            n_expert = (n_expert or 0) + (h.n_expert_params or 0)
    return meta_from_gguf_kv(
        header.kv, n_params=n_params, n_expert_params=n_expert, tokens=header.tokens, filename=p.name
    )


# --------------------------------------------------------------------------- #
# MLX / Hugging Face config.json → ModelMeta
# --------------------------------------------------------------------------- #

_EMBED_MODEL_TYPES = frozenset({
    "bert", "nomic_bert", "modernbert", "xlm-roberta", "roberta", "distilbert", "mpnet",
    "jina_bert", "e5", "gte", "new", "neobert", "eurobert",
})


def _load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        log.debug("cannot read %s: %s", path, exc)
        return None


def read_mlx_config(model_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """``config.json`` of an MLX/HF model directory (raises MetadataError)."""
    d = Path(model_dir)
    cfg = _load_json(d / "config.json")
    if not isinstance(cfg, dict):
        raise MetadataError(f"Brak poprawnego pliku config.json w katalogu {d.name}.")
    return cfg


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    tc = config.get("text_config")
    if isinstance(tc, dict) and tc:
        merged = dict(config)
        merged.update({k: v for k, v in tc.items() if v is not None})
        return merged
    return config


def _first(cfg: dict[str, Any], *keys: str) -> Any:
    for k in keys:
        v = cfg.get(k)
        if v is not None:
            return v
    return None


def mlx_quant_label(config: dict[str, Any]) -> str | None:
    """``{"quantization": {"bits": 4}}`` → ``"4bit"``; mxfp4 → ``"MXFP4"``; bf16 → ``"BF16"``."""
    q = config.get("quantization") or config.get("quantization_config")
    if isinstance(q, dict):
        mode = str(q.get("mode") or "").lower()
        if mode in ("mxfp4", "nvfp4", "mxfp8"):
            return mode.upper()
        bits = _as_int(q.get("bits"))
        if bits:
            return f"{bits}bit"
        method = q.get("quant_method")
        if isinstance(method, str) and method:
            return method.upper()
    dtype = config.get("torch_dtype") or config.get("dtype")
    if not dtype and isinstance(config.get("text_config"), dict):
        dtype = config["text_config"].get("torch_dtype") or config["text_config"].get("dtype")
    if isinstance(dtype, str):
        return {"bfloat16": "BF16", "float16": "F16", "float32": "F32"}.get(dtype.lower(), dtype.upper())
    return None


def _safetensors_header(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "rb") as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                return None
            n = struct.unpack("<Q", raw)[0]
            if n <= 0 or n > 200_000_000:
                return None
            data = fh.read(n)
        header = json.loads(data)
        return header if isinstance(header, dict) else None
    except (OSError, ValueError):
        return None


def _count_mlx_params(model_dir: Path, config: dict[str, Any]) -> tuple[int | None, int | None]:
    """(total, expert) parameter counts from safetensors headers, de-packing quantised weights."""
    q = config.get("quantization") or config.get("quantization_config") or {}
    default_bits = _as_int(q.get("bits")) if isinstance(q, dict) else None
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        return None, None
    headers: dict[str, tuple[str, list[int]]] = {}
    for f in files:
        h = _safetensors_header(f)
        if h is None:
            return None, None
        for name, info in h.items():
            if name == "__metadata__" or not isinstance(info, dict):
                continue
            headers[name] = (str(info.get("dtype", "")), list(info.get("shape") or []))
    total = 0
    expert = 0
    for name, (dtype, shape) in headers.items():
        if name.endswith((".scales", ".biases")):
            continue
        n = math.prod(shape) if shape else 1
        if dtype in ("U32", "U8") and name.endswith(".weight"):
            prefix = name[: -len(".weight")]
            if prefix + ".scales" in headers:
                bits = default_bits or 4
                override = q.get(prefix) if isinstance(q, dict) else None
                if isinstance(override, dict) and _as_int(override.get("bits")):
                    bits = _as_int(override.get("bits")) or bits
                n = n * (32 if dtype == "U32" else 8) // bits
        total += n
        if ".switch_mlp." in name or ".experts." in name:
            expert += n
    return total or None, expert or None


def meta_from_hf_config(
    config: dict[str, Any],
    *,
    n_params: int | None = None,
    n_expert_params: int | None = None,
    name: str | None = None,
) -> ModelMeta:
    """Build :class:`ModelMeta` from a HF/MLX ``config.json`` dict (local or remote)."""
    tc = _text_config(config)
    meta = ModelMeta(arch=_first(config, "model_type") or _first(tc, "model_type"))
    meta.context_length = _as_int(_first(tc, "max_position_embeddings", "max_seq_len", "n_positions",
                                         "seq_length", "max_sequence_length", "n_ctx"))
    n_layers = _as_int(_first(tc, "num_hidden_layers", "n_layers", "num_layers", "n_layer"))
    meta.n_layers = n_layers
    meta.n_heads = _as_int(_first(tc, "num_attention_heads", "n_heads", "num_heads", "n_head"))
    meta.hidden_size = _as_int(_first(tc, "hidden_size", "d_model", "n_embd", "dim"))
    meta.vocab_size = _as_int(_first(tc, "vocab_size"))
    kv_heads = _as_int(_first(tc, "num_key_value_heads", "n_kv_heads", "num_kv_heads")) or meta.n_heads
    head_dim = _as_int(_first(tc, "head_dim"))
    if head_dim is None and meta.hidden_size and meta.n_heads:
        head_dim = meta.hidden_size // meta.n_heads

    # layer layout ------------------------------------------------------------------------
    window = _as_int(tc.get("sliding_window"))
    if tc.get("use_sliding_window") is False:
        window = None
    meta.sliding_window = window if window and window > 0 else None
    is_swa: list[bool] | None = None
    is_recurrent: list[bool] | None = None
    has_full_marker = False
    layer_types = tc.get("layer_types")
    if n_layers and isinstance(layer_types, list) and layer_types:
        lt = [str(t).lower() for t in layer_types][:n_layers]
        lt += [lt[-1]] * (n_layers - len(lt))
        is_swa = [("sliding" in t or "chunked" in t or t == "local") for t in lt]
        is_recurrent = [
            any(s in t for s in ("linear", "mamba", "conv", "ssm", "recurrent", "deltanet", "rwkv"))
            for t in lt
        ]
        has_full_marker = True
    elif n_layers and isinstance(tc.get("full_attn_idxs"), list):
        idxs = {int(i) for i in tc["full_attn_idxs"] if _as_int(i) is not None}
        is_recurrent = [i not in idxs for i in range(n_layers)]
    elif n_layers and _as_int(tc.get("full_attention_interval")):
        interval = _as_int(tc.get("full_attention_interval")) or 1
        is_recurrent = [((i + 1) % interval) != 0 for i in range(n_layers)]
    elif n_layers and isinstance(tc.get("hybrid_override_pattern"), str):
        pat = tc["hybrid_override_pattern"]
        is_recurrent = [(pat[i] != "*") if i < len(pat) else True for i in range(n_layers)]
    elif n_layers and meta.sliding_window and _as_int(tc.get("sliding_window_pattern")):
        period = _as_int(tc.get("sliding_window_pattern")) or 1
        if period > 1:
            is_swa = [(i % period) != (period - 1) for i in range(n_layers)]
    elif n_layers and meta.arch in ("mamba", "mamba2", "rwkv7", "rwkv6"):
        is_recurrent = [True] * n_layers
    if is_swa is not None and not any(is_swa):
        is_swa = None
    # A plain `sliding_window` without a layer pattern (Qwen2/Mistral/Phi-3 configs) is
    # applied by mlx-lm to no layer or to every layer; stay conservative: all layers full.

    shared = _as_int(_first(tc, "num_kv_shared_layers")) or 0
    n_full, _ = _kv_layout(
        n_layers=n_layers,
        kv_heads=[kv_heads or 0] * n_layers if (n_layers and kv_heads) else None,
        is_swa=is_swa, is_recurrent=is_recurrent, shared_kv_layers=shared,
    )
    meta.n_full_attn_layers = n_full
    # gemma4-style: global (full) layers use their own kv-head count / head size
    global_heads = _as_int(tc.get("num_global_key_value_heads"))
    global_dim = _as_int(tc.get("global_head_dim"))
    if (is_swa is not None or has_full_marker) and (global_heads or global_dim):
        kv_heads = global_heads or kv_heads
        head_dim = global_dim or head_dim
    meta.n_kv_heads = kv_heads
    meta.head_dim = head_dim
    kv_lora = _as_int(tc.get("kv_lora_rank"))
    if kv_lora:
        rope = _as_int(tc.get("qk_rope_head_dim")) or 64
        meta.n_kv_heads = 1
        meta.head_dim = math.ceil((kv_lora + rope) / 2)

    # parameters ------------------------------------------------------------------------
    label_total, label_active = parse_size_label(name)
    meta.params_b = _round_b(n_params) if n_params else (round(label_total, 3) if label_total else None)
    n_experts = _as_int(_first(tc, "num_experts", "n_routed_experts", "num_local_experts"))
    per_tok = _as_int(_first(tc, "num_experts_per_tok", "experts_per_token", "top_k_experts", "moe_topk"))
    if label_active:
        meta.active_params_b = round(label_active, 3)
    elif n_params and n_expert_params and n_experts and per_tok:
        meta.active_params_b = _round_b(n_params - n_expert_params + n_expert_params * per_tok / n_experts)

    meta.quant = mlx_quant_label(config)
    archs = [str(a) for a in (config.get("architectures") or []) if isinstance(a, str)]
    encoder_class = any(
        ("Embedding" in a or a.endswith(("Model", "ForMaskedLM")))
        and not a.endswith(("ForCausalLM", "ForConditionalGeneration"))
        for a in archs
    )
    lname = (name or "").lower()
    meta.is_embedding = bool(
        str(meta.arch or "").lower() in _EMBED_MODEL_TYPES
        or encoder_class
        or ("embed" in lname and "embedded" not in lname)
    )
    return meta


def read_hf_chat_template(model_dir: str | os.PathLike[str]) -> str | None:
    """Chat template from ``chat_template.jinja`` > ``chat_template.json`` > ``tokenizer_config.json``."""
    d = Path(model_dir)
    try:
        t = (d / "chat_template.jinja").read_text(encoding="utf-8")
        if t.strip():
            return t
    except (OSError, UnicodeDecodeError):
        pass
    data = _load_json(d / "chat_template.json")
    if isinstance(data, dict) and isinstance(data.get("chat_template"), str):
        return data["chat_template"]
    tok = _load_json(d / "tokenizer_config.json")
    if isinstance(tok, dict):
        t = tok.get("chat_template")
        if isinstance(t, str) and t.strip():
            return t
        if isinstance(t, list) and t:
            named = {str(x.get("name")): x.get("template") for x in t if isinstance(x, dict)}
            chosen = named.get("default") or next((v for v in named.values() if isinstance(v, str)), None)
            if isinstance(chosen, str):
                return chosen
    return None


def _token_str(v: Any) -> str | None:
    if isinstance(v, str):
        return v
    if isinstance(v, dict) and isinstance(v.get("content"), str):
        return v["content"]
    return None


def read_mlx_meta(model_dir: str | os.PathLike[str]) -> ModelMeta:
    """Metadata of an MLX model directory (config.json + tokenizer files)."""
    d = Path(model_dir)
    config = read_mlx_config(d)
    n_params: int | None = None
    n_expert: int | None = None
    index = _load_json(d / "model.safetensors.index.json")
    if isinstance(index, dict) and isinstance(index.get("metadata"), dict):
        n_params = _as_int(index["metadata"].get("total_parameters"))
    counted, expert = _count_mlx_params(d, config)
    if n_params is None:
        n_params = counted
    n_expert = expert
    meta = meta_from_hf_config(config, n_params=n_params, n_expert_params=n_expert, name=d.name)
    meta.chat_template = read_hf_chat_template(d)
    tok = _load_json(d / "tokenizer_config.json")
    special = _load_json(d / "special_tokens_map.json")
    for attr, key in (("bos_token", "bos_token"), ("eos_token", "eos_token")):
        value = None
        if isinstance(tok, dict):
            value = _token_str(tok.get(key))
        if value is None and isinstance(special, dict):
            value = _token_str(special.get(key))
        setattr(meta, attr, value)
    if not meta.is_embedding and (d / "modules.json").exists() and (d / "sentence_bert_config.json").exists():
        meta.is_embedding = True
    return meta


# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #

_TOOLS_RE = re.compile(r"\btools\b")
_THINK_SWITCH_RE = re.compile(
    r"enable_thinking|reasoning_effort|thinking_budget|<\|channel\|>|\[THINK\]|<\|think\|>|<seed:think>"
)
_THINK_GEN_RE = re.compile(r"add_generation_prompt[\s\S]{0,400}?<think>")
_THINK_NAME_RE = re.compile(
    r"(?i)(thinking|reasoning|(?:^|[-_/ .])r1(?:$|[-_/ .])|qwq|magistral|gpt-oss|deepseek-r1)"
)


def template_supports_tools(template: str | None) -> bool:
    return bool(template) and _TOOLS_RE.search(template or "") is not None


def template_supports_thinking(template: str | None, name: str | None = None) -> bool:
    """Heuristic: a thinking switch, harmony channels, or ``<think>`` in the generation prompt.

    A bare ``<think>`` inside the history-rendering code (Qwen3-*-Instruct-2507,
    Qwen3-Coder strip old reasoning) does **not** count.
    """
    if template and (_THINK_SWITCH_RE.search(template) or _THINK_GEN_RE.search(template)):
        return True
    return bool(name) and _THINK_NAME_RE.search(name or "") is not None


# --------------------------------------------------------------------------- #
# Image models (text-to-image): families, companions, layouts
# --------------------------------------------------------------------------- #

#: ``ModelMeta.image_family`` values (mirrors ``localcode.images.base.IMAGE_FAMILIES``).
IMAGE_FAMILIES = ("flux", "flux2", "qwen-image", "sd1", "sdxl", "sd3", "unknown")

# Files that belong to a diffusion model but are not models on their own: text
# encoders (CLIP-L/CLIP-G/T5-XXL/UMT5), autoencoders and tiny preview decoders.
# They ship next to split FLUX/SD3 checkpoints (second-state/*, city96/*) and
# must never show up in the library as separate entries.
_IMAGE_COMPANION_RE = re.compile(
    r"(?i)(?:^|[-._/])"
    r"(clip[-._]?[lgh]|clip[-._]?vision|open[-._]?clip|t5xxl|t5[-._]?v1|umt5|"
    r"text[-._]?encoder(?:[-._]?\d)?|tokenizer|taesd\w*|vae(?:[-._]?ft[-._]?\w+)?|"
    r"ae|autoencoder|diffusion[-._]?pytorch[-._]?model)"
    r"(?:$|[-._])"
)

# Which component a companion file provides (``ModelMeta.image_components`` key).
_COMPONENT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("clip_l", re.compile(r"(?i)(?:^|[-._])clip[-._]?l(?:$|[-._])")),
    ("clip_g", re.compile(r"(?i)(?:^|[-._])clip[-._]?g(?:$|[-._])")),
    ("t5xxl", re.compile(r"(?i)(?:^|[-._])(?:t5xxl|t5[-._]?v1|umt5)(?:$|[-._])")),
    ("taesd", re.compile(r"(?i)(?:^|[-._])taesd")),
    ("vae", re.compile(r"(?i)(?:^|[-._])(?:vae|ae|autoencoder)(?:$|[-._])")),
)

# Names that identify a family without looking inside the file.
_FAMILY_NAME_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("flux2", re.compile(r"(?i)flux[-._]?2|flux2|klein")),
    ("qwen-image", re.compile(r"(?i)qwen[-._]?image")),
    ("flux", re.compile(r"(?i)flux")),
    ("sd3", re.compile(r"(?i)sd[-._]?3|stable[-._]?diffusion[-._]?3")),
    ("sdxl", re.compile(r"(?i)sd[-._]?xl|stable[-._]?diffusion[-._]?xl|xl[-._]?base|xl[-._]?refiner|"
                        r"illustrious|pony[-._]?diffusion|juggernaut[-._]?xl|playground[-._]?v2")),
    ("sd1", re.compile(r"(?i)sd[-._]?1|stable[-._]?diffusion[-._]?v?1|v1[-._]?5|"
                       r"dreamshaper|realistic[-._]?vision|anything[-._]?v\d")),
)

# Tensor-name evidence that a GGUF holds a diffusion model (these files usually
# carry **no** KV metadata at all — verified on
# second-state/stable-diffusion-v1-5-GGUF, whose header has 0 KV pairs and the
# prefixes model.diffusion_model.*, cond_stage_model.*, first_stage_model.*).
_DIFFUSION_TENSOR_MARKERS = (
    "model.diffusion_model.", "first_stage_model.", "cond_stage_model.",
    "double_blocks.", "single_blocks.", "joint_blocks.", "input_blocks.",
    "output_blocks.", "middle_block.", "time_embed.", "unet.", "conditioner.embedders.",
    "img_in.", "txt_in.", "x_embedder.", "transformer_blocks.",
)
_FLUX_TENSOR_MARKERS = ("double_blocks.", "single_blocks.", "img_in.", "txt_in.")
_SD3_TENSOR_MARKERS = ("joint_blocks.", "context_embedder.")
_SDXL_TENSOR_MARKERS = ("conditioner.embedders.1.", "label_emb.")
_UNET_TENSOR_MARKERS = ("input_blocks.", "output_blocks.", "middle_block.", "time_embed.")

#: How many tensor names the image probe reads (the first ones already decide).
IMAGE_TENSOR_PROBE = 256

#: GGUF architectures written by diffusion converters.
_IMAGE_ARCHS = frozenset({"sd", "sd1", "sdxl", "sd3", "flux", "unet", "diffusion", "stable-diffusion"})

#: Video diffusion models look exactly like image ones to the probe above — a
#: LTX-Video GGUF carries ``transformer_blocks.`` just like FLUX does — so
#: without this they end up on the "Obrazy" shelf and the image backend dies on
#: them with "NULL pointer access". They belong to ``localcode.video``
#: (``classify_video_model``), which recognises them by name the same way.
_VIDEO_ARCHS = frozenset({"ltxv", "ltx-video", "wan", "wanx", "hunyuan-video", "hyvideo",
                          "mochi", "cogvideox", "svd", "stable-video-diffusion"})
_VIDEO_NAME_RE = re.compile(
    r"(ltx[\W_]*video|ltxv|wan\s*2|wan2\.|hunyuan[\W_]*video|cogvideo|mochi[\W_]*1|"
    r"stable[\W_]*video[\W_]*diffusion|animatediff|svd[\W_]*xt)", re.I)


def is_video_checkpoint(arch: str | None = None, filename: str | None = None) -> bool:
    """True for a *video* diffusion checkpoint (LTX-Video, Wan, HunyuanVideo…).

    Separate from :func:`gguf_image_family` on purpose: the tensor layout of a
    video transformer is indistinguishable from an image one at this depth, so
    the only honest signals are the GGUF architecture key and the file name.
    """
    if str(arch or "").strip().lower() in _VIDEO_ARCHS:
        return True
    return bool(filename and _VIDEO_NAME_RE.search(str(filename)))


def is_image_companion(filename: str) -> bool:
    """True for a CLIP/T5/VAE/tokenizer file that belongs to a diffusion model.

    Such files are loaded *through* the main model (``ModelMeta.image_components``)
    and must not be listed as models of their own.
    """
    base = os.path.basename(str(filename))
    stem = base.rsplit(".", 1)[0] if "." in base else base
    if is_mmproj(base):
        return False                     # vision projectors have their own handling
    return bool(_IMAGE_COMPANION_RE.search(stem))


def image_component_kind(filename: str) -> str | None:
    """``clip_l`` / ``clip_g`` / ``t5xxl`` / ``vae`` / ``taesd`` for a companion file."""
    base = os.path.basename(str(filename))
    stem = base.rsplit(".", 1)[0] if "." in base else base
    for kind, pattern in _COMPONENT_PATTERNS:
        if pattern.search(stem):
            return kind
    return None


def image_family_from_name(name: str | None) -> str | None:
    """Family guessed from a repo id / file name (``…sd3.5_medium-Q4_0`` → ``sd3``)."""
    text = str(name or "")
    if not text:
        return None
    for family, pattern in _FAMILY_NAME_PATTERNS:
        if pattern.search(text):
            return family
    return None


def gguf_image_family(kv: dict[str, Any], tensor_names: Iterable[str] = (),
                      filename: str | None = None) -> str | None:
    """Image family of a GGUF file, or ``None`` when it is not a diffusion model.

    Diffusion GGUFs are recognised by their tensor names (they carry no KV
    metadata); the concrete family comes from the names when they are
    unambiguous (FLUX blocks, SD3 joint blocks, SDXL's second text encoder) and
    otherwise from the file/repo name.
    """
    arch = str(kv.get("general.architecture") or "").lower()
    names = list(tensor_names)
    joined = "\n".join(names)
    diffusion = arch in _IMAGE_ARCHS or any(m in joined for m in _DIFFUSION_TENSOR_MARKERS)
    if not diffusion:
        return None
    if is_video_checkpoint(arch, filename):
        return None          # wideo, nie obraz — patrz localcode.video
    if any(m in joined for m in _FLUX_TENSOR_MARKERS):
        # FLUX.1 and FLUX.2 share the block layout; only the name tells them apart.
        return "flux2" if image_family_from_name(filename) == "flux2" else "flux"
    if any(m in joined for m in _SD3_TENSOR_MARKERS):
        return "sd3"
    by_name = image_family_from_name(filename)
    if by_name:
        return by_name
    if any(m in joined for m in _SDXL_TENSOR_MARKERS):
        return "sdxl"
    if any(m in joined for m in _UNET_TENSOR_MARKERS) or "model.diffusion_model." in joined:
        return "sd1"
    return "unknown"


def is_image_model_dir(model_dir: str | os.PathLike[str]) -> bool:
    """True for an MLX/diffusers diffusion checkpoint directory.

    The layout of every mflux/MLX image repo (``Runpod/FLUX.2-klein-4B-mflux-4bit``,
    ``mlx-community/Qwen-Image-2512-4bit``, ``dhairyashil/FLUX.1-schnell-mflux-4bit``)
    is ``transformer/`` + ``vae/`` + ``text_encoder*/`` sub-directories with the
    weights inside, which is why the plain MLX scanner (top-level safetensors)
    never sees them.
    """
    d = Path(model_dir)
    if not d.is_dir():
        return False
    if is_video_checkpoint(filename=d.name):
        return False         # wideo, nie obraz — patrz localcode.video
    if (d / "model_index.json").is_file():
        return True
    core = next((sub for sub in ("transformer", "unet", "diffusion_model")
                 if _has_weights(d / sub)), None)
    if core is None:
        return False
    return _has_weights(d / "vae") or any(
        _has_weights(d / name) for name in ("text_encoder", "text_encoder_2"))


def _has_weights(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    try:
        return any(f.is_file() and not f.name.startswith(".")
                   for f in directory.glob("*.safetensors"))
    except OSError:  # pragma: no cover - unreadable directory
        return False


def read_image_dir_meta(model_dir: str | os.PathLike[str], repo_id: str | None = None) -> ModelMeta:
    """``ModelMeta`` of an MLX diffusion checkpoint directory.

    Only what the image runtime needs: the family, the quantisation label and
    the component directories (so the UI can show what is on disk).
    """
    d = Path(model_dir)
    config = read_mlx_config(d)
    meta = ModelMeta()
    name = f"{repo_id or ''} {d.name}"
    meta.image_family = image_family_from_name(name) or _family_from_image_config(config) or "unknown"
    meta.arch = str(config.get("model_type") or config.get("_class_name") or "") or None
    meta.quant = mlx_quant_label(config) or _quant_from_transformer(d)
    components: dict[str, str] = {}
    for key, sub in (("unet", "transformer"), ("unet", "unet"), ("vae", "vae"),
                     ("clip_l", "text_encoder"), ("t5xxl", "text_encoder_2")):
        path = d / sub
        if key not in components and _has_weights(path):
            components[key] = str(path)
    meta.image_components = components
    return meta


def _family_from_image_config(config: dict[str, Any]) -> str | None:
    text = " ".join(str(config.get(k) or "") for k in ("_class_name", "model_type", "architecture"))
    return image_family_from_name(text)


def _quant_from_transformer(model_dir: Path) -> str | None:
    """Quantisation of an mflux checkpoint, read from the transformer's config."""
    for sub in ("transformer", "unet"):
        config = _load_json(model_dir / sub / "config.json")
        if isinstance(config, dict):
            label = mlx_quant_label(config)
            if label:
                return label
    return None


def detect_image_family(
    *,
    meta: ModelMeta | None = None,
    name: str | None = None,
    files: Iterable[str] = (),
) -> str | None:
    """Best-effort family for a model that is already known to be an image model."""
    if meta is not None and meta.image_family:
        return meta.image_family
    by_name = image_family_from_name(name)
    if by_name:
        return by_name
    for f in files:
        by_file = image_family_from_name(os.path.basename(str(f)))
        if by_file:
            return by_file
    return None


#: Architectures and config markers of speech models, by direction.
_STT_MARKERS = ("whisper", "parakeet", "conformer", "wav2vec", "hubert", "moonshine",
                "sensevoice", "canary", "nemo_asr", "transcribe")
_TTS_MARKERS = ("kokoro", "orpheus", "bark", "vits", "xtts", "styletts", "csm", "outetts",
                "fish-speech", "f5-tts", "spark-tts", "neutts", "piper", "melotts")


def detect_speech(meta: Any, files: Iterable[str] = (), *,
                  config: dict[str, Any] | None = None, name: str | None = None) -> str | None:
    """``"stt"``, ``"tts"`` or ``None`` for a speech model.

    Recognised by architecture and by the shape of the config — a Whisper port
    has ``n_mels`` and ``n_audio_layer`` and nothing an LLM would have — because
    these checkpoints carry no chat template to read.
    """
    haystack = " ".join(filter(None, [str(getattr(meta, "arch", "") or ""),
                                      str(name or ""), " ".join(str(f) for f in files)])).lower()
    model_type = str((config or {}).get("model_type") or "").lower()
    if model_type:
        haystack += " " + model_type
    if any(marker in haystack for marker in _TTS_MARKERS):
        return "tts"
    if any(marker in haystack for marker in _STT_MARKERS):
        return "stt"
    if config and {"n_mels", "n_audio_layer"} <= set(config):
        return "stt"          # a Whisper-shaped config, whatever it is called
    return None


def detect_capabilities(
    meta: ModelMeta,
    files: Iterable[str] = (),
    *,
    config: dict[str, Any] | None = None,
    name: str | None = None,
) -> list[str]:
    """Capabilities in canonical order: ``tools``, ``thinking``, ``vision``, ``embedding``.

    * ``image`` — text-to-image models (nothing else is reported for them);
    * ``embedding`` — embedding/reranker models (nothing else is reported for them);
    * ``tools`` — the chat template renders a ``tools`` list natively;
    * ``thinking`` — see :func:`template_supports_thinking`;
    * ``vision`` — a GGUF projector (``mmproj*.gguf``) among ``files`` or a
      ``vision_config`` in the MLX ``config`` (the checkpoint contains a vision tower).
    """
    if meta.image_family:
        return ["image"]
    if meta.is_embedding:
        return ["embedding"]
    speech = detect_speech(meta, files, config=config, name=name)
    if speech:
        return [speech]
    caps: list[str] = []
    if template_supports_tools(meta.chat_template):
        caps.append("tools")
    if template_supports_thinking(meta.chat_template, name):
        caps.append("thinking")
    has_projector = any(is_mmproj(f) for f in files)
    has_vision_tower = isinstance(config, dict) and isinstance(config.get("vision_config"), dict) and bool(config["vision_config"])
    if has_projector or has_vision_tower:
        caps.append("vision")
    return caps
