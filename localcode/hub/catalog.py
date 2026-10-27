"""In-app model store: a friendly presentation layer over the hub.

The hub modules (:mod:`localcode.hub.search`, :mod:`localcode.hub.curated`,
:mod:`localcode.hub.library`) speak in repository ids, quantisation labels and
file lists.  That is the *advanced* view.  This module turns the same data into
something a person who has never heard of Hugging Face can act on:

* a clean display name (``"Qwen3 Coder 30B"``) instead of
  ``unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF``,
* one Polish line saying what the model is for,
* a category (``kodowanie``, ``czat``, ``rozumowanie``, ``obrazy``, ``wizja``,
  ``embedding``),
* whether it fits into *this* Mac's memory budget,
* and exactly one action: install — the recommended variant is chosen here, not
  by the user.

The provenance is never hidden, only demoted: every entry carries
``source = {"provider": "Hugging Face", "repo": …}`` and the UI shows the
:data:`SOURCE_NOTE` footer.

Design notes
------------
* Search results are cached for :data:`SEARCH_TTL_S` so switching a filter in
  the UI feels instant.
* Building an entry must never hit the network per repository: sizes come from
  the curated table or from a parameter-count estimate.  Real variants (file
  names, exact bytes) are fetched only when the user installs, or when
  :meth:`Catalog.detail` is called for the "Szczegóły" sheet.
* Offline is a normal state, not an error: :meth:`Catalog.entries` falls back to
  the staff picks plus what is already on disk and sets :attr:`Catalog.notice`.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
import unicodedata
from typing import Any, Literal, Sequence, TypedDict

from localcode.hub import store_data
from localcode.hub.curated import (
    CURATED_IMAGE_REPOS,
    CURATED_REPOS,
    curated_entry,
    curated_models,
)
from localcode.hub.metadata import quant_warning
from localcode.hub.publishers import OFFICIAL_PUBLISHERS
from localcode.hub.trust import Trust

#: Display names of the laboratories, as a trusted conversion's index row
#: carries them in ``publisher``.
_LAB_NAMES = frozenset(OFFICIAL_PUBLISHERS.values())

__all__ = [
    "SOURCE_NOTE",
    "strip_quant_phrases",
    "OFFLINE_NOTE",
    "CATEGORIES",
    "SORTS",
    "SORT_LABELS",
    "QUALITY_LEVELS",
    "QUALITY_LABELS",
    "quant_for_quality",
    "CATEGORY_LABELS",
    "CatalogEntry",
    "Catalog",
    "CatalogError",
    "catalog_entries",
    "fmt_params",
    "params_label",
    "install",
    "resolve_name",
    "parse_repo_ref",
    "display_name",
    "catalog_settings_dto",
]

#: Shown in the catalogue footer and in the settings screen.
SOURCE_NOTE = "Modele pobierane z Hugging Face"
#: Shown instead of an error when the catalogue cannot reach the network.
OFFLINE_NOTE = (
    "Brak połączenia z katalogiem online — pokazujemy modele polecane "
    "i te, które masz już na dysku."
)

#: User-facing category ids (Polish, stable — the Swift client switches on them).
CATEGORIES = ("kodowanie", "czat", "rozumowanie", "obrazy", "wizja", "embedding",
              "mowa", "wideo")

#: How the store can be ordered.  "polecane" is the store's own judgement —
#: installed first, then staff picks, then what fits this Mac; the others are
#: plain orderings on one number, because a person who picks "najwięcej
#: polubień" wants that list, not our opinion of it.
SORTS = ("polecane", "popularne", "lubiane", "zaufanie", "oficjalne", "najwieksze", "najmniejsze")
SORT_LABELS: dict[str, str] = {
    "polecane": "Polecane",
    "popularne": "Najpopularniejsze",
    "lubiane": "Najwięcej polubień",
    "zaufanie": "Najwyższe zaufanie",
    "oficjalne": "Od wydawców modeli",
    "najwieksze": "Największe, które się zmieszczą",
    "najmniejsze": "Najmniejsze",
}

#: A laboratory's release has to be used by somebody before it outranks a
#: popular conversion: some organisations publish dozens of experiments.
LAB_RELEASE_MIN_DOWNLOADS = 1000
#: Fine-tunes that strip a model's safety training or merge several models: the
#: base model's laboratory did not make them, whatever the metadata says.
_DERIVATIVE_RE = re.compile(
    r"abliterat|uncensor|unredact|heretic|derestrict|nsfw|lewd|erotic|roleplay|(^|[-_ ])rp([-_ ]|$)|"
    r"merge|blend",
    re.IGNORECASE)


def lab_release(repo: str, *, provenance: str | None, lab_known: bool,
                downloads: int) -> bool:
    """The laboratory's own model: its repository, or a faithful conversion of it.

    "Polecane" puts these ahead of community fine-tunes — a search for "qwen"
    used to open on abliterated and "uncensored" remixes, with the first of
    Qwen's own repositories in fifteenth place.
    """
    if _DERIVATIVE_RE.search(repo.partition("/")[2]):
        return False
    if provenance == "official":
        return downloads >= LAB_RELEASE_MIN_DOWNLOADS
    return provenance == "trusted" and lab_known
CategoryId = Literal["kodowanie", "czat", "rozumowanie", "obrazy", "wizja", "embedding",
                     "mowa", "wideo"]

#: ``curated.category`` / hub tags → catalogue category.
_CATEGORY_FROM_CURATED = {
    "code": "kodowanie",
    "chat": "czat",
    "reasoning": "rozumowanie",
    "vision": "wizja",
    "embedding": "embedding",
    "image": "obrazy",
    "speech": "mowa",
    "video": "wideo",
}
CATEGORY_LABELS: dict[str, str] = {
    "kodowanie": "Kodowanie",
    "czat": "Czat",
    "rozumowanie": "Rozumowanie",
    "obrazy": "Obrazy",
    "wizja": "Wizja",
    "embedding": "Wyszukiwanie w dokumentach",
    "mowa": "Mowa",
    "wideo": "Wideo",
}

FIT_OK = "Zmieści się"
FIT_TIGHT = "Na styk"
FIT_NO = "Za duży"
FIT_UNKNOWN = "Nie wiadomo"

#: ``MemoryEstimate.fits`` → catalogue verdict.
_FIT_FROM_ESTIMATE = {"yes": FIT_OK, "tight": FIT_TIGHT, "no": FIT_NO}

#: Extra memory a loaded model needs beyond its weights (KV cache + overhead).
LOAD_OVERHEAD = 1.25
#: How long a search result set stays fresh.
SEARCH_TTL_S = 300.0
#: How many hub hits we ask for when the user is just browsing.
DEFAULT_LIMIT = 30


class CatalogError(Exception):
    """Catalogue failure with a Polish, user-presentable message."""

    def __init__(self, message: str, code: str = "catalog_error", status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


class CatalogVariant(TypedDict, total=False):
    """One installable option; hidden behind "Zaawansowane" in the UI."""

    id: str
    label: str
    sizeBytes: int
    quant: str | None
    files: list[str]
    recommended: bool
    fit: str
    fitDetail: str


class CatalogSource(TypedDict):
    provider: str
    repo: str


class CatalogEntry(TypedDict, total=False):
    """One row of the in-app store."""

    id: str                       # repository id (also the install id)
    name: str                     # "Qwen3 Coder 30B"
    tagline: str                  # one Polish line: what it is for
    category: str                 # one of CATEGORIES
    format: str                   # "gguf" | "mlx"
    quant: str | None             # the variant an install would take ("Q4_K_M", "4bit")
    quants: list[str]             # every quantisation the repository offers
    sizeBytes: int | None
    sizeLabel: str                # "ok. 18,5 GB"
    fit: str                      # FIT_OK | FIT_TIGHT | FIT_NO | FIT_UNKNOWN
    fitDetail: str                # Polish explanation of `fit`
    speedHint: str                # "Bardzo szybki", "Szybki", ...
    installed: bool
    installedKey: str | None      # LocalModel.key when installed
    recommendedVariant: CatalogVariant | None
    variants: list[CatalogVariant]
    badges: list[str]             # ["narzędzia", "myślenie", "wizja", "kontekst 256k"]
    alternatives: list[CatalogSource]   # same model, other packaging (advanced view)
    source: CatalogSource
    requiresToken: bool
    curated: bool
    trust: dict[str, Any]         # score 0-100 + Polish reasons (localcode.hub.trust)
    publisher: dict[str, Any]     # who published it, and whether that is the laboratory
    downloads: int | None         # last-month downloads, when known
    likes: int | None
    paramsB: float | None         # total parameters in billions
    activeParamsB: float | None   # of which active per token (MoE), when known
    paramsLabel: str              # "30 mld, aktywne 3 mld" — ready for the UI
    watchOut: str | None          # one Polish caveat ("model bez filtrów"), when any
    external: bool                # pasted repository, not part of our catalogue


# --------------------------------------------------------------------------- #
# Friendly names
# --------------------------------------------------------------------------- #

#: Publisher prefixes/suffixes and packaging noise stripped from repo names.
_NOISE_TOKENS = frozenset({
    "gguf", "mlx", "mflux", "safetensors", "quantized", "quant", "community",
    "4bit", "8bit", "6bit", "5bit", "3bit", "2bit", "16bit", "bf16", "fp16", "fp8",
    "mxfp4", "q8", "q80", "q4", "q40", "q5", "q6", "ud", "hf", "base",
})
_QUANT_RE = re.compile(r"^(?:ud-)?q\d(?:_[a-z0-9]+)*$", re.IGNORECASE)
#: A quantisation label written with dashes (``-GGUF-Q4_K_M``, ``-IQ4-XS``) is not
#: one token, so removing tokens alone left "K M" at the end of the name.  The
#: label is therefore cut out as a phrase before the name is split.
_QUANT_PHRASE_RE = re.compile(
    r"(?:^|(?<=[-_\s]))(?:ud-|i1-)?(?:iq|q)\d+(?:[-_](?:k|s|m|l|xs|xl|xxs|xxl|nl|0|1))*"
    r"(?=$|[-_\s])", re.IGNORECASE)
_SIZE_RE = re.compile(r"^(a?\d+(?:\.\d+)?)([bm])$", re.IGNORECASE)
_MOE_RE = re.compile(r"^e\d+b$", re.IGNORECASE)
_DATE_RE = re.compile(r"^(?:20)?\d{4}$")

#: Tokens that read better in their canonical spelling.
_WORD_MAP = {
    "qwen": "Qwen", "qwen3": "Qwen3", "qwen2": "Qwen2", "gemma": "Gemma",
    "llama": "Llama", "mistral": "Mistral", "ministral": "Ministral",
    "devstral": "Devstral", "granite": "Granite", "nemotron": "Nemotron",
    "phi": "Phi", "glm": "GLM", "flux": "FLUX", "sd": "SD", "sdxl": "SDXL",
    "vl": "VL", "moe": "MoE", "oss": "OSS", "gpt": "GPT", "nvidia": "NVIDIA",
    "ibm": "IBM", "coder": "Coder", "instruct": "Instruct", "chat": "Chat",
    "reasoning": "Reasoning", "thinking": "Thinking", "embedding": "Embedding",
    "embed": "Embed", "text": "Text", "image": "Image", "nano": "Nano",
    "flash": "Flash", "small": "Small", "medium": "Medium", "large": "Large",
    "klein": "klein", "schnell": "schnell", "dev": "dev", "krea": "Krea",
    "stable": "Stable", "diffusion": "Diffusion", "xl": "XL", "next": "Next",
    "it": "Instruct", "lfm2": "LFM2", "liquidai": "LiquidAI",
}

#: Hand-written names where the derivation would read badly.
NAME_OVERRIDES: dict[str, str] = {
    "ggml-org/gpt-oss-20b-GGUF": "GPT-OSS 20B",
    "mlx-community/gpt-oss-20b-MXFP4-Q8": "GPT-OSS 20B",
    "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF": "Qwen3 Coder 30B",
    "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit": "Qwen3 Coder 30B",
    "unsloth/Qwen3-Coder-Next-GGUF": "Qwen3 Coder Next 80B",
    "unsloth/Qwen3.6-35B-A3B-GGUF": "Qwen3.6 35B",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "Gemma 4 26B",
    "unsloth/gemma-4-E4B-it-GGUF": "Gemma 4 E4B",
    "lmstudio-community/gemma-4-E4B-it-MLX-4bit": "Gemma 4 E4B",
    "lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4bit": "Nemotron 3 Nano 30B",
    "LiquidAI/LFM2.5-1.2B-Instruct-GGUF": "LFM2.5 1.2B",
    "nomic-ai/nomic-embed-text-v1.5-GGUF": "Nomic Embed Text 1.5",
    "ggml-org/embeddinggemma-300M-GGUF": "EmbeddingGemma 300M",
    "Qwen/Qwen3-Embedding-0.6B-GGUF": "Qwen3 Embedding 0.6B",
    "second-state/stable-diffusion-v1-5-GGUF": "Stable Diffusion 1.5",
    "gpustack/stable-diffusion-xl-base-1.0-GGUF": "Stable Diffusion XL",
    "second-state/stable-diffusion-3.5-medium-GGUF": "Stable Diffusion 3.5 Medium",
    "Runpod/FLUX.2-klein-4B-mflux-4bit": "FLUX.2 klein 4B",
    "mlx-community/flux2-klein-9b-4bit": "FLUX.2 klein 9B",
    "dhairyashil/FLUX.1-schnell-mflux-4bit": "FLUX.1 schnell",
    "second-state/FLUX.1-schnell-GGUF": "FLUX.1 schnell",
    "filipstrand/FLUX.1-Krea-dev-mflux-4bit": "FLUX.1 Krea dev",
    "second-state/FLUX.1-dev-GGUF": "FLUX.1 dev",
    "mlx-community/Qwen-Image-2512-4bit": "Qwen Image",
    "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF": "Devstral Small 2 24B",
    "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit": "Devstral Small 2 24B",
    "lmstudio-community/Ministral-3-14B-Reasoning-2512-GGUF": "Ministral 3 14B Reasoning",
    "unsloth/GLM-4.7-Flash-GGUF": "GLM 4.7 Flash",
}

#: Extra search words that should find an entry ("flux" → every FLUX model).
ALIASES: dict[str, tuple[str, ...]] = {
    "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF": ("qwen coder", "coder 30b"),
    "ggml-org/gpt-oss-20b-GGUF": ("gptoss", "openai oss", "gpt oss"),
    "unsloth/Qwen3-Coder-Next-GGUF": ("coder next", "qwen coder next"),
}


#: A repository reference somebody pasted: "org/name", a full Hugging Face URL,
#: or one with a branch or file path after it.
_REPO_REF_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)?/?"
    r"(?P<org>[A-Za-z0-9][\w.-]*)/(?P<name>[\w.-]+)"
    r"(?:/(?:tree|blob|resolve)/[^\s]*)?/?(?:[?#].*)?$")


def parse_repo_ref(text: str) -> str | None:
    """``"https://huggingface.co/Qwen/Qwen3-8B/tree/main"`` → ``"Qwen/Qwen3-8B"``.

    Returns ``None`` for anything that is not a repository reference, so a
    normal search query ("qwen coder") is never mistaken for one.
    """
    candidate = (text or "").strip()
    if not candidate or " " in candidate:
        return None
    match = _REPO_REF_RE.match(candidate)
    if not match:
        return None
    org, name = match.group("org"), match.group("name")
    if not org or not name or org.lower() in ("http:", "https:"):
        return None
    return f"{org}/{name}"


def _watch_out(caveat: Any, quant: str | None) -> str | None:
    """The caveat shown on a store card: the row's own, plus the quantisation.

    A 1-bit or 2-bit build deserves a word before the download, not after the
    first nonsense answer.
    """
    parts = [str(caveat).strip() for caveat in (caveat,) if caveat and str(caveat).strip()]
    warning = quant_warning(quant)
    if warning:
        parts.append(warning)
    return " ".join(parts) or None


def strip_quant_phrases(name: str) -> str:
    """Remove quantisation labels, however they are punctuated.

    >>> strip_quant_phrases("Mellum2-12B-Instruct-GGUF-Q4_K_M")
    'Mellum2-12B-Instruct-GGUF'
    """
    return _QUANT_PHRASE_RE.sub("", name).strip("-_ .")


def _clean_tokens(name: str) -> list[str]:
    raw = re.split(r"[-_\s]+", strip_quant_phrases(name))
    out: list[str] = []
    for token in raw:
        if not token:
            continue
        low = token.lower()
        if low in _NOISE_TOKENS or _QUANT_RE.match(low):
            continue
        if _DATE_RE.match(low) and out:          # "2507", "2512" release stamps
            continue
        out.append(token)
    return out


def _pretty_token(token: str) -> str:
    low = token.lower()
    size = _SIZE_RE.match(token)
    if size:
        return f"{size.group(1).upper()}{size.group(2).upper()}"
    if _MOE_RE.match(token):
        return token.upper()
    if low in _WORD_MAP:
        return _WORD_MAP[low]
    # keep names that already carry capitals (FLUX.1, GLM-4.7, Qwen3.5)
    if any(c.isupper() for c in token) or any(c.isdigit() for c in token):
        return token
    return token[:1].upper() + token[1:]


def display_name(repo: str) -> str:
    """A clean, human display name for ``org/name``.

    >>> display_name("unsloth/Qwen3-4B-Instruct-2507-GGUF")
    'Qwen3 4B Instruct'
    """
    repo = (repo or "").strip()
    override = NAME_OVERRIDES.get(repo)
    if override:
        return override
    name = repo.partition("/")[2] or repo
    tokens = _clean_tokens(name)
    if not tokens:
        tokens = [name]
    return " ".join(_pretty_token(t) for t in tokens).strip()


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

#: Bytes per parameter for the usual quantisations (weights only).
_BYTES_PER_PARAM = {
    "q2": 0.36, "q3": 0.45, "q4": 0.58, "q5": 0.68, "q6": 0.80, "q8": 1.06,
    "mxfp4": 0.60, "nvfp4": 0.60, "2bit": 0.30, "3bit": 0.44, "4bit": 0.56, "5bit": 0.68,
    "6bit": 0.80, "8bit": 1.06, "int4": 0.56, "int8": 1.06, "fp8": 1.06,
    "bf16": 2.0, "f16": 2.0, "fp16": 2.0,
}


#: What "4-bit" and friends mean in each packaging, best match first.
QUALITY_LEVELS = ("4bit", "5bit", "6bit", "8bit", "max")
QUALITY_LABELS: dict[str, str] = {
    "4bit": "4-bitowa (najmniejsza, domyślna)",
    "5bit": "5-bitowa",
    "6bit": "6-bitowa",
    "8bit": "8-bitowa (bliska oryginałowi)",
    "max": "Pełna precyzja (największa)",
}
_GGUF_BY_QUALITY: dict[str, tuple[str, ...]] = {
    "4bit": ("Q4_K_M", "UD-Q4_K_XL", "Q4_K_S", "IQ4_XS", "IQ4_NL", "Q4_0", "Q4_1"),
    "5bit": ("Q5_K_M", "Q5_K_S", "Q5_0", "Q5_1"),
    "6bit": ("Q6_K", "Q6_K_L"),
    "8bit": ("Q8_0", "Q8_K"),
    "max": ("F16", "BF16", "F32", "Q8_0"),
}
_MLX_BY_QUALITY: dict[str, tuple[str, ...]] = {
    "4bit": ("4bit", "MXFP4", "NVFP4", "3bit"),
    "5bit": ("6bit", "4bit"),
    "6bit": ("6bit", "8bit"),
    "8bit": ("8bit", "6bit"),
    "max": ("BF16", "F16", "8bit"),
}
#: When the wanted level is missing, walk outwards from it rather than jumping
#: to the extremes: a repository without Q5 should answer Q4 or Q6, not F16.
_NEIGHBOURS = {"4bit": ("5bit", "6bit", "8bit", "max"),
               "5bit": ("4bit", "6bit", "8bit", "max"),
               "6bit": ("5bit", "8bit", "4bit", "max"),
               "8bit": ("6bit", "max", "5bit", "4bit"),
               "max": ("8bit", "6bit", "5bit", "4bit")}


def quant_for_quality(quants: Sequence[str], quality: str, fmt: str = "gguf") -> str | None:
    """The variant label matching a requested quality, or the nearest one.

    >>> quant_for_quality(["Q2_K", "Q4_K_M", "Q8_0"], "4bit")
    'Q4_K_M'
    >>> quant_for_quality(["Q5_K_M", "Q8_0"], "4bit")
    'Q5_K_M'
    """
    if not quants:
        return None
    table = _MLX_BY_QUALITY if fmt == "mlx" else _GGUF_BY_QUALITY
    available = {str(q).upper(): str(q) for q in quants}
    wanted = quality if quality in QUALITY_LEVELS else "4bit"
    for level in (wanted, *_NEIGHBOURS.get(wanted, ())):
        for label in table.get(level, ()):
            if label.upper() in available:
                return available[label.upper()]
    return str(quants[0])


def _bytes_per_param(quant: str | None) -> float:
    q = (quant or "").lower().removeprefix("ud-")
    if q in _BYTES_PER_PARAM:
        return _BYTES_PER_PARAM[q]
    for prefix in ("q2", "q3", "q4", "q5", "q6", "q8"):
        if q.startswith(prefix):
            return _BYTES_PER_PARAM[prefix]
    return _BYTES_PER_PARAM["q4"]


def fmt_gb(n_bytes: float) -> str:
    """``18_500_000_000`` → ``"18,5 GB"`` (Polish decimal comma)."""
    if n_bytes <= 0:
        return "—"
    if n_bytes < 1e9:
        return f"{int(round(n_bytes / 1e6))} MB"
    return f"{n_bytes / 1e9:.1f}".replace(".", ",") + " GB"


def _fold(text: str) -> str:
    """Lower-case, accent-free, alphanumeric-only form used for fuzzy matching."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", stripped.lower()).strip()


def _fmt_ctx(n: int) -> str:
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


def _steps_badge(n: int) -> str:
    """``4`` → ``"4 kroki"`` (Polish plural: 1 krok, 2–4 kroki, 5+ kroków)."""
    tens, units = n % 100, n % 10
    if n == 1:
        form = "krok"
    elif 2 <= units <= 4 and not 12 <= tens <= 14:
        form = "kroki"
    else:
        form = "kroków"
    return f"{n} {form}"


def fmt_params(value: float | None) -> str:
    """``30.0`` → ``"30 mld"``, ``0.6`` → ``"600 mln"`` — Polish, no trailing zeros."""
    if not value or float(value) <= 0:
        return ""
    billions = float(value)
    if billions < 1:
        millions = round(billions * 1000)
        return f"{millions} mln"
    text = f"{billions:.1f}".rstrip("0").rstrip(".")
    return f"{text.replace('.', ',')} mld"


def params_label(params_b: float | None, active_b: float | None) -> str:
    """How big the model is, and how much of it runs for each token.

    A mixture-of-experts model is the reason this is two numbers: "30 mld,
    aktywne 3 mld" is the difference between what it costs in memory and what
    it costs in speed, and without the second number a 30B MoE looks like a
    30B dense model that it in no way resembles.
    """
    total = fmt_params(params_b)
    if not total:
        return ""
    if active_b and params_b and float(active_b) < float(params_b):
        return f"{total}, aktywne {fmt_params(active_b)}"
    return total


def speed_hint(params_b: float | None, active_b: float | None) -> str:
    """A Polish one-word guess at generation speed from the active parameters."""
    effective = active_b or params_b
    if not effective:
        return "Szybkość zależy od Maca"
    if effective <= 4:
        return "Bardzo szybki"
    if effective <= 9:
        return "Szybki"
    if effective <= 16:
        return "Umiarkowanie szybki"
    if effective <= 30:
        return "Wolniejszy, ale dokładniejszy"
    return "Powolny — do zadań, gdzie liczy się jakość"


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #


def trustworthy_size(listed_bytes: int, measured_bytes: int) -> int:
    """The variant's measured size, or 0 when it cannot be the model's.

    A repository whose weights live in sub-folders (an mflux image model:
    transformer/, vae/, text_encoder/) is measured by the hub at its top level
    only — 3 291 bytes for FLUX.2 klein, whose listing says 4,6 GB. Taken at
    face value that turned the store's detail sheet into "ok. 0 MB" and its fit
    into "wagi <0,1 GB". A tenth of the listed size is a failed measurement,
    not a quantisation (even a 2-bit build of a model listed at 8-bit is a
    quarter).
    """
    if measured_bytes <= 0:
        return 0
    if listed_bytes and measured_bytes * 10 < listed_bytes:
        return 0
    return measured_bytes


class Catalog:
    """Friendly store view over a :class:`~localcode.hub.search.HubClient`.

    ``hub``, ``library`` and ``downloads`` are injected so the tests (and the
    server :class:`~localcode.server.core.Core`) can pass their own objects;
    when omitted they are built from ``settings``.
    """

    def __init__(
        self,
        settings: Any,
        *,
        hub: Any | None = None,
        library: Any | None = None,
        downloads: Any | None = None,
        budget_bytes: int | None = None,
    ) -> None:
        self.settings = settings
        self._hub = hub
        self._library = library
        self.downloads = downloads
        self._budget_override = budget_bytes
        self._cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}
        self._lock = threading.Lock()
        #: Polish explanation set when the last :meth:`entries` call fell back
        #: to the offline catalogue; ``None`` otherwise.
        self.notice: str | None = None
        #: ``True`` when the last :meth:`entries` call could not reach the hub.
        self.offline = False
        #: How many models matched the last :meth:`entries` filters (before ``limit``).
        self.total_matches = 0

    # plumbing ----------------------------------------------------------------
    @property
    def hub(self) -> Any:
        if self._hub is None:
            from localcode.hub.search import HubClient

            self._hub = HubClient(self.settings)
        return self._hub

    @property
    def library(self) -> Any:
        if self._library is None:
            from localcode.hub.library import ModelLibrary

            self._library = ModelLibrary(self.settings)
        return self._library

    @property
    def budget_bytes(self) -> int:
        """RAM the runtime may use for models, in bytes."""
        if self._budget_override is not None:
            return int(self._budget_override)
        try:
            from localcode.runtime.memory import default_ram_budget

            return int(default_ram_budget(self.settings))
        except Exception:  # noqa: BLE001 - the store must render without the runtime
            return 0

    @property
    def quality(self) -> str:
        """Requested quantisation level (``settings.store.quality``)."""
        store = getattr(self.settings, "store", None)
        value = str(getattr(store, "quality", "") or "4bit")
        return value if value in QUALITY_LEVELS else "4bit"

    @property
    def preferred_format(self) -> str:
        store = getattr(self.settings, "store", None)
        value = str(getattr(store, "prefer_format", "") or "mlx")
        return value if value in ("mlx", "gguf", "auto") else "mlx"

    @property
    def requires_token(self) -> bool:
        return not bool(getattr(self.settings, "hf_token", None))

    # installed models --------------------------------------------------------
    def _installed_by_repo(self) -> dict[str, dict[str, Any]]:
        """``{repo.lower(): LocalModel DTO}`` for everything already on disk."""
        out: dict[str, dict[str, Any]] = {}
        try:
            models = list(self.library.list())
        except Exception:  # noqa: BLE001 - a broken library must not hide the store
            return out
        for model in models:
            dto = model.to_dict() if hasattr(model, "to_dict") else dict(model)
            repo = dto.get("repo") or ""
            if repo:
                out.setdefault(str(repo).lower(), dto)
        return out

    # entries -----------------------------------------------------------------
    def entries(
        self,
        *,
        category: str | None = None,
        query: str | None = None,
        installed: bool | None = None,
        deduplicate: bool = True,
        limit: int | None = None,
        min_trust: int = 0,
        official_only: bool = False,
        sort: str = "polecane",
    ) -> list[CatalogEntry]:
        """The store listing.

        ``category`` is one of :data:`CATEGORIES`, ``query`` is free text and
        ``installed=True`` keeps only models already on disk (``False``: only
        the ones that are not).  The same model packaged twice (GGUF and MLX)
        is shown once, the other packaging moving to ``alternatives``; pass
        ``deduplicate=False`` for the advanced, one-row-per-repository view.

        ``limit`` caps the listing (and the work: rows beyond it are never
        built), ``min_trust`` hides everything below a trust score and
        ``official_only`` keeps just the laboratories' own repositories.
        :attr:`total_matches` counts what passed the filters.

        The listing comes from the shipped index (``localcode/hub/data``), the
        staff picks and what is on disk, so it is complete without a network;
        a live hub search is merged in on top when one succeeds.
        """
        if category is not None and category not in CATEGORIES:
            raise CatalogError(
                f"Nieznana kategoria „{category}”. Dostępne: {', '.join(CATEGORIES)}.",
                "invalid_category",
            )
        sort = (sort or "polecane").strip() or "polecane"
        if sort not in SORTS:
            raise CatalogError(
                f"Nieznane sortowanie „{sort}”. Dostępne: {', '.join(SORTS)}.",
                "invalid_sort",
            )
        self.notice = None
        self.offline = False
        query = (query or "").strip()
        on_disk = self._installed_by_repo()

        rows: dict[str, dict[str, Any]] = {}

        def slot(repo: str) -> dict[str, Any]:
            return rows.setdefault(
                repo, {"curated": curated_entry(repo), "hub": None, "store": None})

        for entry in curated_models(include_images=True):
            rows[entry["repo"]] = {"curated": entry, "hub": None, "store": None}

        # The shipped index (thousands of models, no network) is filtered here
        # rather than after building, so a query never costs 6000 DTOs.
        index_rows = store_data.search(query, category=category, min_score=min_trust,
                                       official_only=official_only, include_excluded=False)
        for row in index_rows:
            slot(str(row["repo"]))["store"] = row

        # Everything already on disk belongs in the listing even when it is in
        # neither the index nor the staff picks.
        for repo, local in on_disk.items():
            slot(str(local.get("repo") or repo))

        if query or installed is not True:
            for model in self._search(query, category):
                repo = str(model.get("id") or "")
                if not repo:
                    continue
                slot(repo)["hub"] = model

        # Pre-rank on facts alone (installed, staff pick, popularity) so that a
        # limited listing builds only the rows it is going to show.  Rows that
        # are *not* in the index — staff picks, models on disk, live search hits
        # — are few and are always built, so nothing hand-picked can fall off
        # the end of a page.
        order: list[tuple[Any, ...]] = []
        for repo, parts in rows.items():
            store = parts.get("store") or {}
            hub = parts.get("hub") or {}
            downloads = int(store.get("downloads") or hub.get("downloads") or 0)
            likes = int(store.get("likes") or hub.get("likes") or 0)
            installed_first = repo.lower() not in on_disk
            if sort == "polecane":
                key: tuple[Any, ...] = (
                    installed_first,
                    parts.get("curated") is None,
                    not self._index_lab_release(repo, store, downloads),
                    store.get("rank", 10 ** 9) if store else 10 ** 9,
                    -downloads,
                )
            elif sort == "najwieksze":
                # Biggest first among what this Mac can hold — measured the
                # same way the built entry will be (see ``_fit``).
                size = int(store.get("sizeBytes") or 0)
                room = self.budget_bytes
                fits = bool(size) and (not room or size * LOAD_OVERHEAD <= room)
                key = (not fits, -size, -downloads)
            elif sort == "popularne":
                key = (-downloads, -likes)
            elif sort == "lubiane":
                key = (-likes, -downloads)
            elif sort == "zaufanie":
                key = (-self._sort_score(repo, parts), -downloads)
            elif sort == "oficjalne":
                key = (0 if str(store.get("provenance") or "") == "official" else 1,
                       -self._sort_score(repo, parts), -downloads)
            else:                                   # najmniejsze
                size = int(store.get("sizeBytes") or 0) or 10 ** 15
                key = (size, -downloads)
            order.append((*key, repo))
        order.sort()

        budget = None if limit is None else max(limit * 4, 64)
        out: list[CatalogEntry] = []
        extra_matches = 0
        built_from_index = 0
        for key in order:
            repo = key[-1]
            parts = rows[repo]
            from_index = parts.get("store") is not None
            if from_index and budget is not None and built_from_index >= budget:
                continue
            entry = self._build(repo, parts["curated"], parts["hub"], parts.get("store"), on_disk)
            # Skip excluded entries unless they are installed (so user can delete them)
            if (entry.get("trust") or {}).get("excluded") and not entry["installed"]:
                continue
            if category and entry["category"] != category:
                continue
            if installed is True and not entry["installed"]:
                continue
            if installed is False and entry["installed"]:
                continue
            # Index rows were matched against the query already, on a wider
            # haystack (publisher, architecture, base model); re-checking them
            # here would drop rows the total count says are there.
            if query and not from_index and not self._matches(entry, query):
                continue
            if min_trust and int((entry.get("trust") or {}).get("score") or 0) < min_trust:
                continue
            if official_only and not (entry.get("publisher") or {}).get("official"):
                continue
            if from_index:
                built_from_index += 1
            else:
                extra_matches += 1
            out.append(entry)

        # The index rows were already filtered by query/category/trust, so the
        # honest total is "all of them that survive the installed filter" plus
        # whatever else matched — counted without building anything.
        index_matches = 0
        for row in index_rows:
            is_installed = str(row["repo"]).lower() in on_disk
            if installed is True and not is_installed:
                continue
            if installed is False and is_installed:
                continue
            index_matches += 1
        self.total_matches = index_matches + extra_matches
        out.sort(key=lambda entry: self._sort_key(entry, sort))
        if deduplicate:
            out = self._deduplicate(out)
        return out[:limit] if limit is not None else out

    def _deduplicate(self, entries: list[CatalogEntry]) -> list[CatalogEntry]:
        """One row per model: GGUF and MLX packagings of the same model merge.

        Installed always wins; after that the packaging named by
        ``settings.store.prefer_format`` does, because on Apple Silicon the MLX
        build of the same model is usually the faster one.  The other packaging
        stays in ``alternatives`` for the advanced view.
        """
        preferred = self.preferred_format
        best: dict[tuple[str, str], CatalogEntry] = {}
        order: list[tuple[str, str]] = []
        for entry in entries:
            key = (_fold(entry["name"]), entry["category"])
            keeper = best.get(key)
            if keeper is None:
                best[key] = entry
                order.append(key)
                continue
            takes_over = entry["installed"] and not keeper["installed"]
            if not takes_over and preferred != "auto" and not keeper["installed"]:
                takes_over = (entry.get("format") == preferred
                              and keeper.get("format") != preferred)
            if takes_over:
                entry["alternatives"] = [*keeper.get("alternatives", []), keeper["source"]]
                best[key] = entry
            else:
                keeper.setdefault("alternatives", []).append(entry["source"])
        return [best[k] for k in order]

    def _search(self, query: str, category: str | None) -> list[dict[str, Any]]:
        """Cached hub search; offline returns ``[]`` and sets :attr:`notice`."""
        # Image models are not indexed by the LLM search (their pipeline tag is
        # excluded), so the "obrazy" tab is curated-only.
        if category == "obrazy":
            return []
        key = f"{query}|{category or ''}"
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit and now - hit[0] < SEARCH_TTL_S:
                return hit[1]
        try:
            results = list(self.hub.search(query, format="any", limit=DEFAULT_LIMIT))
        except Exception:  # noqa: BLE001 - offline is a normal state here
            self.offline = True
            self.notice = OFFLINE_NOTE
            return []
        with self._lock:
            self._cache[key] = (now, results)
        return results

    @staticmethod
    def _matches(entry: CatalogEntry, query: str) -> bool:
        needle = _fold(query)
        if not needle:
            return True
        hay = " ".join([
            _fold(entry["name"]), _fold(entry["id"]), _fold(entry.get("tagline") or ""),
            _fold(entry["category"]), *(_fold(a) for a in ALIASES.get(entry["id"], ())),
        ])
        return all(word in hay for word in needle.split())

    @staticmethod
    def _sort_score(repo: str, parts: dict[str, Any]) -> int:
        """Trust score of a row before it is built (index rows only, cached)."""
        store = parts.get("store")
        if not store:
            return 0
        return int(store_data.trust_of(store).score)

    def _sort_key(self, entry: CatalogEntry, sort: str) -> tuple[Any, ...]:
        """Final ordering of the built page."""
        if sort == "polecane":
            return self._rank(entry)
        fit_order = {FIT_OK: 0, FIT_TIGHT: 1, FIT_UNKNOWN: 2, FIT_NO: 3}
        downloads = -int(entry.get("downloads") or 0)
        likes = -int(entry.get("likes") or 0)
        score = -int((entry.get("trust") or {}).get("score") or 0)
        size = int(entry.get("sizeBytes") or 0) or 10 ** 15
        if sort == "popularne":
            return (downloads, likes, entry["name"])
        if sort == "lubiane":
            return (likes, downloads, entry["name"])
        if sort == "zaufanie":
            return (score, downloads, entry["name"])
        if sort == "oficjalne":
            return (0 if (entry.get("publisher") or {}).get("official") else 1,
                    score, downloads, entry["name"])
        if sort == "najwieksze":
            # The most capable model this Mac can hold: everything that fits,
            # biggest first, then what does not fit — so a Mac with 64 GB is
            # not shown the same 4 GB models as one with 16 GB.
            fits = 0 if entry["fit"] in (FIT_OK, FIT_TIGHT) else (1 if entry["fit"] == FIT_UNKNOWN else 2)
            # In 4 GB steps, so that among models of about the same size the
            # laboratory's own and the trusted one come before a remix. Below
            # the line, the ones that miss by the least come first.
            step = size // (4 * 10 ** 9) if size < 10 ** 15 else 0
            bucket = -step if fits == 0 else step
            publisher = entry.get("publisher") or {}
            lab = lab_release(str(entry.get("id") or ""), provenance=publisher.get("kind"),
                              lab_known=bool(publisher.get("labKnown")),
                              downloads=int(entry.get("downloads") or 0))
            return (fits, bucket, not lab, score, downloads, entry["name"])
        # najmniejsze: the smallest model that still fits comes first, but in
        # 100 MB buckets — otherwise a 40 MB experiment nobody has ever run
        # outranks a 120 MB embedding model that works.
        bucket = size // (100 * 10 ** 6)
        return (fit_order.get(entry["fit"], 4), bucket, score, entry["name"])

    @staticmethod
    def _index_lab_release(repo: str, store: dict[str, Any], downloads: int) -> bool:
        """:func:`lab_release` for an index row, before any entry is built."""
        if not store:
            return False
        kind = str(store.get("provenance") or "")
        # A trusted conversion names the laboratory as its publisher when the
        # weights' origin is known (see ``publishers.classify``).
        lab_known = kind == "official" or str(store.get("publisher") or "") in _LAB_NAMES
        return lab_release(repo, provenance=kind, lab_known=lab_known, downloads=downloads)

    @staticmethod
    def _rank(entry: CatalogEntry) -> tuple[Any, ...]:
        """Installed first, then staff picks, then what fits, then the
        laboratories' own models, then what people use.

        Popularity replaced "largest file first" when the listing grew from
        sixty hand-picked models to thousands: among equals, the model with a
        million downloads is the better default than the biggest one.
        """
        fit_order = {FIT_OK: 0, FIT_TIGHT: 1, FIT_UNKNOWN: 2, FIT_NO: 3}
        publisher = entry.get("publisher") or {}
        downloads = int(entry.get("downloads") or 0)
        return (
            not entry["installed"],
            not entry.get("curated"),
            fit_order.get(entry["fit"], 4),
            not lab_release(str(entry.get("id") or entry.get("repo") or ""),
                            provenance=publisher.get("kind"),
                            lab_known=bool(publisher.get("labKnown")), downloads=downloads),
            -downloads,
            entry.get("format") != "gguf",
            -(entry.get("sizeBytes") or 0),
            entry["name"],
        )

    # entry building ----------------------------------------------------------
    def _build(
        self,
        repo: str,
        curated: dict[str, Any] | None,
        hub: dict[str, Any] | None,
        store: dict[str, Any] | None,
        on_disk: dict[str, dict[str, Any]],
    ) -> CatalogEntry:
        hub = hub or {}
        curated = curated or {}
        store = store or {}
        local = on_disk.get(repo.lower())
        fmt = str(curated.get("format") or store.get("format") or hub.get("format") or "gguf")
        params = curated.get("paramsB") or store.get("paramsB") or hub.get("paramsB")
        active = (curated.get("activeParamsB") or store.get("activeParamsB")
                  or hub.get("activeParamsB"))
        quant = (curated.get("recommendedQuant")
                 or quant_for_quality(store.get("quants") or (), self.quality, fmt)
                 or store.get("quant"))
        category = self._category(repo, curated, hub, local, store)

        if local and local.get("sizeBytes"):
            size = int(local["sizeBytes"])
        elif curated.get("sizeGb"):
            size = int(float(curated["sizeGb"]) * 1e9)
        elif store.get("sizeBytes"):
            size = int(store["sizeBytes"])
        elif params:
            size = int(float(params) * 1e9 * _bytes_per_param(quant))
        else:
            size = 0

        fit, fit_detail = self._fit(size, category=category)
        recommended: CatalogVariant | None = None
        # Image models are installed as a set: the diffusion weights plus the
        # text encoders and the VAE.  Downloading only the first file yields a
        # repository that cannot render anything, so the exact list — from the
        # staff picks or from the index — becomes the recommended variant.
        files = list(curated.get("files") or store.get("imageFiles") or [])
        if files:
            recommended = {
                "id": str(files[0]).rsplit(".", 1)[0],
                "label": str(quant or fmt.upper()),
                "sizeBytes": size,
                "quant": quant,
                "files": files,
                "recommended": True,
            }

        trust = self._trust(repo, curated, hub, store)
        entry: CatalogEntry = {
            "id": repo,
            "name": str(store.get("name") or "").strip() or display_name(repo),
            "tagline": self._tagline(curated, hub, category, params, store),
            "category": category,
            "format": fmt,
            "quant": quant,
            "quants": list(store.get("quants") or ([quant] if quant else [])),
            "sizeBytes": size or None,
            "sizeLabel": ("ok. " + fmt_gb(size)) if size else "rozmiar nieznany",
            "fit": fit,
            "fitDetail": fit_detail,
            "speedHint": speed_hint(params, active),
            "installed": local is not None,
            "installedKey": str(local["key"]) if local else None,
            "recommendedVariant": recommended,
            "variants": [recommended] if recommended else [],
            "badges": self._badges(curated, hub, local, category, store),
            "source": {"provider": "Hugging Face", "repo": repo},
            "alternatives": [],
            "requiresToken": (bool(hub.get("gated")) or bool(store.get("gated")))
                              and self.requires_token,
            "curated": repo in CURATED_REPOS or repo in CURATED_IMAGE_REPOS,
            "trust": trust.to_dict(),
            "publisher": (trust.provenance.to_dict() if trust.provenance else {}),
            "downloads": store.get("downloads") or hub.get("downloads"),
            "likes": store.get("likes") or hub.get("likes"),
            "paramsB": params,
            "activeParamsB": active,
            "paramsLabel": params_label(params, active),
            "watchOut": _watch_out(store.get("watchOut"), quant),
        }
        return entry

    def _trust(self, repo: str, curated: dict[str, Any], hub: dict[str, Any],
               store: dict[str, Any]) -> Trust:
        """The trust score for a row, from the best facts available.

        The shipped index carries the file list summary and the licence, so its
        rows score on full information; a model seen only in a live hub search
        is scored on what that search returned, and says so in its reasons.
        """
        is_curated = bool(curated) or repo in CURATED_REPOS or repo in CURATED_IMAGE_REPOS
        return store_data.trust_for(repo, hub=hub, curated=is_curated)

    @staticmethod
    def _category(repo: str, curated: dict[str, Any], hub: dict[str, Any],
                  local: dict[str, Any] | None, store: dict[str, Any] | None = None) -> str:
        if curated.get("category"):
            return _CATEGORY_FROM_CURATED.get(str(curated["category"]), "czat")
        if (store or {}).get("category") in CATEGORIES:
            return str(store["category"])
        caps = list((local or {}).get("capabilities") or [])
        if "image" in caps or (local or {}).get("imageFamily"):
            return "obrazy"
        if "embedding" in caps:
            return "embedding"
        tags = {str(t).lower() for t in (hub.get("tags") or [])}
        pipeline = str(hub.get("pipelineTag") or "").lower()
        text = _fold(repo)
        if pipeline in ("text-to-image", "image-to-image") or "diffusion" in tags:
            return "obrazy"
        if pipeline in ("feature-extraction", "sentence-similarity") or "embedding" in text:
            return "embedding"
        if "vision" in caps or pipeline == "image-text-to-text" or " vl" in f" {text}":
            return "wizja"
        if any(w in text for w in ("coder", "code", "devstral", "codestral")):
            return "kodowanie"
        if any(w in text for w in ("reasoning", "thinking", "math", "r1", "qwq")):
            return "rozumowanie"
        return "czat"

    @staticmethod
    def _tagline(curated: dict[str, Any], hub: dict[str, Any], category: str,
                 params: float | None, store: dict[str, Any] | None = None) -> str:
        note = curated.get("note") or (store or {}).get("tagline")
        if not note:
            # Hub descriptions are derived from tags and often quote the source
            # repository id — that is exactly the jargon this layer hides.
            candidate = str(hub.get("description") or "")
            note = candidate if candidate and "/" not in candidate else None
        if note:
            return str(note)
        what = {
            "kodowanie": "Model do pisania i poprawiania kodu",
            "czat": "Model do rozmowy i pisania tekstów",
            "rozumowanie": "Model do analiz, planowania i matematyki",
            "obrazy": "Model do generowania obrazów",
            "wizja": "Model, który rozumie obrazy i zrzuty ekranu",
            "embedding": "Model do wyszukiwania w Twoich dokumentach",
            "mowa": "Model do rozpoznawania mowy albo czytania tekstu na głos",
            "wideo": "Model do generowania wideo",
        }[category]
        if params:
            rounded = f"{float(params):.1f}".rstrip("0").rstrip(".") or "0"
            return f"{what} — ok. {rounded.replace('.', ',')} mld parametrów."
        return what + "."

    @staticmethod
    def _badges(curated: dict[str, Any], hub: dict[str, Any],
                local: dict[str, Any] | None, category: str,
                store: dict[str, Any] | None = None) -> list[str]:
        caps = set((local or {}).get("capabilities") or []) | set((store or {}).get("caps") or [])
        tags = {str(t).lower() for t in (hub.get("tags") or [])}
        note = str(curated.get("note") or "")
        badges: list[str] = []
        if category == "obrazy":
            # Image models get their own vocabulary; tools/thinking mean nothing here.
            return [_steps_badge(int(curated["steps"]))] if curated.get("steps") else []
        if "tools" in caps or "tools" in tags or "narzędzi" in note:
            badges.append("narzędzia")
        if "thinking" in caps or "reasoning" in tags or category == "rozumowanie" \
                or "myślen" in note:
            badges.append("myślenie")
        if "vision" in caps or category == "wizja":
            badges.append("wizja")
        ctx = (hub.get("contextLength") or (local or {}).get("contextLength")
               or (store or {}).get("contextLength"))
        if ctx:
            badges.append(f"kontekst {_fmt_ctx(int(ctx))}")
        if (curated.get("activeParamsB") or hub.get("activeParamsB")
                or (store or {}).get("activeParamsB")):
            badges.append("szybki MoE")
        # stable order, no duplicates
        seen: set[str] = set()
        return [b for b in badges if not (b in seen or seen.add(b))]

    def _fit(self, size_bytes: int, *, category: str = "czat",
             estimate: dict[str, Any] | None = None) -> tuple[str, str]:
        """``(fit, fitDetail)`` for a download size against this Mac's budget."""
        if estimate and estimate.get("fits") in _FIT_FROM_ESTIMATE:
            verdict = _FIT_FROM_ESTIMATE[str(estimate["fits"])]
            return verdict, str(estimate.get("message") or verdict)
        budget = self.budget_bytes
        if not size_bytes or not budget:
            return FIT_UNKNOWN, "Nie udało się oszacować zapotrzebowania na pamięć."
        need = size_bytes * (1.0 if category == "obrazy" else LOAD_OVERHEAD)
        limit = fmt_gb(budget)
        if need <= budget * 0.7:
            return FIT_OK, f"Potrzebuje ok. {fmt_gb(need)} z {limit} dostępnych dla modeli."
        if need <= budget:
            return (
                FIT_TIGHT,
                f"Potrzebuje ok. {fmt_gb(need)} przy limicie {limit} — zamknij inne aplikacje.",
            )
        return (
            FIT_NO,
            f"Potrzebuje ok. {fmt_gb(need)}, a dla modeli jest tylko {limit}.",
        )

    # details / install -------------------------------------------------------
    def detail(self, entry_id: str) -> CatalogEntry:
        """An entry with the real variant list (one hub request).

        This is what the "Szczegóły / Zaawansowane" sheet shows.
        """
        entry = self._entry_for(entry_id)
        repo = entry["id"]
        try:
            details = self.hub.details(repo)
        except Exception as exc:  # noqa: BLE001 - mapped to a Polish message
            raise CatalogError(
                f"Nie udało się pobrać informacji o modelu „{entry['name']}”: {exc}",
                "detail_failed", 502,
            ) from exc
        variants = [v for v in details.get("variants") or [] if not v.get("isVisionProjector")]
        chosen = self._choose_variant(entry, variants)
        entry["variants"] = self._variant_rows(entry, variants, chosen)
        if chosen is not None:
            entry["recommendedVariant"] = self._variant_dto(chosen, recommended=True)
            # A repository whose weights live in sub-folders (an mflux image
            # model: transformer/, vae/, text_encoder/) reports its one
            # variant as 0 bytes. Overwriting with that turned FLUX.2 klein's
            # 4,6 GB into "ok. 0 MB" and its fit into "wagi <0,1 GB"; the
            # entry's own size is the better number then.
            chosen_bytes = trustworthy_size(int(entry.get("sizeBytes") or 0),
                                            int(chosen.get("sizeBytes") or 0))
            if chosen_bytes:
                entry["sizeBytes"] = chosen_bytes
                entry["sizeLabel"] = "ok. " + fmt_gb(chosen_bytes)
                entry["fit"], entry["fitDetail"] = self._fit(
                    chosen_bytes,
                    category=entry["category"],
                    estimate=chosen.get("estimate"),
                )
        caps = list(details.get("capabilities") or [])
        if caps:
            entry["badges"] = self._badges(
                curated_entry(repo) or {},
                {"contextLength": details.get("contextLength"), "id": repo},
                {"capabilities": caps},
                entry["category"],
                store_data.get(repo),
            )
        if details.get("accessError"):
            entry["requiresToken"] = True
        return entry

    @staticmethod
    def _variant_dto(variant: dict[str, Any], *, recommended: bool | None = None) -> CatalogVariant:
        return {
            "id": str(variant.get("id") or ""),
            "label": str(variant.get("label") or variant.get("quant") or "—"),
            "sizeBytes": int(variant.get("sizeBytes") or 0),
            "quant": variant.get("quant"),
            "files": list(variant.get("files") or []),
            "recommended": bool(variant.get("recommended")) if recommended is None else recommended,
        }

    def _variant_rows(self, entry: CatalogEntry, variants: Sequence[dict[str, Any]],
                      chosen: dict[str, Any] | None) -> list[CatalogVariant]:
        """Every quantisation on offer, each with its own "does it fit" verdict.

        This is the list "Zainstaluj" lets the person choose from; the one the
        store would pick by itself is the one marked ``recommended``.
        """
        rows: list[CatalogVariant] = []
        category = str(entry.get("category") or "czat")
        for variant in variants:
            dto = self._variant_dto(variant, recommended=(
                chosen is not None and variant.get("id") == chosen.get("id")))
            fit, detail = self._fit(int(variant.get("sizeBytes") or 0), category=category)
            dto["fit"] = fit
            dto["fitDetail"] = detail
            rows.append(dto)
        return rows

    def _choose_variant(self, entry: CatalogEntry,
                        variants: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
        """Best quality that still fits; otherwise the smallest one there is.

        ``HubClient.details`` already marks one variant ``recommended`` using
        this Mac's memory budget (see :func:`localcode.hub.search.pick_recommended`);
        we honour it and only fall back when nothing fits at all — the caller
        then warns the user instead of silently refusing to install.
        """
        usable = [v for v in variants if v.get("complete", True) and v.get("files")]
        if not usable:
            usable = list(variants)
        if not usable:
            return None
        wanted = quant_for_quality([str(v.get("quant") or "") for v in usable if v.get("quant")],
                                    self.quality, str(entry.get("format") or "gguf"))
        if wanted:
            match = next((v for v in usable if str(v.get("quant") or "") == wanted), None)
            if match is not None and self._fits(match):
                return match
        chosen = next((v for v in usable if v.get("recommended")), None)
        if chosen is not None:
            return chosen
        budget = self.budget_bytes
        fitting = [
            v for v in usable
            if not budget or int(v.get("sizeBytes") or 0) * LOAD_OVERHEAD <= budget
        ]
        if fitting:
            return max(fitting, key=lambda v: int(v.get("sizeBytes") or 0))
        return min(usable, key=lambda v: int(v.get("sizeBytes") or 0))

    def _fits(self, variant: dict[str, Any]) -> bool:
        """True when a variant still fits this Mac's model budget."""
        budget = self.budget_bytes
        size = int(variant.get("sizeBytes") or 0)
        return not budget or not size or size * LOAD_OVERHEAD <= budget

    def _entry_for(self, entry_id_or_name: str) -> CatalogEntry:
        entry = self.resolve_name(entry_id_or_name)
        if entry is None:
            raise CatalogError(
                f"Nie znaleziono modelu „{entry_id_or_name}” w katalogu.", "not_found", 404
            )
        return entry

    def resolve_name(self, text: str) -> CatalogEntry | None:
        """Find an entry by friendly name, repository id or alias.

        Matching is accent- and case-insensitive: ``"qwen3 coder"``,
        ``"Qwen3 Coder 30B"`` and the full repository id all work.  When the
        default listing (staff picks + trending) does not contain the model,
        the hub is searched for ``text`` as well, so that anything
        ``localcode models search`` can show is also installable by name.
        """
        text = (text or "").strip()
        if not text:
            return None
        listing = self.entries()
        # A full repository id must win over a fuzzy match, wherever it lives.
        # Otherwise "city96/LTX-Video-gguf" — every word of which also appears in
        # "city96/LTX-Video-0.9.6-distilled-gguf" — installs the wrong model,
        # which is exactly what happened.
        found = self._resolve_in(text, listing, exact_only=True)
        if found is not None:
            return found
        try:
            searched = self.entries(query=text)
        except Exception:  # noqa: BLE001 - offline: the default listing is all we have
            searched = []
        found = self._resolve_in(text, searched, exact_only=True)
        if found is not None:
            return found
        found = self._resolve_in(text, listing)
        if found is not None:
            return found
        found = self._resolve_in(text, searched)
        if found is not None:
            return found
        # Check if it's an exact repository id match that's excluded: we still
        # need to find it so install() can refuse it with the proper error.
        repo = parse_repo_ref(text)
        if repo:
            # Try to build the entry even if it's excluded
            store_entry = store_data.get(repo)
            if store_entry:
                on_disk = self._installed_by_repo()
                entry = self._build(repo, None, None, store_entry, on_disk)
                return entry
            # Otherwise try as a remote entry
            return self.remote_entry(repo)
        return None

    def remote_entry(self, repo: str) -> CatalogEntry | None:
        """A store row for a repository that is not in our catalogue.

        Anything on Hugging Face can be installed this way — it simply arrives
        without the things the catalogue adds: no Polish description, no staff
        verification.  What it does get is the same trust score and publisher
        badge, because those are computed from the repository itself.
        """
        try:
            details = self.hub.details(repo)
        except Exception:  # noqa: BLE001 - a typo in a pasted id is not an error
            return None
        model = details.get("model") if isinstance(details.get("model"), dict) else {}
        entry = self._build(repo, None, {**model, "id": repo}, None, self._installed_by_repo())
        variants = [v for v in details.get("variants") or [] if not v.get("isVisionProjector")]
        chosen = self._choose_variant(entry, variants)
        entry["variants"] = self._variant_rows(entry, variants, chosen)
        if chosen is not None:
            entry["recommendedVariant"] = self._variant_dto(chosen, recommended=True)
            size = int(chosen.get("sizeBytes") or 0)
            if size:
                entry["sizeBytes"] = size
                entry["sizeLabel"] = "ok. " + fmt_gb(size)
                entry["fit"], entry["fitDetail"] = self._fit(
                    size, category=entry["category"], estimate=chosen.get("estimate"))
        # The generated line still says what the model is; what the row does not
        # get is a description written for it, and the badge says so.
        entry["external"] = True
        entry["badges"] = ["spoza katalogu", *entry.get("badges", [])][:4]
        return entry

    def _resolve_in(self, text: str, pool: list[CatalogEntry],
                    *, exact_only: bool = False) -> CatalogEntry | None:
        """:meth:`resolve_name` against one already-built listing.

        ``exact_only`` stops after the repository id and the full display name,
        so the caller can look for an exact hit everywhere before letting a fuzzy
        match from the first listing decide.
        """
        exact = next((e for e in pool if e["id"].lower() == text.lower()), None)
        if exact is not None:
            return exact
        needle = _fold(text)
        if not needle:
            return None
        by_name = [e for e in pool if _fold(e["name"]) == needle]
        if by_name:
            return by_name[0]
        if exact_only:
            return None
        words = needle.split()
        scored: list[tuple[tuple[Any, ...], CatalogEntry]] = []
        for e in pool:
            name = _fold(e["name"])
            repo = _fold(e["id"])
            aliases = [_fold(a) for a in ALIASES.get(e["id"], ())]
            hay = " ".join([name, repo, *aliases])
            if not all(w in hay for w in words):
                continue
            scored.append(((0 if name.startswith(needle) else 1,
                            0 if needle in name else 1,
                            len(name)), e))
        if not scored:
            return None
        scored.sort(key=lambda pair: (pair[0], self._rank(pair[1])))
        return scored[0][1]

    def install(self, entry_id_or_name: str, *, variant: str | None = None) -> dict[str, Any]:
        """Install a model: pick the right files and hand them to the downloader.

        ``variant`` (the advanced path) names a variant id explicitly; by
        default the recommended one is used.  Returns the ``DownloadJob`` dict.
        """
        manager = self.downloads
        if manager is None:
            raise CatalogError(
                "Pobieranie modeli jest w tej chwili niedostępne.", "downloads_unavailable", 503
            )
        entry = self._entry_for(entry_id_or_name)
        trust = entry.get("trust") or {}
        if trust.get("excluded"):
            exclusion_text = trust.get("exclusion", "Model został odrzucony.")
            name = entry.get("name", "Model")
            raise CatalogError(f'„{name}" został odrzucony: {exclusion_text}',
                             "rejected_model", 409)
        return self._start(manager, entry, self.install_files(entry, variant=variant))

    @staticmethod
    def _start(manager: Any, entry: CatalogEntry, files: list[str]) -> dict[str, Any]:
        repo = entry["id"]
        try:
            job = manager.start(repo, files, label=entry["name"])
        except TypeError:      # managers that do not take a label
            job = manager.start(repo, files)
        return job if isinstance(job, dict) else {"repo": repo, "files": files}

    async def ainstall(self, entry_id_or_name: str, *, variant: str | None = None) -> dict[str, Any]:
        """:meth:`install` for a server request.

        The two halves have to run in different places: resolving the entry and
        listing its files is blocking network work that belongs in a thread,
        while ``DownloadManager.start`` schedules tasks and therefore has to be
        called *on the event loop* — from a worker thread it raises
        "Menedżer pobierania wymaga działającej pętli asyncio", which is exactly
        what the app showed instead of starting a download.
        """
        manager = self.downloads
        if manager is None:
            raise CatalogError(
                "Pobieranie modeli jest w tej chwili niedostępne.", "downloads_unavailable", 503
            )
        entry = await asyncio.to_thread(self._entry_for, entry_id_or_name)
        trust = entry.get("trust") or {}
        if trust.get("excluded"):
            exclusion_text = trust.get("exclusion", "Model został odrzucony.")
            name = entry.get("name", "Model")
            raise CatalogError(f'„{name}" został odrzucony: {exclusion_text}',
                             "rejected_model", 409)
        files = await asyncio.to_thread(self.install_files, entry, variant=variant)
        return self._start(manager, entry, files)

    def install_files(self, entry: CatalogEntry, *, variant: str | None = None) -> list[str]:
        """The file list to download for ``entry`` (companions included)."""
        curated = curated_entry(entry["id"]) or {}
        known = list(curated.get("files")
                     or ((entry.get("recommendedVariant") or {}).get("files")
                         if entry.get("category") == "obrazy" else None)
                     or [])
        if variant is None and known:
            return known                            # image models: model + encoders
        try:
            details = self.hub.details(entry["id"])
        except Exception as exc:  # noqa: BLE001
            if known:
                return known
            raise CatalogError(
                f"Nie udało się przygotować pobierania modelu „{entry['name']}”: {exc}",
                "install_failed", 502,
            ) from exc
        variants = list(details.get("variants") or [])
        models = [v for v in variants if not v.get("isVisionProjector")]
        if variant:
            chosen = next((v for v in variants if str(v.get("id")) == variant), None)
            if chosen is None:
                raise CatalogError(
                    f"Model „{entry['name']}” nie ma wariantu „{variant}”.",
                    "variant_not_found", 404,
                )
        else:
            chosen = self._choose_variant(entry, models)
        if chosen is None:
            raise CatalogError(
                f"Model „{entry['name']}” nie ma plików do pobrania.", "no_variants", 422
            )
        files = list(chosen.get("files") or [])
        projector = next(
            (v for v in variants if v.get("isVisionProjector") and v.get("recommended")), None
        )
        if projector:
            files += [f for f in (projector.get("files") or []) if f not in files]
        return files


# --------------------------------------------------------------------------- #
# Module-level convenience API
# --------------------------------------------------------------------------- #


def catalog_entries(
    settings: Any,
    *,
    category: str | None = None,
    query: str | None = None,
    installed: bool | None = None,
    catalog: Catalog | None = None,
) -> list[CatalogEntry]:
    """The store listing for ``settings`` (see :meth:`Catalog.entries`)."""
    return (catalog or Catalog(settings)).entries(
        category=category, query=query, installed=installed
    )


def install(
    settings: Any,
    entry_id_or_name: str,
    *,
    variant: str | None = None,
    catalog: Catalog | None = None,
) -> dict[str, Any]:
    """Install a model by friendly name or repository id (see :meth:`Catalog.install`)."""
    return (catalog or Catalog(settings)).install(entry_id_or_name, variant=variant)


def resolve_name(settings: Any, text: str, *, catalog: Catalog | None = None) -> CatalogEntry | None:
    """Fuzzy-match ``text`` against the catalogue (see :meth:`Catalog.resolve_name`)."""
    return (catalog or Catalog(settings)).resolve_name(text)


def catalog_settings_dto() -> dict[str, Any]:
    """The catalogue block of ``GET /api/app-settings`` (provenance strings)."""
    return {
        "sourceNote": SOURCE_NOTE,
        "offlineNote": OFFLINE_NOTE,
        "categories": [
            {"id": cid, "label": CATEGORY_LABELS[cid]} for cid in CATEGORIES
        ],
        "sorts": [
            {"id": sid, "label": SORT_LABELS[sid]} for sid in SORTS
        ],
        "qualities": [
            {"id": qid, "label": QUALITY_LABELS[qid]} for qid in QUALITY_LEVELS
        ],
    }
