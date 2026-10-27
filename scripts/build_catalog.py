#!/usr/bin/env python3
"""Turn the raw Hub dump into the in-app store catalogue.

    python scripts/hub_fetch.py --out .cache/hub-raw.jsonl        # 1. facts
    python scripts/build_catalog.py select                        # 2. what is servable
    python scripts/build_catalog.py batches --out .cache/batches  # 3. work for the writers
    python scripts/build_catalog.py build                         # 4. localcode/hub/data/catalog.json.gz

Step 2 keeps only what ``llama-server`` / ``mlx-lm`` / the image runtime can
actually load, deduplicates the packagings of one model and computes every fact
the store shows.  Step 4 merges in the Polish one-liners written in step 3 and
writes the gzipped catalogue that ships with LocalCode.

The catalogue stores *facts*, never verdicts: the trust score is recomputed from
those facts by :mod:`localcode.hub.trust` every time it is displayed, so the
rubric can change without regenerating the file.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localcode.hub.catalog import (  # noqa: E402
    _LAB_NAMES,
    _bytes_per_param,
    display_name,
    lab_release,
    strip_quant_phrases,
)
from localcode.hub.curated import CURATED_IMAGE_REPOS, CURATED_REPOS, curated_entry  # noqa: E402
from localcode.hub.metadata import (  # noqa: E402
    is_auxiliary_gguf,
    is_mmproj,
    parse_quant,
    parse_size_label,
)
from localcode.hub.publishers import classify  # noqa: E402
from localcode.hub.trust import evaluate, license_from_tags, summarize_files  # noqa: E402

RAW_DEFAULT = ".cache/hub-raw.jsonl"
#: ``hub_fetch.py --extra``: every laboratory's releases and the big-model sweep.
EXTRA_DEFAULT = ".cache/hub-extra.jsonl"
SELECTED_DEFAULT = ".cache/selected.jsonl"
AUTHORS_DEFAULT = ".cache/hub-authors.json"
OUT_DEFAULT = "localcode/hub/data/catalog.json.gz"
BATCH_SIZE = 24
CATEGORIES = ("kodowanie", "czat", "rozumowanie", "obrazy", "wizja", "embedding",
              "mowa", "wideo")
#: Rows above this rank get a hand-written (model-written) Polish one-liner.
WRITTEN_TAGLINE_LIMIT = 1500
#: Hard cap on the shipped catalogue (laboratory releases and the large models
#: below are kept on top of it — see ``protected``).
MAX_ENTRIES = 6000
#: A model this large is what a 32-128 GB Mac is for; the popularity charts
#: rank them low only because fewer people can run them.
BIG_PARAMS_B = 24.0
BIG_SIZE_BYTES = 14 * 10**9
BIG_MIN_DOWNLOADS = 200
OFFICIAL_MIN_DOWNLOADS = 20

# --------------------------------------------------------------------------- #
# What can be served
# --------------------------------------------------------------------------- #

TEXT_PIPELINES = frozenset({"text-generation", "image-text-to-text", "text2text-generation",
                            "conversational", None})
EMBED_PIPELINES = frozenset({"feature-extraction", "sentence-similarity"})
IMAGE_PIPELINES = frozenset({"text-to-image"})

#: Speech in, speech out and video: shelves of their own rather than rejects.
SPEECH_PIPELINES = frozenset({"automatic-speech-recognition", "text-to-speech", "text-to-audio"})
VIDEO_PIPELINES = frozenset({"text-to-video", "image-to-video", "image-text-to-video"})
_SPEECH_TAGS = frozenset({"text-to-speech", "tts", "asr", "stt", "speech-to-text",
                          "automatic-speech-recognition", "voice-cloning"})
_VIDEO_TAGS = frozenset({"text-to-video", "image-to-video", "video-generation"})


def modality(row: dict[str, Any]) -> str | None:
    """``"mowa"`` / ``"wideo"`` when the row is one of those, else ``None``."""
    pipeline = row.get("pipelineTag")
    tags = _tags(row)
    if pipeline in SPEECH_PIPELINES or (tags & _SPEECH_TAGS):
        return "mowa"
    if pipeline in VIDEO_PIPELINES or (tags & _VIDEO_TAGS):
        return "wideo"
    return None


#: Pipelines the LocalCode runtime has no engine for.
REJECT_PIPELINES = frozenset({
    "audio-to-audio", "audio-classification", "audio-text-to-text",
    "video-classification", "video-text-to-text", "any-to-any",
    "image-classification", "object-detection", "image-segmentation", "depth-estimation",
    "zero-shot-image-classification", "zero-shot-object-detection", "mask-generation",
    "unconditional-image-generation", "image-feature-extraction", "reinforcement-learning",
    "robotics", "tabular-classification", "tabular-regression", "time-series-forecasting",
    "text-to-3d", "image-to-3d", "keypoint-detection", "token-classification",
    "text-classification", "zero-shot-classification", "voice-activity-detection",
    "graph-ml", "translation", "summarization", "question-answering",
    "table-question-answering", "document-question-answering", "visual-question-answering",
    "text-ranking", "image-to-text", "fill-mask", "image-to-image", "visual-document-retrieval",
})

#: Tags that mean "this repository is not a chat/coding model you load and talk to".
REJECT_TAGS = frozenset({
    "draft-model", "speculative-decoding", "lora", "peft", "adapter", "controlnet",
    "text-to-video", "image-to-video", "video-generation", "video", "text-to-speech",
    "text-to-audio", "tts", "asr", "speech-to-text", "automatic-speech-recognition",
    "audio", "music", "music-generation", "voice-cloning", "upscaler", "super-resolution",
    "reranker", "rerank", "not-for-all-audiences", "nsfw", "nfaa", "vae", "clip",
    "diffusers-controlnet", "esrgan", "whisper.cpp", "transcribe.cpp", "audio.cpp",
    "sentence-transformers-lora", "onnx-community",
})

#: Model families that are not text, whatever their tags say.  A repository can
#: carry no pipeline tag at all (``mlx-community/whisper-tiny-mlx`` is a plain
#: MLX directory), and then only the name gives it away.
REJECT_NAME_WORDS = (
    "whisper", "parakeet", "wav2vec", "seamless", "musicgen", "audiogen", "bark",
    "vits", "kokoro", "orpheus", "sesame csm", "moonshine", "sensevoice", "funasr",
    "audio", "speech", "tts", "asr", "ace step", "stable audio",
    "ltx video", "ltxv", "animatediff", "hunyuanvideo", "cogvideo", "mochi",
    "wan2", "wan 2", "zeroscope", "upscaler", "esrgan", "controlnet",
    "text encoder", "text encoders", "encoder only", "clip vision", "vae decoder",
    "reranker", "yolo", "segment anything", "depth anything",
)
#: Matched on word boundaries: "tts" must not reject a model called "Watts".
_REJECT_NAME_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(w) for w in REJECT_NAME_WORDS) + r")\b", re.IGNORECASE)


def looks_non_text(repo: str) -> bool:
    """True when the *name* gives away a speech, music or video model."""
    return bool(_REJECT_NAME_RE.search(re.sub(r"[-_./]+", " ", repo)))

#: GGUF architectures that are components or non-text models.
REJECT_ARCHS = frozenset({
    "clip", "t5", "t5encoder", "umt5", "vae", "wan", "ltxv", "flux", "qwen_image", "sd1",
    "sdxl", "sd3", "stable-diffusion", "whisper", "parakeet", "audiocpp", "bark", "vits",
    "kokoro", "snac", "dia", "csm", "moonshine", "wavtokenizer", "nemo", "sam", "yolo",
    "rwkv", "mtp", "dflash", "dspark", "eagle", "medusa", "hunyuan_video", "cosmos",
})

#: Non-text GGUF architectures that we still keep, because the store has a place
#: for them: embeddings feed the chatbot's document search.
EMBED_ARCHS = frozenset({
    "bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert-v3", "modernbert",
    "xlm-roberta", "roberta", "gemma-embedding", "embeddinggemma", "distilbert", "mpnet",
})

#: Image models need their text encoders next to the weights; a repository with
#: only the diffusion file cannot be installed as one unit.
_IMAGE_COMPONENT_RE = re.compile(r"(clip[-_]?[lg]?|t5xxl|t5-v1|umt5|\bae\b|vae|text_encoder)",
                                 re.IGNORECASE)
_MFLUX_MARKERS = ("transformer/", "text_encoder/", "model_index.json", "vae/")

CODE_WORDS = ("coder", "code", "codestral", "devstral", "starcoder", "codegemma", "codellama",
              "codegeex", "sqlcoder", "codeqwen", "opencoder", "granite-code", "deepseek-coder",
              "codebooga", "wizardcoder", "magicoder", "codeshell", "stable-code", "refact")
REASON_WORDS = ("reasoning", "thinking", "-r1", "r1-", "qwq", "math", "deepthink", "reasoner",
                "cogito", "marco-o1", "openthinker", "deepscaler", "s1-", "phi-4-reasoning")
VISION_WORDS = ("-vl", "vl-", "vision", "llava", "moondream", "internvl", "minicpm-v", "smolvlm",
                "pixtral", "gemma-3-vision", "idefics", "florence", "qwen-vl", "janus", "ovis")
EMBED_WORDS = ("embed", "bge-", "gte-", "e5-", "minilm", "mxbai", "nomic-embed", "stella",
               "jina-embed", "sentence-t5", "labse", "arctic-embed")

IMAGE_WORDS = ("flux", "diffusion", "sdxl", "sd3", "sd1", "qwen-image", "krea", "image")

#: A description writer may re-categorise a model only when its own name says so;
#: "this sounds technical, so it must be a coding model" is exactly the kind of
#: guess that puts a chat model in the wrong shelf.
CATEGORY_EVIDENCE: dict[str, tuple[str, ...]] = {
    "kodowanie": CODE_WORDS,
    "rozumowanie": REASON_WORDS,
    "wizja": VISION_WORDS,
    "embedding": EMBED_WORDS,
    "obrazy": IMAGE_WORDS,
}


def category_change_allowed(repo: str, name: str, category: str) -> bool:
    """True when the model's own name supports moving it to ``category``."""
    if category == "czat":                      # the fallback shelf needs no proof
        return True
    words = CATEGORY_EVIDENCE.get(category)
    if not words:
        return False
    return _has(f"{repo} {name}", words)


#: Lookarounds, not consumed separators: in "Qwen3-32B-MLX-4bit" the dash
#: between "MLX" and "4bit" belongs to both, and consuming it hid the "4bit" —
#: the repository was then sized as bf16, four times too big to fit.
_QUANT_IN_NAME = re.compile(
    r"(?:^|(?<=[-_.]))(?P<q>\d+bit|bf16|fp16|fp32|f16|mxfp4|nvfp4|fp8|int4|int8|q\d[a-z0-9_]*|"
    r"awq|gptq|dwq|hqq|mlx|4bit-dwq|6bit|8bit)(?=$|[-_.])", re.IGNORECASE)
_NOISE_FOR_KEY = re.compile(
    r"[-_.]?(?:gguf|mlx|mflux|awq|gptq|bnb|int4|int8|fp8|fp16|fp32|bf16|mxfp4|nvfp4|nf4|"
    r"\d+bits?|q\d[a-z0-9_]*|iq\d[a-z0-9_]*|imatrix|i1|ud|quantized|quants?|converted|onnx|"
    r"openvino|safetensors|ct2|ctranslate2|exl2|exl3|hqq|dwq|mtp|unsloth|dynamic|hf|"
    r"instruct|it|chat)\b", re.IGNORECASE)

#: Quantisation preference by model size (best first).  Small models are stored
#: at higher precision because the quality loss is proportionally larger.
QUANT_PREFERENCE_SMALL = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "F16", "BF16")
QUANT_PREFERENCE_MID = ("Q4_K_M", "UD-Q4_K_XL", "Q4_K_S", "IQ4_XS", "Q5_K_M", "Q6_K", "Q8_0")
QUANT_PREFERENCE_BIG = ("IQ4_XS", "Q4_K_M", "UD-Q4_K_XL", "Q3_K_M", "IQ3_XXS", "Q4_0")
QUANT_PREFERENCE_IMAGE = ("Q8_0", "Q6_K", "F16", "BF16")


def _tags(row: dict[str, Any]) -> set[str]:
    return {str(t).lower() for t in (row.get("tags") or ())}


def _has(name: str, words: Iterable[str]) -> bool:
    low = name.lower().replace("_", "-")
    return any(w in low for w in words)


def norm_key(repo: str) -> str:
    """Grouping key for "the same model, packaged differently"."""
    name = repo.partition("/")[2] or repo
    name = _NOISE_FOR_KEY.sub("", strip_quant_phrases(name).replace("_", "-"))
    return re.sub(r"[^a-z0-9.]+", "", name.lower())


def _gguf_files(siblings: Iterable[str]) -> list[str]:
    return [f for f in siblings if str(f).lower().endswith(".gguf")]


def _quants(row: dict[str, Any]) -> list[str]:
    """Quantisation labels a repository offers, best-guess order preserved."""
    if row["format"] == "gguf":
        out: list[str] = []
        for f in _gguf_files(row.get("siblings") or ()):
            if is_auxiliary_gguf(f):
                continue
            q = parse_quant(f)
            if q and q not in out:
                out.append(q)
        return out
    labels = [m.group("q").upper() for m in _QUANT_IN_NAME.finditer(row["id"].partition("/")[2] or "")]
    label = next((q for q in labels if q != "MLX"), "")
    if not label:
        return ["BF16"]
    return [label.replace("BIT", "bit")]


def _pick_quant(quants: list[str], params_b: float | None, *, image: bool = False) -> str | None:
    if not quants:
        return None
    if image:
        order = QUANT_PREFERENCE_IMAGE
    elif params_b is None:
        order = QUANT_PREFERENCE_MID
    elif params_b <= 3.5:
        order = QUANT_PREFERENCE_SMALL
    elif params_b <= 40:
        order = QUANT_PREFERENCE_MID
    else:
        order = QUANT_PREFERENCE_BIG
    upper = {q.upper(): q for q in quants}
    for want in order:
        if want in upper:
            return upper[want]
    # Nothing preferred: the smallest sane thing the repository has.
    for q in quants:
        if not q.upper().startswith(("Q2", "IQ1", "IQ2")):
            return q
    return quants[0]


def _params(row: dict[str, Any]) -> tuple[float | None, float | None]:
    name = row["id"].partition("/")[2] or ""
    total, active = parse_size_label(name)
    if total is None and row["format"] == "gguf" and row.get("ggufTotalParams"):
        total = round(float(row["ggufTotalParams"]) / 1e9, 2)
    if total is None and row.get("safetensorsTotal"):
        # Quantised MLX repositories report packed element counts, which
        # understate the model; only trust this for unquantised weights.
        quants = _quants(row)
        if not quants or quants[0].upper() in ("BF16", "F16", "FP16", "F32"):
            total = round(float(row["safetensorsTotal"]) / 1e9, 2)
    if total is not None and total <= 0:
        total = None
    return total, active


def _category(row: dict[str, Any], *, is_image: bool, is_embed: bool) -> str:
    name = row["id"]
    tags = _tags(row)
    kind = modality(row)
    if kind:
        return kind
    if is_image:
        return "obrazy"
    if is_embed:
        return "embedding"
    if _has(name, CODE_WORDS) or "code" in tags:
        return "kodowanie"
    if _has(name, REASON_WORDS) or "reasoning" in tags:
        return "rozumowanie"
    # Vision belongs in the category only for models built around images; every
    # recent general model reads pictures too, and calling them all "wizja"
    # would empty the chat category of everything worth using.
    if _has(name, VISION_WORDS):
        return "wizja"
    return "czat"


def _caps(row: dict[str, Any], category: str) -> list[str]:
    tpl = row.get("template") or {}
    caps: list[str] = []
    if tpl.get("tools"):
        caps.append("tools")
    if tpl.get("thinking") or category == "rozumowanie":
        caps.append("thinking")
    if tpl.get("effort"):
        caps.append("effort")
    if (category == "wizja" or row.get("pipelineTag") == "image-text-to-text"
            or tpl.get("vision") or any(is_mmproj(f) for f in (row.get("siblings") or ()))):
        caps.append("vision")
    if category == "embedding":
        caps.append("embedding")
    if category == "obrazy":
        caps.append("image")
    if category == "mowa":
        caps.append(_speech_direction(row))
    if category == "wideo":
        caps.append("video")
    return caps


#: Words in a repository name or tag that mean "listens", not "speaks".
_STT_WORDS = ("asr", "stt", "speech-to-text", "speech_to_text", "transcri", "whisper",
              "parakeet", "conformer", "wav2vec", "hubert", "moonshine", "sensevoice",
              "canary", "subtitle", "caption")
_TTS_WORDS = ("tts", "text-to-speech", "text_to_speech", "voice-clon", "vocoder",
              "kokoro", "orpheus", "bark", "vits", "xtts", "styletts", "outetts",
              "piper", "melotts", "f5-tts", "fish-speech")


def _speech_direction(row: dict[str, Any]) -> str:
    """``"stt"`` or ``"tts"`` for a speech repository.

    The pipeline tag alone was not enough: ``Qwen3-ASR`` carries no
    ``automatic-speech-recognition`` tag, so it fell through to the ``tts``
    default and the store described a transcription model as a narrator. The
    name and the tags decide when the pipeline tag does not.
    """
    tag = str(row.get("pipelineTag") or "").lower()
    if tag in ("automatic-speech-recognition", "speech-to-text"):
        return "stt"
    if tag in ("text-to-speech", "text-to-audio"):
        return "tts"
    haystack = " ".join([str(row.get("id") or ""), " ".join(str(t) for t in (row.get("tags") or ()))]).lower()
    stt = next((w for w in _STT_WORDS if w in haystack), None)
    tts = next((w for w in _TTS_WORDS if w in haystack), None)
    if stt and not tts:
        return "stt"
    if tts and not stt:
        return "tts"
    if stt and tts:                      # both named: the more specific word wins
        return "stt" if haystack.index(stt) < haystack.index(tts) else "tts"
    return "tts"


def _image_ok(row: dict[str, Any]) -> bool:
    """True when an image repository carries everything needed to render."""
    siblings = [str(f) for f in (row.get("siblings") or ())]
    if row["format"] == "mlx":
        return any(any(m in f for m in _MFLUX_MARKERS) for f in siblings)
    weights = [f for f in _gguf_files(siblings) if not is_auxiliary_gguf(f)]
    components = [f for f in siblings if _IMAGE_COMPONENT_RE.search(f)]
    return bool(weights) and len(components) >= 2


def reject_reason(row: dict[str, Any]) -> str | None:
    """Why this repository cannot be offered, or ``None`` when it can."""
    repo = row["id"]
    if row.get("private") or row.get("disabled"):
        return "prywatne/wyłączone"
    tags = _tags(row)
    pipeline = row.get("pipelineTag")
    arch = str(row.get("arch") or "").lower()
    siblings = [str(f) for f in (row.get("siblings") or ())]
    if not siblings:
        return "brak listy plików"
    if modality(row):
        # Speech and video have their own shelves now; such a row only has to
        # carry weights — a bare component (an encoder, a VAE) still does not.
        if not [f for f in siblings
                if f.lower().endswith((".gguf", ".safetensors", ".npz", ".bin", ".onnx", ".pt"))]:
            return "brak wag"
        if any(f.endswith("adapter_config.json") for f in siblings):
            return "adapter LoRA"
        return None
    if tags & REJECT_TAGS:
        return "tag: " + ",".join(sorted(tags & REJECT_TAGS)[:2])
    if any(f.endswith("adapter_config.json") for f in siblings):
        return "adapter LoRA"
    if looks_non_text(repo):
        return "nazwa: model nietekstowy"

    is_image = pipeline in IMAGE_PIPELINES or "text-to-image" in tags
    if is_image:
        if not _image_ok(row):
            return "obraz bez enkoderów"
        return None
    if pipeline in REJECT_PIPELINES:
        return f"pipeline: {pipeline}"

    is_embed = pipeline in EMBED_PIPELINES or arch in EMBED_ARCHS or _has(repo, EMBED_WORDS)
    if row["format"] == "gguf":
        weights = [f for f in _gguf_files(siblings) if not is_auxiliary_gguf(f)]
        if not weights:
            return "tylko pliki pomocnicze"
        if arch in REJECT_ARCHS and not is_embed:
            # The reported architecture belongs to the first file in the repo,
            # which may be a projector; trust the pipeline tag when it disagrees.
            if pipeline not in ("text-generation", "image-text-to-text"):
                return f"architektura: {arch}"
    else:
        if not any(f.endswith(".safetensors") or f.endswith(".npz") for f in siblings):
            return "brak wag safetensors"
        if not any(f.endswith("config.json") for f in siblings):
            return "brak config.json"
    if pipeline not in TEXT_PIPELINES and not is_embed:
        return f"pipeline: {pipeline}"
    return None


def build_row(row: dict[str, Any]) -> dict[str, Any]:
    """One raw Hub row → one catalogue entry (facts only)."""
    repo = row["id"]
    siblings = [str(f) for f in (row.get("siblings") or ())]
    tags = _tags(row)
    pipeline = row.get("pipelineTag")
    arch = str(row.get("arch") or "").lower()
    is_image = pipeline in IMAGE_PIPELINES or "text-to-image" in tags
    is_embed = (not is_image) and (pipeline in EMBED_PIPELINES or arch in EMBED_ARCHS
                                   or _has(repo, EMBED_WORDS))
    params, active = _params(row)
    quants = _quants(row)
    quant = _pick_quant(quants, params, image=is_image)
    category = _category(row, is_image=is_image, is_embed=is_embed)
    caps = _caps(row, category)
    size = None
    if params:
        size = int(float(params) * 1e9 * _bytes_per_param(quant))
    elif row.get("ggufTotalFileSize") and len(quants) <= 1:
        size = int(row["ggufTotalFileSize"])
    prov = classify(repo, base_models=row.get("baseModels"), tags=row.get("tags") or ())
    entry = {
        "repo": repo,
        "format": row["format"],
        "author": row.get("author"),
        "name": display_name(repo),
        "category": category,
        "paramsB": params,
        "activeParamsB": active,
        "contextLength": row.get("ggufContext"),
        "arch": row.get("arch"),
        "quant": quant,
        "quants": quants[:14],
        "sizeBytes": size,
        "downloads": row.get("downloads") or 0,
        "downloadsAllTime": row.get("downloadsAllTime"),
        "likes": row.get("likes") or 0,
        "updatedAt": row.get("lastModified"),
        "createdAt": row.get("createdAt"),
        # Rows fetched before the tag fallback existed have no licence field.
        "license": row.get("license") or license_from_tags(row.get("tags")),
        "licenseName": row.get("licenseName"),
        "gated": bool(row.get("gated")) and row.get("gated") is not False,
        "caps": caps,
        "files": summarize_files(siblings),
        "baseModel": prov.upstream,
        "provenance": prov.kind,
        "publisher": prov.publisher,
        "packager": prov.packager,
        "curated": repo in CURATED_REPOS or repo in CURATED_IMAGE_REPOS,
    }
    if is_image:
        entry["imageFamily"] = _image_family(repo)
        components = [f for f in siblings if _IMAGE_COMPONENT_RE.search(f)]
        weights = [f for f in _gguf_files(siblings) if not is_auxiliary_gguf(f)]
        if row["format"] == "gguf" and weights:
            entry["imageFiles"] = [*weights[:1], *components[:4]]
    curated = curated_entry(repo)
    if curated:
        entry["tagline"] = curated.get("note")
        if curated.get("minRamGb"):
            entry["minRamGb"] = curated["minRamGb"]
        if curated.get("recommendedQuant"):
            entry["quant"] = curated["recommendedQuant"]
    return entry


def _image_family(repo: str) -> str:
    low = repo.lower()
    for frag, family in (("flux.2", "flux2"), ("flux2", "flux2"), ("flux", "flux"),
                         ("qwen-image", "qwen-image"), ("qwen_image", "qwen-image"),
                         ("sd3", "sd3"), ("stable-diffusion-3", "sd3"),
                         ("sdxl", "sdxl"), ("xl-base", "sdxl"),
                         ("v1-5", "sd1"), ("sd15", "sd1"), ("sd1", "sd1")):
        if frag in low:
            return family
    return "unknown"


def _score_of(entry: dict[str, Any]) -> int:
    try:
        trust = evaluate(
            entry["repo"], author=entry.get("author"), downloads=entry.get("downloads"),
            likes=entry.get("likes"), license_id=entry.get("license"),
            license_name=entry.get("licenseName"), file_summary=entry.get("files"),
            base_models=({"models": [entry["baseModel"]]} if entry.get("baseModel") else None),
            last_modified=entry.get("updatedAt"), created_at=entry.get("createdAt"),
            gated=entry.get("gated"), curated=entry.get("curated", False),
            downloads_all_time=entry.get("downloadsAllTime"), author_info=entry.get("authorInfo"),
        )
    except TypeError:
        # Fallback for when trust module does not yet accept the new parameters
        trust = evaluate(
            entry["repo"], author=entry.get("author"), downloads=entry.get("downloads"),
            likes=entry.get("likes"), license_id=entry.get("license"),
            license_name=entry.get("licenseName"), file_summary=entry.get("files"),
            base_models=({"models": [entry["baseModel"]]} if entry.get("baseModel") else None),
            last_modified=entry.get("updatedAt"), created_at=entry.get("createdAt"),
            gated=entry.get("gated"), curated=entry.get("curated", False),
        )
    return trust.score


# --------------------------------------------------------------------------- #
# select
# --------------------------------------------------------------------------- #


def is_big(entry: dict[str, Any]) -> bool:
    return (float(entry.get("paramsB") or 0) >= BIG_PARAMS_B
            or int(entry.get("sizeBytes") or 0) >= BIG_SIZE_BYTES)


def protected(entry: dict[str, Any]) -> bool:
    """Rows the size cap never drops: a laboratory's own release, and a large
    model that is the laboratory's (or a faithful conversion of it)."""
    downloads = int(entry.get("downloads") or 0)
    if entry.get("provenance") == "official":
        return downloads >= OFFICIAL_MIN_DOWNLOADS
    return (is_big(entry) and downloads >= BIG_MIN_DOWNLOADS
            and lab_release(entry["repo"], provenance=entry.get("provenance"),
                            lab_known=str(entry.get("publisher") or "") in _LAB_NAMES,
                            downloads=downloads))


def _raw_rows(raw_path: Path, extra_path: Path | None) -> Iterable[dict[str, Any]]:
    """The popularity dump, then — only the protected ones — the extra sweep."""
    seen: set[str] = set()
    for line in raw_path.open(encoding="utf-8"):
        row = json.loads(line)
        seen.add(row["id"])
        yield row
    if extra_path is None or not extra_path.exists():
        return
    for line in extra_path.open(encoding="utf-8"):
        row = json.loads(line)
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        row["_extra"] = True
        yield row


def select(raw_path: Path, *, max_entries: int = MAX_ENTRIES,
           extra_path: Path | None = None, authors_path: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rejects: dict[str, int] = {}
    kept: list[dict[str, Any]] = []
    extra_ids: set[str] = set()

    authors_map: dict[str, dict[str, Any]] = {}
    if authors_path is not None and authors_path.exists():
        try:
            authors_map = json.loads(authors_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {authors_path.name}: {exc}", file=sys.stderr)

    for row in _raw_rows(raw_path, extra_path):
        # A staff pick has been installed and run by hand; the automatic filter
        # does not get to drop it (it rejects, for instance, image repositories
        # whose encoders it cannot recognise from file names alone).
        is_pick = row["id"] in CURATED_REPOS or row["id"] in CURATED_IMAGE_REPOS
        why = None if is_pick else reject_reason(row)
        if why:
            key = why.split(":")[0]
            rejects[key] = rejects.get(key, 0) + 1
            continue
        entry = build_row(row)
        author = entry.get("author")
        if author and author in authors_map:
            author_info = authors_map[author]
            if "missing" not in author_info:
                entry["authorInfo"] = {
                    "verified": author_info.get("verified", False),
                    "followers": author_info.get("followers", 0),
                    "isOrg": author_info.get("isOrg", False),
                    "plan": author_info.get("plan"),
                }
        # The extra sweep is thousands of remixes; only what it was run for
        # gets in, so the rest of the store does not churn.
        if row.get("_extra"):
            if not protected(entry):
                rejects["spoza list, nie oficjalny i nie duży"] = rejects.get(
                    "spoza list, nie oficjalny i nie duży", 0) + 1
                continue
            extra_ids.add(entry["repo"])
        kept.append(entry)

    # One row per (model, format): the most trusted, then most downloaded packaging.
    best: dict[tuple[str, str], dict[str, Any]] = {}
    others: dict[tuple[str, str], int] = {}
    for entry in kept:
        key = (norm_key(entry["repo"]), entry["format"])
        others[key] = others.get(key, 0) + 1
        current = best.get(key)
        if current is None:
            best[key] = entry
            continue
        challenger = (entry.get("curated", False), _score_of(entry), entry["downloads"])
        champion = (current.get("curated", False), _score_of(current), current["downloads"])
        if challenger > champion:
            best[key] = entry
    for key, entry in best.items():
        entry["packagings"] = others.get(key, 1)

    rows = sorted(best.values(), key=lambda e: (not e.get("curated"), -int(e["downloads"] or 0)))
    # The cap is for the popularity charts; what the extra sweep brought in
    # (laboratory releases, large models) comes on top of it, and nothing
    # protected is ever cut.
    swept = [r for r in rows if r["repo"] in extra_ids]
    charted = [r for r in rows if r["repo"] not in extra_ids]
    if len(charted) > max_entries:
        keep = [r for r in charted if r.get("curated") or protected(r)]
        rest = [r for r in charted if not (r.get("curated") or protected(r))]
        room = max(0, max_entries - len(keep))
        chosen = {id(r) for r in keep} | {id(r) for r in rest[:room]} | {id(r) for r in swept}
        rejects["ponad limit"] = len(rows) - len(chosen)
        rows = [r for r in rows if id(r) in chosen]
    for rank, entry in enumerate(rows):
        entry["rank"] = rank
    return rows, rejects


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_select(args: argparse.Namespace) -> int:
    authors_path = Path(args.authors) if args.authors else None
    rows, rejects = select(Path(args.raw), max_entries=args.max,
                           extra_path=Path(args.extra) if args.extra else None,
                           authors_path=authors_path)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"zostaje {len(rows)} modeli → {out}", file=sys.stderr)
    print("odrzucone:", file=sys.stderr)
    for why, n in sorted(rejects.items(), key=lambda kv: -kv[1]):
        print(f"   {n:6} {why}", file=sys.stderr)
    cats: dict[str, int] = {}
    for row in rows:
        cats[row["category"]] = cats.get(row["category"], 0) + 1
    print("kategorie:", cats, file=sys.stderr)
    return 0


def cmd_batches(args: argparse.Namespace) -> int:
    """Write the batches the description-writing agents consume."""
    rows = [json.loads(l) for l in Path(args.selected).open(encoding="utf-8")]
    done = set(_load_enrichment(args.skip_enriched or []))
    rows = [r for r in rows
            if (protected(r) if args.protected else args.start <= r["rank"] < args.limit)
            and not r.get("tagline") and r["repo"] not in done]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob(f"{args.prefix}-*.json"):
        stale.unlink()
    n = 0
    for i in range(0, len(rows), args.size):
        chunk = rows[i:i + args.size]
        payload = [{
            "repo": r["repo"],
            "name": r["name"],
            "format": r["format"],
            "category": r["category"],
            "paramsB": r.get("paramsB"),
            "activeParamsB": r.get("activeParamsB"),
            "contextLength": r.get("contextLength"),
            "caps": r.get("caps"),
            "publisher": r.get("publisher"),
            "provenance": r.get("provenance"),
            "baseModel": r.get("baseModel"),
            "downloads": r.get("downloads"),
        } for r in chunk]
        (out_dir / f"{args.prefix}-{n:03d}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        n += 1
    print(f"{n} paczek po {args.size} modeli → {out_dir}", file=sys.stderr)
    return 0


#: Description files pile up over several rewrite rounds, and a repository often
#: has an entry in more than one of them.  Plain alphabetical order decided the
#: winner by accident — ``ogon-001.json`` (an early round) beat ``fix-03.json``
#: (a later one), so a corrected description lost to the one it replaced.  Rounds
#: are ranked here, oldest first; anything unknown sits in the middle, and the
#: last round named wins.
_ENRICHMENT_ROUNDS: tuple[str, ...] = (
    "b2", "b3", "b4", "batch", "ogon", "redo", "redo2", "fix", "new", "tail",
)


def _round_rank(name: str) -> int:
    """Sort key for an enrichment file: later rounds overwrite earlier ones."""
    prefix = name.split("-", 1)[0]
    try:
        return _ENRICHMENT_ROUNDS.index(prefix)
    except ValueError:
        return _ENRICHMENT_ROUNDS.index("batch")      # unknown: treat as an early round


def _load_enrichment(dirs: list[str]) -> dict[str, dict[str, Any]]:
    """Merge every ``{repo: {...}}`` / ``[{repo, ...}]`` file found in ``dirs``.

    Files are read oldest round first (:data:`_ENRICHMENT_ROUNDS`), so the newest
    description of a repository is the one that survives.
    """
    out: dict[str, dict[str, Any]] = {}
    for d in dirs:
        base = Path(d)
        if not base.exists():
            continue
        files = (sorted(base.glob("*.json"), key=lambda p: (_round_rank(p.name), p.name))
                 if base.is_dir() else [base])
        for path in files:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001 - a broken batch must not stop the build
                print(f"  ! {path.name}: {exc}", file=sys.stderr)
                continue
            items = data.get("models") if isinstance(data, dict) else data
            for item in items or []:
                if isinstance(item, dict) and item.get("repo"):
                    out[str(item["repo"])] = item
    return out


def _rejected_repos(path: str | None) -> set[str]:
    """Repositories whose description failed ``scripts/check_enrichment.py``.

    They keep the generated fallback line instead: a description that states a
    parameter count nobody provided is worse than a plain one.
    """
    if not path:
        return set()
    file = Path(path)
    if not file.exists():
        return set()
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return set()
    return {str(item.get("repo")) for items in data.values() for item in items if item.get("repo")}


def cmd_build(args: argparse.Namespace) -> int:
    rows = [json.loads(l) for l in Path(args.selected).open(encoding="utf-8")]
    enrichment = _load_enrichment(args.enriched)
    rejected = _rejected_repos(args.reject)
    if rejected:
        for repo in rejected:
            enrichment.pop(repo, None)
        print(f"odrzucone opisy (nie przeszły kontroli): {len(rejected)}", file=sys.stderr)

    authors_map: dict[str, dict[str, Any]] = {}
    authors_path = Path(args.authors if args.authors else AUTHORS_DEFAULT)
    if authors_path.exists():
        try:
            authors_map = json.loads(authors_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! {authors_path.name}: {exc}", file=sys.stderr)

    written = 0
    rejected_categories = 0
    dropped_rows: set[str] = set()
    for row in rows:
        author = row.get("author")
        if author and author in authors_map:
            author_info = authors_map[author]
            if "missing" not in author_info:
                row["authorInfo"] = {
                    "verified": author_info.get("verified", False),
                    "followers": author_info.get("followers", 0),
                    "isOrg": author_info.get("isOrg", False),
                    "plan": author_info.get("plan"),
                }

        extra = enrichment.get(row["repo"])
        if not extra:
            continue
        tagline = str(extra.get("tagline") or "").strip()
        if tagline and not row.get("tagline"):
            row["tagline"] = tagline
            written += 1
        name = str(extra.get("name") or "").strip()
        if name:
            row["name"] = name
        category = str(extra.get("category") or "").strip()
        if (category and category in CATEGORIES and not row.get("curated")
                and category != row["category"]):
            if category_change_allowed(row["repo"], row.get("name") or "", category):
                row["category"] = category
            else:
                rejected_categories += 1
        if category and category not in CATEGORIES:
            # The writer had the facts in front of it and could not place the
            # model in any shelf of this store ("audio", "wideo"): that is a
            # row the filter should have dropped.
            dropped_rows.add(row["repo"])
            continue
        for key in ("bestFor", "strengths", "watchOut"):
            if extra.get(key):
                row[key] = extra[key]
    if dropped_rows:
        rows = [r for r in rows if r["repo"] not in dropped_rows]
        for rank, row in enumerate(rows):
            row["rank"] = rank
        print(f"usunięte jako nie-modele (werdykt opisujących): {len(dropped_rows)}",
              file=sys.stderr)
    payload = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "Hugging Face",
        "models": rows,
        # What the Hub says about each author (verified organisation, followers):
        # the creator is 40 % of the trust score, and a live search result —
        # not in "models" — is scored against this map too.
        "authors": {
            name: {"verified": bool(info.get("verified")), "followers": int(info.get("followers") or 0),
                   "isOrg": bool(info.get("isOrg"))}
            for name, info in sorted(authors_map.items())
            if not info.get("missing") and not info.get("retry")
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    # mtime=0 keeps the file byte-identical between runs with the same input,
    # so regenerating it does not show up as a change when nothing changed.
    with out.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9,
                                              mtime=0) as fh:
        fh.write(blob)
    if rejected_categories:
        print(f"odrzucone zmiany kategorii bez podstawy w nazwie: {rejected_categories}",
              file=sys.stderr)
    print(f"{len(rows)} modeli ({written} z opisem od modelu, "
          f"{sum(1 for r in rows if r.get('tagline'))} z opisem łącznie) → {out} "
          f"({out.stat().st_size / 1e6:.2f} MB, surowo {len(blob) / 1e6:.2f} MB)", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("select", help="filter the raw dump")
    s.add_argument("--raw", default=RAW_DEFAULT)
    s.add_argument("--out", default=SELECTED_DEFAULT)
    s.add_argument("--max", type=int, default=MAX_ENTRIES)
    s.add_argument("--extra", default=EXTRA_DEFAULT,
                   help="zrzut z hub_fetch.py --extra (pusty napis: bez niego)")
    s.add_argument("--authors", default=AUTHORS_DEFAULT,
                   help="mapa autorów z hub_fetch.py --authors (pusty napis: bez niej)")
    s.set_defaults(func=cmd_select)

    b = sub.add_parser("batches", help="emit work for the description writers")
    b.add_argument("--selected", default=SELECTED_DEFAULT)
    b.add_argument("--out", default=".cache/batches")
    b.add_argument("--size", type=int, default=BATCH_SIZE)
    b.add_argument("--limit", type=int, default=WRITTEN_TAGLINE_LIMIT)
    b.add_argument("--start", type=int, default=0)
    b.add_argument("--prefix", default="batch")
    b.add_argument("--skip-enriched", nargs="*", default=[".cache/enriched"])
    b.add_argument("--protected", action="store_true",
                   help="tylko modele laboratoriów i duże modele, bez względu na pozycję")
    b.set_defaults(func=cmd_batches)

    c = sub.add_parser("build", help="write localcode/hub/data/catalog.json.gz")
    c.add_argument("--selected", default=SELECTED_DEFAULT)
    c.add_argument("--enriched", nargs="*", default=[".cache/enriched"])
    c.add_argument("--out", default=OUT_DEFAULT)
    c.add_argument("--reject", default=".cache/redo.json",
                   help="plik z werdyktem scripts/check_enrichment.py")
    c.add_argument("--authors", default=AUTHORS_DEFAULT,
                   help="mapa autorów z hub_fetch.py --authors (pusty napis: bez niej)")
    c.set_defaults(func=cmd_build)

    args = ap.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
