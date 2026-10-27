"""Staff picks: models that work well for a coding agent and chat on Apple Silicon.

Curated in September 2026 from the LM Studio catalog, community rankings of local
coding models for 16–64 GB Macs and the Hugging Face download charts; every
repository id, the recommended quantisation file and its size were verified with
``HfApi.model_info`` (see ``tests/hub/test_curated.py`` — network test).

Fields
------
* ``repo`` — Hugging Face repository id.
* ``format`` — ``"gguf"`` (llama.cpp) or ``"mlx"`` (mlx-lm).
* ``category`` — ``code`` | ``chat`` | ``reasoning`` | ``vision`` | ``embedding``.
* ``paramsB`` / ``activeParamsB`` — total / active (MoE) parameters in billions.
* ``minRamGb`` — smallest Mac (unified memory) on which the recommended variant
  runs comfortably with a 16k context (weights + KV cache + overhead within ~70 %
  of RAM, the default model budget).
* ``recommendedQuant`` — GGUF variant label as produced by
  :func:`localcode.hub.metadata.parse_quant` (e.g. ``Q4_K_M``, ``UD-Q4_K_M``) or
  the MLX quantisation label (``4bit``, ``MXFP4``).
* ``note`` — one line for the UI (Polish).

Tiers: ≤4B for 8–16 GB machines, 7–14B for 16–24 GB, MoE / 24–35B for 32–64 GB,
plus GGUF embedding models for the chatbot's knowledge base (RAG).
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

Category = Literal["code", "chat", "reasoning", "vision", "embedding"]
#: Image models live in their own list (:data:`CURATED_IMAGES`) so that the
#: language picks keep their shape; ``category`` is always ``"image"`` there.
ImageCategory = Literal["image"]


class CuratedModel(TypedDict, total=False):
    repo: str
    format: Literal["gguf", "mlx"]
    category: Category
    paramsB: float
    activeParamsB: float | None
    minRamGb: int
    recommendedQuant: str
    note: str


CURATED: list[CuratedModel] = [
    # ---- small: 8–16 GB ------------------------------------------------------------
    {
        "repo": "unsloth/Qwen3-4B-Instruct-2507-GGUF", "format": "gguf", "category": "code",
        "paramsB": 4.0, "activeParamsB": None, "minRamGb": 8, "recommendedQuant": "Q4_K_M",
        "note": "Mały i szybki model z dobrym wywoływaniem narzędzi — najlepszy start dla agenta na 8–16 GB RAM.",
    },
    {
        "repo": "mlx-community/Qwen3-4B-Instruct-2507-4bit", "format": "mlx", "category": "code",
        "paramsB": 4.0, "activeParamsB": None, "minRamGb": 8, "recommendedQuant": "4bit",
        "note": "Qwen3 4B Instruct w MLX — najszybszy mały agent na Apple Silicon.",
    },
    {
        "repo": "unsloth/Qwen3.5-4B-GGUF", "format": "gguf", "category": "chat",
        "paramsB": 4.2, "activeParamsB": None, "minRamGb": 8, "recommendedQuant": "Q4_K_M",
        "note": "Nowszy mały Qwen3.5 z trybem myślenia i obsługą obrazów (mmproj).",
    },
    {
        "repo": "unsloth/Qwen3.5-2B-GGUF", "format": "gguf", "category": "chat",
        "paramsB": 1.9, "activeParamsB": None, "minRamGb": 8, "recommendedQuant": "Q4_K_M",
        "note": "Bardzo lekki model do czatu i prostych zadań na słabszych Macach.",
    },
    {
        "repo": "LiquidAI/LFM2.5-1.2B-Instruct-GGUF", "format": "gguf", "category": "chat",
        "paramsB": 1.2, "activeParamsB": None, "minRamGb": 8, "recommendedQuant": "Q8_0",
        "note": "Hybrydowy mini-model LiquidAI — błyskawiczny, a długi kontekst prawie nie zajmuje pamięci.",
    },
    {
        "repo": "unsloth/gemma-4-E4B-it-GGUF", "format": "gguf", "category": "vision",
        "paramsB": 7.5, "activeParamsB": 4.0, "minRamGb": 12, "recommendedQuant": "Q4_K_M",
        "note": "Gemma 4 E4B: czat, rozumienie obrazów i tryb myślenia przy małym zużyciu pamięci.",
    },
    {
        "repo": "lmstudio-community/gemma-4-E4B-it-MLX-4bit", "format": "mlx", "category": "chat",
        "paramsB": 8.0, "activeParamsB": 4.0, "minRamGb": 12, "recommendedQuant": "4bit",
        "note": "Gemma 4 E4B w MLX — szybki, wielojęzyczny czat (dobrze radzi sobie z polskim).",
    },
    # ---- mid: 16–24 GB -------------------------------------------------------------
    {
        "repo": "unsloth/Qwen3.5-9B-GGUF", "format": "gguf", "category": "code",
        "paramsB": 9.0, "activeParamsB": None, "minRamGb": 16, "recommendedQuant": "Q4_K_M",
        "note": "Qwen3.5 9B: rozsądny kompromis jakości i szybkości dla agenta i czatu na 16 GB.",
    },
    {
        "repo": "lmstudio-community/Qwen3.5-9B-MLX-4bit", "format": "mlx", "category": "code",
        "paramsB": 9.4, "activeParamsB": None, "minRamGb": 16, "recommendedQuant": "4bit",
        "note": "Qwen3.5 9B w MLX — szybkie generowanie kodu na Apple Silicon.",
    },
    {
        "repo": "unsloth/gemma-4-12b-it-GGUF", "format": "gguf", "category": "chat",
        "paramsB": 11.9, "activeParamsB": None, "minRamGb": 16, "recommendedQuant": "Q4_K_M",
        "note": "Gemma 4 12B: bardzo dobry czat po polsku, obrazy i długi kontekst.",
    },
    {
        "repo": "lmstudio-community/gemma-4-12B-it-MLX-4bit", "format": "mlx", "category": "chat",
        "paramsB": 12.0, "activeParamsB": None, "minRamGb": 16, "recommendedQuant": "4bit",
        "note": "Gemma 4 12B w MLX — naturalny, wielojęzyczny czat.",
    },
    {
        "repo": "lmstudio-community/Ministral-3-14B-Reasoning-2512-GGUF", "format": "gguf", "category": "reasoning",
        "paramsB": 13.5, "activeParamsB": None, "minRamGb": 16, "recommendedQuant": "Q4_K_M",
        "note": "Ministral 3 14B z rozumowaniem — dobry do analiz, planowania i matematyki.",
    },
    {
        "repo": "Qwen/Qwen3-VL-8B-Instruct-GGUF", "format": "gguf", "category": "vision",
        "paramsB": 8.2, "activeParamsB": None, "minRamGb": 16, "recommendedQuant": "Q4_K_M",
        "note": "Oficjalny Qwen3-VL 8B: zrzuty ekranu, OCR, diagramy i dokumenty.",
    },
    {
        "repo": "ibm-granite/granite-4.1-8b-GGUF", "format": "gguf", "category": "chat",
        "paramsB": 8.8, "activeParamsB": None, "minRamGb": 16, "recommendedQuant": "Q4_K_M",
        "note": "IBM Granite 4.1 8B: rzetelne wywoływanie narzędzi, RAG i odpowiedzi w JSON.",
    },
    {
        "repo": "ggml-org/gpt-oss-20b-GGUF", "format": "gguf", "category": "reasoning",
        "paramsB": 20.9, "activeParamsB": 3.6, "minRamGb": 24, "recommendedQuant": "MXFP4",
        "note": "gpt-oss 20B od OpenAI: mocne rozumowanie i narzędzia, regulowany poziom myślenia.",
    },
    {
        "repo": "mlx-community/gpt-oss-20b-MXFP4-Q8", "format": "mlx", "category": "reasoning",
        "paramsB": 20.9, "activeParamsB": 3.6, "minRamGb": 24, "recommendedQuant": "MXFP4",
        "note": "gpt-oss 20B w MLX (MXFP4) — szybkie rozumowanie na Apple Silicon.",
    },
    # ---- large / MoE: 32–64 GB -----------------------------------------------------
    {
        "repo": "unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF", "format": "gguf", "category": "code",
        "paramsB": 30.5, "activeParamsB": 3.3, "minRamGb": 32, "recommendedQuant": "Q4_K_M",
        "note": "Najlepszy lokalny agent kodujący na 32 GB — MoE z ~3B aktywnych parametrów, bardzo szybki.",
    },
    {
        "repo": "lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-MLX-4bit", "format": "mlx", "category": "code",
        "paramsB": 30.5, "activeParamsB": 3.3, "minRamGb": 32, "recommendedQuant": "4bit",
        "note": "Qwen3-Coder 30B-A3B w MLX — najszybszy duży agent kodujący na Macu.",
    },
    {
        "repo": "unsloth/Qwen3.8-27B-GGUF", "format": "gguf", "category": "code",
        "paramsB": 27.3, "activeParamsB": None, "minRamGb": 32, "recommendedQuant": "UD-Q4_K_M",
        "note": "Qwen3.8 27B: najmocniejszy gęsty model do kodu i zadań agentowych (wolniejszy od MoE).",
    },
    {
        "repo": "lmstudio-community/Qwen3.8-27B-MLX-4bit", "format": "mlx", "category": "code",
        "paramsB": 27.4, "activeParamsB": None, "minRamGb": 32, "recommendedQuant": "4bit",
        "note": "Qwen3.8 27B w MLX — topowa jakość kodu i długi kontekst na 32–64 GB.",
    },
    {
        "repo": "unsloth/Qwen3.6-35B-A3B-GGUF", "format": "gguf", "category": "reasoning",
        "paramsB": 34.7, "activeParamsB": 3.0, "minRamGb": 48, "recommendedQuant": "UD-Q4_K_M",
        "note": "Qwen3.6 35B-A3B: szybki MoE z trybem myślenia do kodu i analiz; wygodnie od 48 GB.",
    },
    {
        "repo": "unsloth/Devstral-Small-2-24B-Instruct-2512-GGUF", "format": "gguf", "category": "code",
        "paramsB": 23.6, "activeParamsB": None, "minRamGb": 32, "recommendedQuant": "Q4_K_M",
        "note": "Devstral Small 2 od Mistrala: agentowe kodowanie i eksploracja repozytoriów.",
    },
    {
        "repo": "mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit", "format": "mlx", "category": "code",
        "paramsB": 24.0, "activeParamsB": None, "minRamGb": 32, "recommendedQuant": "4bit",
        "note": "Devstral Small 2 w MLX — agent kodujący Mistrala na Apple Silicon.",
    },
    {
        "repo": "unsloth/GLM-4.7-Flash-GGUF", "format": "gguf", "category": "code",
        "paramsB": 29.9, "activeParamsB": 3.9, "minRamGb": 32, "recommendedQuant": "Q4_K_M",
        "note": "GLM-4.7-Flash: MoE wyspecjalizowany w kodzie i wywoływaniu narzędzi, oszczędny kontekst (MLA).",
    },
    {
        "repo": "unsloth/gemma-4-26B-A4B-it-GGUF", "format": "gguf", "category": "vision",
        "paramsB": 25.2, "activeParamsB": 4.0, "minRamGb": 32, "recommendedQuant": "UD-Q4_K_M",
        "note": "Gemma 4 26B-A4B: szybki MoE z obsługą obrazów i trybem myślenia.",
    },
    {
        "repo": "lmstudio-community/NVIDIA-Nemotron-3-Nano-30B-A3B-MLX-4bit", "format": "mlx", "category": "reasoning",
        "paramsB": 31.6, "activeParamsB": 3.0, "minRamGb": 32, "recommendedQuant": "4bit",
        "note": "Nemotron 3 Nano: hybryda Mamba-MoE — bardzo tani długi kontekst i szybkie rozumowanie.",
    },
    {
        "repo": "unsloth/Qwen3-Coder-Next-GGUF", "format": "gguf", "category": "code",
        "paramsB": 79.7, "activeParamsB": 3.0, "minRamGb": 64, "recommendedQuant": "UD-Q3_K_XL",
        "note": "Qwen3-Coder-Next 80B-A3B: najmocniejszy agent kodujący dla Maców z 64 GB.",
    },
    # ---- embeddings for RAG (GGUF) -------------------------------------------------
    {
        "repo": "Qwen/Qwen3-Embedding-0.6B-GGUF", "format": "gguf", "category": "embedding",
        "paramsB": 0.6, "activeParamsB": None, "minRamGb": 8, "recommendedQuant": "Q8_0",
        "note": "Wielojęzyczne embeddingi Qwen3 (także po polsku) do bazy wiedzy czatu.",
    },
    {
        "repo": "ggml-org/embeddinggemma-300M-GGUF", "format": "gguf", "category": "embedding",
        "paramsB": 0.3, "activeParamsB": None, "minRamGb": 8, "recommendedQuant": "Q8_0",
        "note": "EmbeddingGemma 300M od Google — lekkie, wielojęzyczne embeddingi do RAG.",
    },
    {
        "repo": "nomic-ai/nomic-embed-text-v1.5-GGUF", "format": "gguf", "category": "embedding",
        "paramsB": 0.137, "activeParamsB": None, "minRamGb": 8, "recommendedQuant": "Q8_0",
        "note": "Sprawdzone, bardzo małe (150 MB) embeddingi do RAG — najlepsze dla tekstów angielskich.",
    },
]

CURATED_REPOS: frozenset[str] = frozenset(m["repo"] for m in CURATED)


class CuratedImageModel(TypedDict, total=False):
    """One staff pick for local image generation (see ``docs/IMAGES.md``)."""

    repo: str
    format: Literal["gguf", "mlx"]
    category: ImageCategory
    imageFamily: str                # flux | flux2 | qwen-image | sd1 | sdxl | sd3
    minRamGb: int
    recommendedQuant: str
    sizeGb: float                   # what the recommended files take on disk
    steps: int                      # sensible default step count
    files: list[str]                # recommended files (GGUF: model + companions)
    note: str


# Image generation (text-to-image).  Every repository id, the recommended file
# and its size were verified with ``HfApi.model_info`` in September 2026 (see
# ``tests/images/test_image_detection.py`` — network test).  The split GGUF
# setups (FLUX, SD 3.5) ship their text encoders and autoencoder in the same
# repository, so one download is always enough to generate.
CURATED_IMAGES: list[CuratedImageModel] = [
    # ---- lightest: works on 8-16 GB ------------------------------------------------
    {
        "repo": "second-state/stable-diffusion-v1-5-GGUF", "format": "gguf", "category": "image",
        "imageFamily": "sd1", "minRamGb": 8, "recommendedQuant": "Q8_0", "sizeGb": 1.8, "steps": 20,
        "files": ["stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf"],
        "note": "Stable Diffusion 1.5 — najlżejszy model obrazów, jeden plik, działa na każdym Macu. "
                "Wersja Q8_0: przy Q4_0 UNet tej rodziny gubi detale i zwraca rozmytą plamę.",
    },
    {
        "repo": "gpustack/stable-diffusion-xl-base-1.0-GGUF", "format": "gguf", "category": "image",
        "imageFamily": "sdxl", "minRamGb": 16, "recommendedQuant": "Q8_0", "sizeGb": 5.0, "steps": 25,
        "files": ["stable-diffusion-xl-base-1.0-Q8_0.gguf"],
        "note": "SDXL 1.0 — dobre zdjęcia i grafiki 1024×1024 bez dodatkowych plików.",
    },
    {
        "repo": "Runpod/FLUX.2-klein-4B-mflux-4bit", "format": "mlx", "category": "image",
        "imageFamily": "flux2", "minRamGb": 16, "recommendedQuant": "4bit", "sizeGb": 4.6, "steps": 4,
        "note": "FLUX.2 klein 4B w MLX (4-bit) — najszybszy sensowny generator obrazów na Apple Silicon.",
    },
    # ---- 16-24 GB ------------------------------------------------------------------
    {
        "repo": "second-state/stable-diffusion-3.5-medium-GGUF", "format": "gguf", "category": "image",
        "imageFamily": "sd3", "minRamGb": 20, "recommendedQuant": "Q8_0", "sizeGb": 6.4, "steps": 28,
        "files": ["sd3.5_medium-Q8_0.gguf", "clip_l-Q8_0.gguf", "clip_g-Q8_0.gguf", "t5xxl-Q4_0.gguf"],
        "note": "Stable Diffusion 3.5 Medium — bardzo dobre trzymanie się promptu i napisy na obrazie.",
    },
    {
        "repo": "mlx-community/flux2-klein-9b-4bit", "format": "mlx", "category": "image",
        "imageFamily": "flux2", "minRamGb": 24, "recommendedQuant": "4bit", "sizeGb": 9.5, "steps": 4,
        "note": "FLUX.2 klein 9B (4-bit) — wyraźnie lepsze detale niż wersja 4B, wciąż 4 kroki.",
    },
    {
        "repo": "dhairyashil/FLUX.1-schnell-mflux-4bit", "format": "mlx", "category": "image",
        "imageFamily": "flux", "minRamGb": 24, "recommendedQuant": "4bit", "sizeGb": 9.6, "steps": 4,
        "note": "FLUX.1 schnell (4-bit, MLX) — klasyk: ładne obrazy w 4 krokach, licencja Apache.",
    },
    {
        "repo": "second-state/FLUX.1-schnell-GGUF", "format": "gguf", "category": "image",
        "imageFamily": "flux", "minRamGb": 24, "recommendedQuant": "Q4_0", "sizeGb": 10.0, "steps": 4,
        "files": ["flux1-schnell-Q4_0.gguf", "clip_l-Q8_0.gguf", "t5xxl-Q4_0.gguf", "ae.safetensors"],
        "note": "FLUX.1 schnell w GGUF — ta sama jakość przez stable-diffusion.cpp, z enkoderami w komplecie.",
    },
    # ---- 24-32 GB ------------------------------------------------------------------
    {
        "repo": "filipstrand/FLUX.1-Krea-dev-mflux-4bit", "format": "mlx", "category": "image",
        "imageFamily": "flux", "minRamGb": 24, "recommendedQuant": "4bit", "sizeGb": 9.6, "steps": 25,
        "note": "FLUX.1 Krea dev (4-bit) — fotorealizm bez „plastikowego” wyglądu, 20–30 kroków.",
    },
    {
        "repo": "second-state/FLUX.1-dev-GGUF", "format": "gguf", "category": "image",
        "imageFamily": "flux", "minRamGb": 32, "recommendedQuant": "Q4_0", "sizeGb": 10.3, "steps": 25,
        "files": ["flux1-dev-Q4_0.gguf", "clip_l-Q8_0.gguf", "t5xxl-Q4_0.gguf", "ae.safetensors"],
        "note": "FLUX.1 dev w GGUF — najwyższa jakość z rodziny FLUX.1, wolniejsze 20–30 kroków.",
    },
    {
        "repo": "mlx-community/Qwen-Image-2512-4bit", "format": "mlx", "category": "image",
        "imageFamily": "qwen-image", "minRamGb": 48, "recommendedQuant": "4bit", "sizeGb": 25.9, "steps": 20,
        "note": "Qwen-Image 2512 (4-bit) — świetnie renderuje tekst, także polski; wymaga dużo RAM.",
    },
]

CURATED_IMAGE_REPOS: frozenset[str] = frozenset(m["repo"] for m in CURATED_IMAGES)


def curated_models(*, include_images: bool = False) -> list[CuratedModel]:
    """A copy of the staff-picks list (safe to mutate).

    Image models are a separate list; pass ``include_images=True`` to get them
    appended (the "Odkrywaj" screen does that for its "Obrazy" filter).
    """
    models = [dict(m) for m in CURATED]
    if include_images:
        models += [dict(m) for m in CURATED_IMAGES]
    return models  # type: ignore[return-value]


def curated_image_models() -> list[CuratedImageModel]:
    """A copy of the image staff-picks list (safe to mutate)."""
    return [dict(m) for m in CURATED_IMAGES]  # type: ignore[misc]


def curated_entry(repo: str) -> CuratedModel | None:
    """A curated entry by repository id (language models first, then images)."""
    for m in [*CURATED, *CURATED_IMAGES]:
        if m["repo"].lower() == repo.lower():
            return dict(m)  # type: ignore[return-value]
    return None


def to_hub_model(entry: CuratedModel) -> dict[str, Any]:
    """Curated entry → ``HubModel`` DTO (offline: no download/like counts)."""
    org, _, name = entry["repo"].partition("/")
    tags = [entry["format"], entry["category"]]
    if entry.get("imageFamily"):
        tags.append(str(entry["imageFamily"]))
    return {
        "id": entry["repo"],
        "author": org,
        "name": name,
        "format": entry["format"],
        "downloads": None,
        "likes": None,
        "updatedAt": None,
        "tags": tags,
        "pipelineTag": _pipeline_tag(entry["category"]),
        "paramsB": entry.get("paramsB"),
        "activeParamsB": entry.get("activeParamsB"),
        "gated": False,
        "curated": True,
        "description": entry.get("note"),
        "category": entry["category"],
        "minRamGb": entry.get("minRamGb"),
        "recommendedQuant": entry.get("recommendedQuant"),
        "imageFamily": entry.get("imageFamily"),
    }


def _pipeline_tag(category: str) -> str:
    if category == "embedding":
        return "feature-extraction"
    if category == "image":
        return "text-to-image"
    return "text-generation"
