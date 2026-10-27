"""Who published a model, and who merely repackaged it.

A model on Hugging Face usually exists twice: once in the laboratory's own
repository (``Qwen/Qwen3-8B``) and once, or fifty times, converted to GGUF or MLX
by somebody else (``bartowski/Qwen3-8B-GGUF``).  Both are legitimate; they are
not equally verifiable.  The store therefore says which is which, in one badge.

The classification is derived from data, not from guesswork:

* ``baseModels`` / ``base_model:*`` tags give the upstream repository that a
  conversion was made from, so "Qwen published the weights, Unsloth quantised
  them" is a fact read from the Hub, not an assumption from the name.
* :data:`OFFICIAL_PUBLISHERS` is the list of organisations that publish their own
  models, and :data:`TRUSTED_QUANTIZERS` the packagers whose conversions the
  llama.cpp / MLX communities rely on daily.
* :data:`FAMILY_PUBLISHERS` is the fallback for a repository that carries no
  upstream link: the model family in its name implies the laboratory.

Nothing here downloads anything, so the store can classify thousands of rows
offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Literal

#: How a repository relates to the model it serves.
ProvenanceKind = Literal["official", "trusted", "community", "unknown"]

# --------------------------------------------------------------------------- #
# Organisations that publish their own models
# --------------------------------------------------------------------------- #

#: Hugging Face organisation (lower-case) → name shown in the store.
OFFICIAL_PUBLISHERS: dict[str, str] = {
    # Chinese labs
    "qwen": "Qwen (Alibaba)",
    "alibaba-nlp": "Alibaba NLP",
    "deepseek-ai": "DeepSeek",
    "zai-org": "Z.ai (GLM)",
    "thudm": "THUDM (GLM)",
    "moonshotai": "Moonshot AI (Kimi)",
    "minimaxai": "MiniMax",
    "01-ai": "01.AI (Yi)",
    "baichuan-inc": "Baichuan",
    "internlm": "InternLM",
    "opengvlab": "OpenGVLab",
    "tencent": "Tencent",
    "tencent-hunyuan": "Tencent Hunyuan",
    "baidu": "Baidu (ERNIE)",
    "inclusionai": "InclusionAI (Ling)",
    "openbmb": "OpenBMB (MiniCPM)",
    "stepfun-ai": "StepFun",
    "rednote-hilab": "RedNote hiLab",
    "xiaomimimo": "Xiaomi MiMo",
    "iic": "Alibaba IIC",
    "skywork": "Skywork",
    "tiiuae": "TII (Falcon)",
    # Western labs
    "meta-llama": "Meta",
    "facebook": "Meta AI",
    "google": "Google",
    "google-deepmind": "Google DeepMind",
    "microsoft": "Microsoft",
    "openai": "OpenAI",
    "mistralai": "Mistral AI",
    "nvidia": "NVIDIA",
    "ibm-granite": "IBM Granite",
    "ibm-ai-platform": "IBM",
    "allenai": "Allen Institute (AI2)",
    "cohereforai": "Cohere Labs",
    "coherelabs": "Cohere Labs",
    "ai21labs": "AI21 Labs",
    "databricks": "Databricks",
    "snowflake": "Snowflake",
    "salesforce": "Salesforce",
    "apple": "Apple",
    "amd": "AMD",
    "intel": "Intel",
    "arcee-ai": "Arcee AI",
    "liquidai": "LiquidAI",
    "huggingfacetb": "Hugging Face (SmolLM)",
    "huggingfaceh4": "Hugging Face H4",
    "huggingfacem4": "Hugging Face M4",
    "stabilityai": "Stability AI",
    "black-forest-labs": "Black Forest Labs (FLUX)",
    "lightricks": "Lightricks (LTX)",
    "runwayml": "Runway",
    "compvis": "CompVis",
    "playgroundai": "Playground AI",
    "segmind": "Segmind",
    "upstage": "Upstage (Solar)",
    "lgai-exaone": "LG AI (EXAONE)",
    "kakaocorp": "Kakao",
    "naver-hyperclovax": "NAVER HyperCLOVA X",
    "rakuten": "Rakuten",
    "cyberagent": "CyberAgent",
    "pfnet": "Preferred Networks",
    "sbintuitions": "SB Intuitions",
    "elyza": "ELYZA",
    "tokyotech-llm": "Tokyo Tech",
    "speakleash": "SpeakLeash (Bielik)",
    "utter-project": "UTTER (EuroLLM)",
    "occiglot": "Occiglot",
    "lightonai": "LightOn",
    "kyutai": "Kyutai",
    "jetbrains": "JetBrains",
    "jinaai": "Jina AI",
    "nomic-ai": "Nomic AI",
    "baai": "BAAI",
    "sentence-transformers": "Sentence Transformers",
    "mixedbread-ai": "mixedbread ai",
    "intfloat": "intfloat (E5)",
    "answerdotai": "Answer.AI",
    "teknium": "Teknium (Nous)",
    "nousresearch": "Nous Research",
    "cognitivecomputations": "Cognitive Computations",
    "perplexity-ai": "Perplexity AI",
    "reflection-ai": "Reflection AI",
    "xai-org": "xAI",
    "servicenow-ai": "ServiceNow AI",
    "deepcogito": "Deep Cogito",
    "inceptionai": "Inception (Jais)",
    "prithivmlmods": "Prithiv ML",
    "agentica-org": "Agentica",
    "open-thoughts": "OpenThoughts",
    "nickypro": "Nicky Pro",
    "numind": "NuMind",
    "petals-team": "Petals",
    "togethercomputer": "Together AI",
    "bigcode": "BigCode",
    "bigscience": "BigScience",
    "eleutherai": "EleutherAI",
    "stanford-crfm": "Stanford CRFM",
    "vikhyatk": "Vikhyat (Moondream)",
    "moondream": "Moondream",
    "openchat": "OpenChat",
    "wizardlmteam": "WizardLM",
    "codellama": "Meta (Code Llama)",
    "defog": "Defog",
    "sarvamai": "Sarvam AI",
    "krutrim-ai-labs": "Krutrim",
    "ai4bharat": "AI4Bharat",
    "aisingapore": "AI Singapore (SEA-LION)",
    "scb10x": "SCB 10X (Typhoon)",
    "vilm": "VILM",
    "qihoo360": "Qihoo 360",
    "ictnlp": "ICT NLP",
    "shanghaiailab": "Shanghai AI Lab",
    "ace-step": "ACE Studio",
    "fishaudio": "Fish Audio",
    "suno": "Suno",
    "hexgrad": "hexgrad (Kokoro)",
    "canopylabs": "Canopy Labs",
    "sesame": "Sesame",
    "neutts": "NeuTTS",
    "pytorch": "PyTorch",
}

# --------------------------------------------------------------------------- #
# Packagers: they convert other people's models to GGUF / MLX
# --------------------------------------------------------------------------- #

#: Organisation → label, for conversions the local-model community relies on.
TRUSTED_QUANTIZERS: dict[str, str] = {
    "ggml-org": "zespół llama.cpp",
    "lmstudio-community": "LM Studio Community",
    "mlx-community": "MLX Community",
    "unsloth": "Unsloth",
    "bartowski": "bartowski",
    "mradermacher": "mradermacher",
    "quantfactory": "QuantFactory",
    "thebloke": "TheBloke",
    "second-state": "Second State",
    "gpustack": "GPUStack",
    "tensorblock": "TensorBlock",
    "devquasar": "DevQuasar",
    "mungert": "Mungert",
    "nightmedia": "nightmedia",
    "inferencerlabs": "Inferencer Labs",
    "featherless-ai-quants": "Featherless AI",
    "legraphista": "legraphista",
    "richarderkhov": "richardErkhov",
    "dranger003": "dranger003",
    "qwp4w3hyb": "qwp4w3hyb",
    "city96": "city96",
    "calcuis": "calcuis",
    "leejet": "leejet (stable-diffusion.cpp)",
    "filipstrand": "filipstrand (mflux)",
    "dhairyashil": "dhairyashil (mflux)",
    "runpod": "RunPod",
    "argmaxinc": "Argmax",
    "cnfusion": "cnfusion",
    "goekdenizguelmez": "GoekdenizGuelmez",
    "anthracite-org": "Anthracite",
    "backyardai": "Backyard AI",
    "koboldcpp": "KoboldCpp",
    "brittlewis12": "brittlewis12",
    "prunaai": "Pruna AI",
    "modelcloud": "ModelCloud",
    "neuralmagic": "Neural Magic",
    "redhatai": "Red Hat AI",
    "opencsg": "OpenCSG",
    "maziyarpanahi": "MaziyarPanahi",
    "lmstudio-ai": "LM Studio",
    "nexaai": "Nexa AI",
    "ramonvc": "ramonvc",
    "sanctumai": "Sanctum AI",
    "internlm2": "InternLM",
    "menlo": "Menlo Research",
    "janhq": "Jan",
    "cortexso": "Cortex",
}

#: Model-family fragment → the laboratory that publishes that family.
#: Used only when the Hub gives no upstream repository for a conversion.
FAMILY_PUBLISHERS: tuple[tuple[str, str], ...] = (
    ("qwen", "qwen"),
    ("qwq", "qwen"),
    ("deepseek", "deepseek-ai"),
    ("llama", "meta-llama"),
    ("codellama", "meta-llama"),
    ("gemma", "google"),
    ("codegemma", "google"),
    ("paligemma", "google"),
    ("t5gemma", "google"),
    ("phi-", "microsoft"),
    ("phi3", "microsoft"),
    ("phi4", "microsoft"),
    ("bitnet", "microsoft"),
    ("gpt-oss", "openai"),
    ("whisper", "openai"),
    ("mistral", "mistralai"),
    ("mixtral", "mistralai"),
    ("ministral", "mistralai"),
    ("devstral", "mistralai"),
    ("codestral", "mistralai"),
    ("magistral", "mistralai"),
    ("pixtral", "mistralai"),
    ("voxtral", "mistralai"),
    ("nemotron", "nvidia"),
    ("granite", "ibm-granite"),
    ("olmo", "allenai"),
    ("molmo", "allenai"),
    ("tulu", "allenai"),
    ("command-r", "coherelabs"),
    ("command-a", "coherelabs"),
    ("aya-", "coherelabs"),
    ("jamba", "ai21labs"),
    ("dbrx", "databricks"),
    ("arctic", "snowflake"),
    ("smollm", "huggingfacetb"),
    ("smolvlm", "huggingfacetb"),
    ("smoldocling", "huggingfacetb"),
    ("zephyr", "huggingfaceh4"),
    ("idefics", "huggingfacem4"),
    ("lfm2", "liquidai"),
    ("lfm-", "liquidai"),
    ("glm-", "zai-org"),
    ("chatglm", "thudm"),
    ("codegeex", "thudm"),
    ("kimi", "moonshotai"),
    ("minimax", "minimaxai"),
    ("yi-", "01-ai"),
    ("internlm", "internlm"),
    ("internvl", "opengvlab"),
    ("hunyuan", "tencent"),
    ("ernie", "baidu"),
    ("ling-", "inclusionai"),
    ("ring-", "inclusionai"),
    ("minicpm", "openbmb"),
    ("falcon", "tiiuae"),
    ("exaone", "lgai-exaone"),
    ("solar-", "upstage"),
    ("bielik", "speakleash"),
    ("eurollm", "utter-project"),
    ("seed-", "bytedance-seed"),
    ("apertus", "swiss-ai"),
    ("trillion", "trillionlabs"),
    ("sarashina", "sbintuitions"),
    ("plamo", "pfnet"),
    ("swallow", "tokyotech-llm"),
    ("sea-lion", "aisingapore"),
    ("typhoon", "scb10x"),
    ("sarvam", "sarvamai"),
    ("jais", "inceptionai"),
    ("stablelm", "stabilityai"),
    ("stable-diffusion", "stabilityai"),
    ("sdxl", "stabilityai"),
    ("sd3", "stabilityai"),
    ("flux", "black-forest-labs"),
    ("qwen-image", "qwen"),
    ("wan2", "alibaba-nlp"),
    ("moondream", "vikhyatk"),
    ("nomic-embed", "nomic-ai"),
    ("jina-", "jinaai"),
    ("bge-", "baai"),
    ("e5-", "intfloat"),
    ("mxbai", "mixedbread-ai"),
    ("gte-", "alibaba-nlp"),
    ("all-minilm", "sentence-transformers"),
    ("modernbert", "answerdotai"),
    ("hermes", "nousresearch"),
    ("dolphin", "cognitivecomputations"),
    ("kokoro", "hexgrad"),
    ("orpheus", "canopylabs"),
    ("csm-", "sesame"),
    ("seamless", "facebook"),
    ("wav2vec", "facebook"),
    ("mms-", "facebook"),
    ("starcoder", "bigcode"),
    ("santacoder", "bigcode"),
    ("bloom", "bigscience"),
    ("pythia", "eleutherai"),
    ("gpt-neox", "eleutherai"),
    ("gpt-j", "eleutherai"),
)

#: Packaging suffixes/infixes stripped before matching a family.
_PACKAGING_RE = re.compile(
    r"[-_.]?(?:gguf|mlx|mflux|awq|gptq|bnb|int4|int8|fp8|fp16|bf16|mxfp4|nf4|"
    r"\d+bits?|q\d[a-z0-9_]*|imatrix|quantized|quants?|converted|onnx|openvino|"
    r"safetensors|ct2|ctranslate2|exl2|exl3|hqq|dwq|abliterated|uncensored)\b",
    re.IGNORECASE,
)


#: Where the laboratories' own marks live (``scripts/fetch_logos.py`` fills it
#: from each organisation's Hugging Face avatar — nothing is drawn or guessed).
LOGO_DIR = Path(__file__).resolve().parent / "data" / "logos"


@lru_cache(maxsize=1)
def available_logos() -> frozenset[str]:
    """Organisation slugs that have a logo on disk."""
    try:
        return frozenset(p.stem for p in LOGO_DIR.glob("*.png"))
    except OSError:
        return frozenset()


def logo_for(org: str | None) -> str | None:
    """The logo slug for an organisation, or ``None`` when we have no mark.

    Only laboratories get one: the store shows the brand behind the *weights*,
    so a community conversion of a DeepSeek model shows DeepSeek's whale, and a
    model nobody can attribute shows nothing at all.
    """
    slug = normalize_org(org)
    return slug if slug and slug in available_logos() else None


def normalize_org(author: str | None) -> str:
    return (author or "").strip().lower()


def is_official_org(author: str | None) -> bool:
    return normalize_org(author) in OFFICIAL_PUBLISHERS


def is_trusted_quantizer(author: str | None) -> bool:
    return normalize_org(author) in TRUSTED_QUANTIZERS


def publisher_display(author: str | None) -> str:
    """Human-readable name of an organisation (falls back to the raw handle)."""
    org = normalize_org(author)
    return OFFICIAL_PUBLISHERS.get(org) or TRUSTED_QUANTIZERS.get(org) or (author or "")


def family_publisher(name: str) -> str | None:
    """The laboratory implied by a model name, e.g. ``"Qwen3-8B-GGUF"`` → ``qwen``."""
    haystack = _PACKAGING_RE.sub("", name.replace("_", "-")).lower()
    hits = [(frag, org) for frag, org in FAMILY_PUBLISHERS if frag in haystack]
    if not hits:
        return None
    # Longest fragment wins: "codellama" before "llama", "qwen-image" before "qwen".
    return max(hits, key=lambda h: len(h[0]))[1]


def upstream_repo(base_models: Any, tags: Iterable[str] | None = None) -> str | None:
    """Repository a conversion was made from, from ``baseModels`` or the tags."""
    if isinstance(base_models, dict):
        models = base_models.get("models") or []
        for model in models:
            repo = model.get("id") if isinstance(model, dict) else model
            if repo and "/" in str(repo):
                return str(repo)
    for tag in tags or ():
        text = str(tag)
        if text.startswith("base_model:"):
            repo = text.split(":")[-1]
            if "/" in repo:
                return repo
    return None


@dataclass
class Provenance:
    """Where a repository's weights come from, in store vocabulary."""

    kind: ProvenanceKind
    publisher: str                  # display name of the model's laboratory ("Qwen (Alibaba)")
    publisher_org: str | None = None  # its Hugging Face organisation ("qwen"), for the logo
    packager: str | None = None     # who converted it, when that differs
    upstream: str | None = None     # upstream repository id, when known
    official: bool = False          # True only for the laboratory's own repository
    lab_known: bool = False         # the weights' author is a laboratory we recognise
    label: str = ""                 # short badge, Polish
    detail: str = ""                # one sentence, Polish
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "publisher": self.publisher,
            "labKnown": self.lab_known,
            # The mark belongs to the laboratory, so it is shown only on the
            # laboratory's own repository.  A conversion may be excellent and
            # still is not Google's upload; putting Google's G on it would say
            # something we cannot verify.
            "logo": logo_for(self.publisher_org) if self.official else None,
            "packager": self.packager,
            "upstream": self.upstream,
            "official": self.official,
            "label": self.label,
            "detail": self.detail,
        }


def classify(repo: str, *, base_models: Any = None, tags: Iterable[str] | None = None) -> Provenance:
    """Classify a repository as the publisher's own, a trusted conversion, or neither.

    ``base_models`` is the Hub's ``baseModels`` object (or a list of ids) and
    ``tags`` the raw tag list, which carries ``base_model:<repo>`` as a fallback.
    """
    author, _, name = repo.partition("/")
    org = normalize_org(author)
    upstream = upstream_repo(base_models, tags)
    upstream_org = normalize_org(upstream.partition("/")[0]) if upstream else None
    implied = family_publisher(name)

    # The laboratory's own repository: either no conversion at all, or a
    # conversion the laboratory published itself (Qwen/Qwen3-8B-GGUF).
    if org in OFFICIAL_PUBLISHERS and (upstream_org in (None, org) or implied == org):
        display = OFFICIAL_PUBLISHERS[org]
        return Provenance(
            kind="official", publisher=display, publisher_org=org, upstream=upstream,
            official=True, lab_known=True,
            label="źródło oficjalne",
            detail=f"Repozytorium należy do {display} — twórcy tego modelu.",
            reasons=[f"Wydawca modelu ({display}) publikuje ten plik sam."],
        )

    lab_org = upstream_org or implied
    lab = OFFICIAL_PUBLISHERS.get(lab_org or "", "")

    if org in TRUSTED_QUANTIZERS:
        packager = TRUSTED_QUANTIZERS[org]
        if lab:
            detail = (f"Konwersja przygotowana przez {packager} na podstawie modelu "
                      f"od {lab}" + (f" ({upstream})." if upstream else "."))
            reasons = [f"Znany zespół konwertujący ({packager}).",
                       f"Wagi pochodzą od {lab}."]
        else:
            detail = f"Konwersja przygotowana przez {packager}."
            reasons = [f"Znany zespół konwertujący ({packager})."]
        return Provenance(
            kind="trusted", publisher=lab or packager, publisher_org=(lab_org if lab else None),
            packager=packager, upstream=upstream, official=False, lab_known=bool(lab),
            label="sprawdzona konwersja", detail=detail, reasons=reasons,
        )

    if lab:
        return Provenance(
            kind="community", publisher=lab, publisher_org=lab_org, packager=author or None,
            upstream=upstream, official=False, lab_known=True, label="konwersja społeczności",
            detail=(f"Wagi pochodzą od {lab}, ale plik przygotował niezależny autor "
                    f"({author})."),
            reasons=[f"Wagi pochodzą od {lab}.",
                     f"Plik przygotował niezależny autor ({author})."],
        )

    if upstream:
        # The repository names its source, but that source is not a laboratory
        # we know — say so instead of claiming there is no source at all.
        detail = (f"Repozytorium prowadzi {author}; wagi pochodzą z {upstream}, ale tego "
                  "wydawcy nie ma na liście znanych laboratoriów.")
        reasons = [f"Model źródłowy: {upstream}."]
    else:
        detail = (f"Nie udało się ustalić, czyj to model — repozytorium prowadzi {author} "
                  "i nie wskazuje modelu źródłowego.")
        reasons = []
    return Provenance(
        kind="unknown", publisher=author or "nieznany", packager=author or None,
        upstream=upstream, official=False, label="źródło nieznane",
        detail=detail, reasons=reasons,
    )
