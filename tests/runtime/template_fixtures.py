"""Real chat templates used by the template / tool-call / formatter tests.

The templates of the families LocalCode has to support are checked into
``tests/fixtures/templates/<slug>.jinja`` so the whole suite runs offline. They
come from three places:

* Hugging Face repositories — ``chat_template.jinja`` when the repo has one,
  otherwise ``tokenizer_config.json['chat_template']`` (a plain string, or a
  list of named templates of which ``default`` / ``tool_use`` is taken);
* local GGUF test models — the ``tokenizer.chat_template`` metadata key read
  with ``gguf.GGUFReader``;
* local MLX test models — ``chat_template.jinja`` / ``tokenizer_config.json``
  inside the model directory.

``refresh()`` re-downloads everything; it is called by the ``network``-marked
test in ``test_templates.py`` and by ``python tests/runtime/template_fixtures.py``.
Everything else in the suite only ever reads the checked-in files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "templates"


@dataclass(frozen=True)
class TemplateSpec:
    """Where a fixture comes from and what it is an example of."""

    slug: str
    family: str                 # expected detect_tool_format() name
    repo: str | None = None     # Hugging Face repo id
    local_model: str | None = None   # key in lc_testing.TEST_MODELS
    variant: str | None = None  # named template inside tokenizer_config.json
    note: str = ""


SPECS: tuple[TemplateSpec, ...] = (
    TemplateSpec("qwen3-4b-instruct-2507", "hermes", repo="Qwen/Qwen3-4B-Instruct-2507",
                 note="Hermes <tool_call> JSON, no thinking switch (Instruct-2507 is non-thinking)"),
    TemplateSpec("qwen3-coder-30b", "qwen3_coder", repo="Qwen/Qwen3-Coder-30B-A3B-Instruct",
                 note="XML <function=…><parameter=…> calls"),
    TemplateSpec("gpt-oss-20b", "harmony", repo="openai/gpt-oss-20b",
                 note="harmony channels, reasoning_effort switch"),
    TemplateSpec("devstral-small-2507", "mistral", repo="unsloth/Devstral-Small-2507",
                 note="[TOOL_CALLS] with 9-character call ids"),
    TemplateSpec("glm-4.5-air", "glm", repo="zai-org/GLM-4.5-Air",
                 note="<tool_call>name<arg_key>/<arg_value>; GLM-4.6 ships the identical template"),
    TemplateSpec("gemma-3-4b-it", "hermes", repo="unsloth/gemma-3-4b-it",
                 note="no system role, no native tools -> prompted Hermes calls"),
    TemplateSpec("llama-3.1-8b-instruct", "llama3", repo="unsloth/Llama-3.1-8B-Instruct",
                 note="JSON {'name','parameters'} and <|python_tag|>"),
    TemplateSpec("deepseek-v3.1", "deepseek", repo="deepseek-ai/DeepSeek-V3.1",
                 note="<｜tool▁calls▁begin｜> sections, 'thinking' switch"),
    TemplateSpec("minimax-m2", "minimax_m2", repo="MiniMaxAI/MiniMax-M2"),
    TemplateSpec("kimi-k2", "kimi_k2", repo="moonshotai/Kimi-K2-Instruct"),
    TemplateSpec("granite-3.3-8b", "granite", repo="ibm-granite/granite-3.3-8b-instruct"),
    # the five local test models
    TemplateSpec("local-mlx-tiny", "hermes", local_model="mlx_tiny",
                 note="Qwen3-0.6B: enable_thinking switch"),
    TemplateSpec("local-mlx-small", "hermes", local_model="mlx_small"),
    TemplateSpec("local-gguf-tiny", "hermes", local_model="gguf_tiny"),
    TemplateSpec("local-gguf-small", "hermes", local_model="gguf_small"),
    TemplateSpec("local-gguf-embed", "hermes", local_model="gguf_embed",
                 note="embedding model: usually has no chat template at all"),
)

SPEC_BY_SLUG: dict[str, TemplateSpec] = {s.slug: s for s in SPECS}


def path_of(slug: str) -> Path:
    return FIXTURE_DIR / f"{slug}.jinja"


def available() -> list[TemplateSpec]:
    """Specs whose fixture file is checked in (the embedding model has none)."""
    return [s for s in SPECS if path_of(s.slug).is_file()]


def load(slug: str) -> str:
    """The checked-in template source."""
    return path_of(slug).read_text(encoding="utf-8")


def load_all() -> dict[str, str]:
    return {s.slug: load(s.slug) for s in available()}


# --------------------------------------------------------------------------- #
# Downloading (network / local models only)
# --------------------------------------------------------------------------- #


def _from_tokenizer_config(data: dict[str, Any], variant: str | None) -> str | None:
    value = data.get("chat_template")
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        named = {t.get("name"): t.get("template") for t in value if isinstance(t, dict)}
        for key in (variant, "default", "tool_use"):
            if key and isinstance(named.get(key), str):
                return named[key]
        for template in named.values():
            if isinstance(template, str):
                return template
    return None


def _from_repo(repo: str, variant: str | None) -> str | None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError

    for filename in ("chat_template.jinja", "assets/chat_template.jinja"):
        try:
            return Path(hf_hub_download(repo, filename)).read_text(encoding="utf-8")
        except (EntryNotFoundError, HfHubHTTPError, OSError):
            continue
    try:
        path = hf_hub_download(repo, "tokenizer_config.json")
    except (EntryNotFoundError, HfHubHTTPError, OSError):
        return None
    return _from_tokenizer_config(json.loads(Path(path).read_text(encoding="utf-8")), variant)


def _from_local_model(key: str, variant: str | None) -> str | None:
    from lc_testing import TEST_MODELS

    path = TEST_MODELS[key]
    if path.suffix == ".gguf":
        if not path.is_file():
            return None
        import gguf

        field = gguf.GGUFReader(path, "r").fields.get("tokenizer.chat_template")
        if field is None:
            return None
        value = field.contents()
        return value if isinstance(value, str) else None
    if not path.is_dir():
        return None
    jinja = path / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    config = path / "tokenizer_config.json"
    if config.is_file():
        return _from_tokenizer_config(json.loads(config.read_text(encoding="utf-8")), variant)
    return None


def fetch(spec: TemplateSpec) -> str | None:
    """Download/read the template of one spec (``None`` when unavailable)."""
    if spec.repo:
        return _from_repo(spec.repo, spec.variant)
    if spec.local_model:
        return _from_local_model(spec.local_model, spec.variant)
    return None


def refresh(specs: tuple[TemplateSpec, ...] = SPECS) -> dict[str, str]:
    """Write every obtainable template to ``tests/fixtures/templates``."""
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for spec in specs:
        try:
            template = fetch(spec)
        except Exception as exc:  # noqa: BLE001 - a broken repo must not stop the rest
            print(f"  {spec.slug}: FAILED ({type(exc).__name__}: {exc})")
            continue
        if not template or not template.strip():
            print(f"  {spec.slug}: no chat template")
            continue
        text = template if template.endswith("\n") else template + "\n"
        path_of(spec.slug).write_text(text, encoding="utf-8")
        written[spec.slug] = template
        print(f"  {spec.slug}: {len(template)} chars")
    return written


if __name__ == "__main__":  # pragma: no cover - manual refresh
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    print(f"writing to {FIXTURE_DIR}")
    refresh()
