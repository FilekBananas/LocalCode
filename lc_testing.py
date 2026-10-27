"""Shared test helpers importable from any tests/<area>/ directory.

pytest inserts each test directory into ``sys.path`` (no __init__.py packages),
so a plain ``from conftest import ...`` resolves to whichever conftest.py comes
first. The helpers therefore live here, next to the root ``conftest.py`` that
puts the repository root on the path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

MODELS = Path("~/.localcode/models").expanduser()

TEST_MODELS = {
    "mlx_tiny": MODELS / "mlx-community/Qwen3-0.6B-4bit",
    "mlx_small": MODELS / "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "gguf_tiny": MODELS / "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf",
    "gguf_small": MODELS / "unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    "gguf_embed": MODELS / "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0.gguf",
}


def _ready(p: Path) -> bool:
    if p.suffix == ".gguf":
        return p.is_file() and p.stat().st_size > 1_000_000
    return (p / "config.json").is_file() and any(p.glob("*.safetensors"))


def model_path(name: str) -> Path:
    """Path of a test model, or skip the test when it is not downloaded."""
    p = TEST_MODELS[name]
    if not _ready(p):
        pytest.skip(f"test model {name} not downloaded: {p}")
    return p
