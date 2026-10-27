"""Tests for hub/search.py: variant grouping, estimates, DTO shape, error mapping.

The offline tests drive :class:`HubClient` with a fake ``HfApi`` (and a fake
``_http_get`` that serves a synthetic GGUF header), so the whole ``details()``
pipeline — remote header parsing, variant grouping, memory estimates,
recommendation — runs without a network.  The ``network`` tests then check the
same code against the real Hub.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from hub_fixtures import write_gguf
from localcode.config import Settings
from localcode.hub.search import (DEFAULT_ESTIMATE_CTX, HubClient, HubError, fallback_estimate,
                                  group_gguf_variants, hub_error_from, mlx_variant, pick_recommended,
                                  readme_excerpt)
from localcode.hub.types import ModelMeta

QWEN_REPO = "unsloth/Qwen3-0.6B-GGUF"


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeSibling:
    def __init__(self, name: str, size: int) -> None:
        self.rfilename = name
        self.size = size
        self.lfs = None


class FakeInfo:
    def __init__(self, repo: str, files: dict[str, int] | None = None, **kwargs: Any) -> None:
        self.id = repo
        self.author = repo.split("/")[0]
        self.siblings = [FakeSibling(n, s) for n, s in (files or {}).items()]
        self.downloads = kwargs.get("downloads", 1234)
        self.likes = kwargs.get("likes", 7)
        self.last_modified = kwargs.get("last_modified", datetime(2026, 5, 1, tzinfo=timezone.utc))
        self.tags = kwargs.get("tags", ["gguf", "text-generation", "base_model:quantized:Qwen/Qwen3-0.6B",
                                        "license:apache-2.0"])
        self.pipeline_tag = kwargs.get("pipeline_tag", "text-generation")
        self.gated = kwargs.get("gated", False)
        self.gguf = kwargs.get("gguf")
        self.safetensors = kwargs.get("safetensors")
        self.trending_score = kwargs.get("trending_score", 3.0)
        self.config = kwargs.get("config")
        self.sha = "deadbeef"


class FakeApi:
    def __init__(self, listings: dict[str, list[FakeInfo]] | None = None,
                 infos: dict[str, FakeInfo] | None = None) -> None:
        self.listings = listings or {}
        self.infos = infos or {}
        self.calls: list[dict[str, Any]] = []

    def list_models(self, **kwargs: Any) -> list[FakeInfo]:
        self.calls.append(kwargs)
        return list(self.listings.get(kwargs.get("filter"), []))

    def model_info(self, repo: str, **kwargs: Any) -> FakeInfo:
        self.calls.append({"model_info": repo, **kwargs})
        if repo not in self.infos:
            from huggingface_hub.errors import RepositoryNotFoundError

            raise RepositoryNotFoundError("404", response=httpx.Response(
                404, request=httpx.Request("GET", "https://huggingface.co")))
        return self.infos[repo]


@pytest.fixture
def gguf_header_bytes(tmp_path: Path) -> bytes:
    path = write_gguf(tmp_path / "Qwen3-0.6B-Q4_K_M.gguf", general={"parameter_count": 596_049_920},
                      chat_template="{%- if tools %}{{ tools }}{%- endif %}")
    return path.read_bytes()


# --------------------------------------------------------------------------- #
# Variant grouping
# --------------------------------------------------------------------------- #


def test_group_gguf_variants_by_quantisation():
    variants = group_gguf_variants([
        ("Qwen3-0.6B-Q4_K_M.gguf", 400), ("Qwen3-0.6B-Q8_0.gguf", 700), ("README.md", 10),
        ("Qwen3-0.6B-BF16-00001-of-00002.gguf", 600), ("Qwen3-0.6B-BF16-00002-of-00002.gguf", 600),
    ])
    assert [v["id"] for v in variants] == ["Qwen3-0.6B-Q4_K_M", "Qwen3-0.6B-Q8_0", "Qwen3-0.6B-BF16"]
    assert [v["quant"] for v in variants] == ["Q4_K_M", "Q8_0", "BF16"]
    split = variants[-1]
    assert split["sizeBytes"] == 1200
    assert split["files"] == ["Qwen3-0.6B-BF16-00001-of-00002.gguf", "Qwen3-0.6B-BF16-00002-of-00002.gguf"]
    assert all(v["complete"] for v in variants)


def test_group_gguf_variants_marks_missing_shards():
    variants = group_gguf_variants([("x-Q4_K_M-00001-of-00003.gguf", 5), ("x-Q4_K_M-00003-of-00003.gguf", 5)])
    assert variants[0]["complete"] is False


def test_group_gguf_variants_handles_quant_subfolders():
    variants = group_gguf_variants([("Q4_K_M/model.gguf", 100), ("BF16/model.gguf", 900)])
    by_file = {v["files"][0]: v for v in variants}
    assert by_file["Q4_K_M/model.gguf"]["quant"] == "Q4_K_M"
    assert by_file["BF16/model.gguf"]["quant"] == "BF16"
    assert len({v["id"] for v in variants}) == 2          # ids stay unique across folders


def test_projectors_are_listed_and_attached_to_model_variants():
    variants = group_gguf_variants([
        ("Qwen3-VL-8B-Q4_K_M.gguf", 4000), ("mmproj-Qwen3-VL-8B-F16.gguf", 300),
        ("mmproj-Qwen3-VL-8B-Q8_0.gguf", 150),
    ])
    model = [v for v in variants if not v["isVisionProjector"]][0]
    projectors = [v for v in variants if v["isVisionProjector"]]
    assert len(projectors) == 2
    assert model["files"] == ["Qwen3-VL-8B-Q4_K_M.gguf", "mmproj-Qwen3-VL-8B-F16.gguf"]
    assert model["sizeBytes"] == 4300 and model["modelBytes"] == 4000
    assert model["projector"] == "mmproj-Qwen3-VL-8B-F16.gguf"
    assert "Projektor wizji" in projectors[0]["label"]


def test_mlx_variant_takes_the_top_level_files():
    variant = mlx_variant("mlx-community/Qwen3-0.6B-4bit", [
        ("config.json", 900), ("model.safetensors", 335_000_000), ("tokenizer.json", 11_000_000),
        ("README.md", 800), (".gitattributes", 100), ("images/demo.png", 50_000),
    ], "4bit")
    assert "images/demo.png" not in variant["files"]
    assert ".gitattributes" not in variant["files"]
    assert variant["modelBytes"] == 335_000_000
    assert variant["sizeBytes"] == 900 + 335_000_000 + 11_000_000 + 800
    assert variant["quant"] == "4bit" and variant["complete"] is True


def test_pick_recommended_prefers_the_best_quant_that_fits():
    variants = [
        {"id": "q4", "quant": "Q4_K_M", "sizeBytes": 400, "estimate": {"fits": "yes"}, "isVisionProjector": False},
        {"id": "q8", "quant": "Q8_0", "sizeBytes": 800, "estimate": {"fits": "yes"}, "isVisionProjector": False},
        {"id": "bf16", "quant": "BF16", "sizeBytes": 1600, "estimate": {"fits": "no"}, "isVisionProjector": False},
    ]
    assert pick_recommended(variants) == "q8"
    assert [v["recommended"] for v in variants] == [False, True, False]

    variants[1]["estimate"] = {"fits": "no"}
    assert pick_recommended(variants) == "q4"

    for v in variants:
        v["estimate"] = {"fits": "no"}
    variants[0]["estimate"] = {"fits": "tight"}
    assert pick_recommended(variants) == "q4"


def test_pick_recommended_ignores_projectors_and_incomplete_variants():
    variants = [
        {"id": "proj", "quant": "F16", "sizeBytes": 100, "estimate": {"fits": "yes"}, "isVisionProjector": True},
        {"id": "part", "quant": "Q4_K_M", "sizeBytes": 200, "estimate": {"fits": "yes"},
         "isVisionProjector": False, "complete": False},
    ]
    assert pick_recommended(variants) is None


# --------------------------------------------------------------------------- #
# Estimates, README
# --------------------------------------------------------------------------- #


def test_fallback_estimate_uses_the_kv_geometry():
    meta = ModelMeta(n_layers=28, n_full_attn_layers=28, n_kv_heads=8, head_dim=128)
    est = fallback_estimate(meta, 400_000_000, n_ctx=8192, ram_limit_gb=32)
    expected_kv = 2 * 8 * 128 * 2 * 28 * 8192
    assert est.kv_cache_bytes == expected_kv
    assert est.total_bytes == est.weights_bytes + est.kv_cache_bytes + est.overhead_bytes
    assert est.fits == "yes"
    assert "Zmieści się" in est.message

    big = fallback_estimate(meta, 60_000_000_000, n_ctx=8192, ram_limit_gb=32)
    assert big.fits == "no" and "Za duży" in big.message


def test_fallback_estimate_of_a_sliding_window_model():
    meta = ModelMeta(n_layers=4, n_full_attn_layers=1, n_kv_heads=4, head_dim=128, sliding_window=512)
    est = fallback_estimate(meta, 1000, n_ctx=4096, ram_limit_gb=32)
    per_layer_token = 2 * 4 * 128 * 2
    assert est.kv_cache_bytes == per_layer_token * 4096 + per_layer_token * 3 * 512


def test_readme_excerpt_strips_front_matter_and_truncates():
    text = "---\nlicense: apache-2.0\n---\n\n# Qwen3\n\n" + "para\n\n" * 1500
    out = readme_excerpt(text)
    assert out is not None
    assert not out.startswith("---") and out.startswith("# Qwen3")
    assert len(out) <= 4100 and out.endswith("…")
    assert readme_excerpt(None) is None


# --------------------------------------------------------------------------- #
# Error mapping
# --------------------------------------------------------------------------- #


def _http_error(status: int):
    from huggingface_hub.errors import HfHubHTTPError

    request = httpx.Request("GET", "https://huggingface.co/x")
    return HfHubHTTPError(f"HTTP {status}", response=httpx.Response(status, request=request))


@pytest.mark.parametrize("status,code", [(401, "unauthorized"), (403, "gated"), (429, "rate_limited"),
                                         (500, "unavailable"), (418, "http_error")])
def test_http_errors_become_polish_hub_errors(status: int, code: str):
    err = hub_error_from(_http_error(status), "org/repo")
    assert err.code == code
    assert err.message and err.message[0].isupper()
    assert not err.message.isascii() or code == "http_error"     # Polish diacritics in the message


def test_network_errors_become_polish_hub_errors():
    assert hub_error_from(httpx.ConnectError("boom"), "org/repo").code == "network"
    assert hub_error_from(httpx.ReadTimeout("slow"), "org/repo").code == "timeout"
    assert "internetem" in hub_error_from(httpx.ConnectError("boom"), "org/repo").message


def test_gated_and_missing_repos():
    from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

    request = httpx.Request("GET", "https://huggingface.co/x")
    gated = hub_error_from(GatedRepoError("g", response=httpx.Response(403, request=request)), "meta/llama")
    assert gated.code == "gated" and "meta/llama" in gated.message and gated.status == 403
    missing = hub_error_from(RepositoryNotFoundError("n", response=httpx.Response(404, request=request)), "a/b")
    assert missing.code == "not_found" and missing.status == 404


# --------------------------------------------------------------------------- #
# HubClient with a fake API
# --------------------------------------------------------------------------- #


def test_search_builds_hub_model_dtos(settings: Settings):
    api = FakeApi({"gguf": [FakeInfo(QWEN_REPO, gguf={"architecture": "qwen3", "context_length": 40960,
                                                      "total": 596_049_920})]})
    client = HubClient(settings, api=api)
    (model,) = client.search("qwen3", format="gguf")
    assert model["id"] == QWEN_REPO and model["author"] == "unsloth" and model["name"] == "Qwen3-0.6B-GGUF"
    assert model["format"] == "gguf"
    assert model["downloads"] == 1234 and model["likes"] == 7
    assert model["updatedAt"] == int(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp() * 1000)
    assert model["paramsB"] == 0.596     # the real tensor count wins over the "0.6B" label
    assert model["arch"] == "qwen3" and model["contextLength"] == 40960
    assert model["curated"] is False and model["gated"] is False
    assert "license:apache-2.0" not in model["tags"]
    assert "Kwantyzacja modelu Qwen/Qwen3-0.6B" == model["description"]
    assert api.calls[0]["filter"] == "gguf" and api.calls[0]["sort"] == "trending_score"


def test_search_results_carry_the_trust_score(settings: Settings):
    """The advanced browser lists arbitrary repositories, so it needs the badge
    at least as much as the store does."""
    api = FakeApi({"gguf": [FakeInfo(QWEN_REPO, tags=["gguf", "base_model:Qwen/Qwen3-0.6B"])]})
    (model,) = HubClient(settings, api=api).search("qwen3", format="gguf")
    assert 0 <= model["trust"]["score"] <= 100
    assert model["trust"]["label"]
    assert model["publisher"]["kind"] == "trusted"        # unsloth conversion of a Qwen model
    assert model["publisher"]["publisher"] == "Qwen (Alibaba)"
    assert model["publisher"]["official"] is False


def test_search_drops_repos_that_cannot_be_served(settings: Settings):
    api = FakeApi({"gguf": [FakeInfo("org/whisper-GGUF", pipeline_tag="automatic-speech-recognition"),
                            FakeInfo("org/sd-GGUF", pipeline_tag="text-to-image"),
                            FakeInfo(QWEN_REPO)]})
    ids = [m["id"] for m in HubClient(settings, api=api).search("", format="gguf")]
    assert ids == [QWEN_REPO]


def test_search_merges_both_formats_and_marks_curated(settings: Settings):
    api = FakeApi({
        "gguf": [FakeInfo("nomic-ai/nomic-embed-text-v1.5-GGUF", downloads=10, likes=1, trending_score=1.0)],
        "mlx": [FakeInfo("mlx-community/Qwen3-0.6B-4bit", downloads=99, likes=5, trending_score=9.0)],
    })
    models = HubClient(settings, api=api).search("qwen", format="any", sort="downloads")
    assert [m["id"] for m in models] == ["mlx-community/Qwen3-0.6B-4bit", "nomic-ai/nomic-embed-text-v1.5-GGUF"]
    assert [m["format"] for m in models] == ["mlx", "gguf"]
    assert models[1]["curated"] is True          # it is in hub/curated.py
    assert models[1]["description"]


def test_search_validates_its_arguments(settings: Settings):
    client = HubClient(settings, api=FakeApi())
    for kwargs in ({"format": "onnx"}, {"sort": "stars"}, {"limit": "many"}):
        with pytest.raises(HubError) as exc:
            client.search("x", **kwargs)
        assert exc.value.code == "invalid" and exc.value.status == 400


def test_details_of_a_gguf_repo(settings: Settings, gguf_header_bytes: bytes,
                                monkeypatch: pytest.MonkeyPatch):
    files = {"README.md": 500, "Qwen3-0.6B-Q4_K_M.gguf": 400_000_000, "Qwen3-0.6B-Q8_0.gguf": 700_000_000,
             "Qwen3-0.6B-BF16.gguf": 1_500_000_000}
    api = FakeApi(infos={QWEN_REPO: FakeInfo(QWEN_REPO, files, gguf={
        "architecture": "qwen3", "context_length": 40960,
        # the Hub returns the chat template in the `gguf` expand field
        "chat_template": "{%- if tools %}{{ tools }}{%- endif %}"})})

    def fake_get(self, repo: str, filename: str, *, byte_range=None):
        if filename == "README.md":
            return b"---\nlicense: apache-2.0\n---\n\n# Qwen3 0.6B\n\nOpis modelu."
        return gguf_header_bytes

    monkeypatch.setattr(HubClient, "_http_get", fake_get)
    detail = HubClient(settings, api=api).details(QWEN_REPO)

    assert detail["arch"] == "qwen3"
    assert detail["contextLength"] == 40960
    assert detail["isEmbedding"] is False
    assert "tools" in detail["capabilities"]
    assert detail["readme"].startswith("# Qwen3 0.6B")
    assert [v["quant"] for v in detail["variants"]] == ["Q4_K_M", "Q8_0", "BF16"]
    assert all(v["estimate"]["nCtx"] == min(40960, DEFAULT_ESTIMATE_CTX) for v in detail["variants"])
    assert all(v["estimate"]["totalBytes"] > v["sizeBytes"] for v in detail["variants"])
    assert sum(v["recommended"] for v in detail["variants"]) == 1
    assert {f["name"] for f in detail["files"]} == set(files)
    assert detail["accessError"] is None
    assert json.dumps(detail)                      # the DTO is JSON-serialisable


def test_details_of_an_mlx_repo(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    config = {"model_type": "qwen3", "hidden_size": 1024, "num_hidden_layers": 28,
              "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
              "max_position_embeddings": 40960, "quantization": {"bits": 4, "group_size": 64}}
    files = {"config.json": 900, "model.safetensors": 335_000_000, "README.md": 100}
    api = FakeApi(infos={"mlx-community/Qwen3-0.6B-4bit": FakeInfo("mlx-community/Qwen3-0.6B-4bit", files)})
    monkeypatch.setattr(HubClient, "_http_get",
                        lambda self, repo, filename, byte_range=None: json.dumps(config).encode()
                        if filename == "config.json" else b"# readme")
    detail = HubClient(settings, api=api).details("mlx-community/Qwen3-0.6B-4bit")
    assert detail["arch"] == "qwen3" and detail["contextLength"] == 40960
    assert len(detail["variants"]) == 1
    variant = detail["variants"][0]
    assert variant["quant"] == "4bit" and variant["recommended"] is True
    assert variant["estimate"]["totalBytes"] > 335_000_000


def test_details_rejects_repos_without_a_supported_model(settings: Settings):
    api = FakeApi(infos={"org/pytorch": FakeInfo("org/pytorch", {"pytorch_model.bin": 10, "README.md": 1})})
    with pytest.raises(HubError) as exc:
        HubClient(settings, api=api).details("org/pytorch")
    assert exc.value.code == "unsupported"
    assert "GGUF" in exc.value.message


def test_details_validates_the_repo_id_and_maps_404(settings: Settings):
    client = HubClient(settings, api=FakeApi())
    with pytest.raises(HubError) as bad:
        client.details("not-a-repo")
    assert bad.value.code == "invalid"
    with pytest.raises(HubError) as missing:
        client.details("org/missing")
    assert missing.value.code == "not_found"


def test_details_are_cached(settings: Settings, gguf_header_bytes: bytes, monkeypatch: pytest.MonkeyPatch):
    api = FakeApi(infos={QWEN_REPO: FakeInfo(QWEN_REPO, {"Qwen3-0.6B-Q4_K_M.gguf": 400_000_000})})
    monkeypatch.setattr(HubClient, "_http_get", lambda *a, **k: gguf_header_bytes)
    client = HubClient(settings, api=api)
    first = client.details(QWEN_REPO)
    second = client.details(QWEN_REPO)
    assert first == second
    assert first is not second                     # a copy, so callers cannot poison the cache
    assert len([c for c in api.calls if "model_info" in c]) == 1
    client.details(QWEN_REPO, refresh=True)
    assert len([c for c in api.calls if "model_info" in c]) == 2


def test_variant_lookup(settings: Settings, gguf_header_bytes: bytes, monkeypatch: pytest.MonkeyPatch):
    api = FakeApi(infos={QWEN_REPO: FakeInfo(QWEN_REPO, {"Qwen3-0.6B-Q4_K_M.gguf": 400_000_000})})
    monkeypatch.setattr(HubClient, "_http_get", lambda *a, **k: gguf_header_bytes)
    client = HubClient(settings, api=api)
    variant = client.variant(QWEN_REPO, "Qwen3-0.6B-Q4_K_M")
    assert variant["files"] == ["Qwen3-0.6B-Q4_K_M.gguf"]
    with pytest.raises(HubError) as exc:
        client.variant(QWEN_REPO, "nope")
    assert exc.value.code == "variant_not_found"


def test_curated_list_is_offline(settings: Settings):
    models = HubClient(settings, api=FakeApi()).curated()
    assert len(models) > 10
    assert all(m["curated"] is True for m in models)
    assert all(m["format"] in ("gguf", "mlx") for m in models)


# --------------------------------------------------------------------------- #
# Real Hugging Face (network)
# --------------------------------------------------------------------------- #


@pytest.mark.network
@pytest.mark.parametrize("fmt", ["gguf", "mlx"])
def test_real_search(settings: Settings, fmt: str):
    models = HubClient(settings).search("qwen3", format=fmt, limit=10)
    assert 0 < len(models) <= 10
    assert all(m["format"] == fmt for m in models)
    assert all(isinstance(m["id"], str) and "/" in m["id"] for m in models)
    assert all(m["downloads"] is not None for m in models)
    assert any("qwen" in m["id"].lower() for m in models)
    assert all(m["pipelineTag"] not in ("text-to-image", "automatic-speech-recognition") for m in models)


@pytest.mark.network
def test_real_search_any_format_merges_both(settings: Settings):
    models = HubClient(settings).search("qwen3", format="any", sort="downloads", limit=20)
    assert {m["format"] for m in models} == {"gguf", "mlx"}
    downloads = [m["downloads"] or 0 for m in models]
    assert downloads == sorted(downloads, reverse=True)


@pytest.mark.network
def test_real_details_of_qwen3_06b_gguf(settings: Settings):
    detail = HubClient(settings).details(QWEN_REPO)
    assert detail["arch"] == "qwen3"
    assert detail["contextLength"] == 40960
    assert detail["isEmbedding"] is False
    assert "tools" in detail["capabilities"]
    assert detail["readme"]

    variants = {v["id"]: v for v in detail["variants"]}
    assert "Qwen3-0.6B-Q4_K_M" in variants
    q4 = variants["Qwen3-0.6B-Q4_K_M"]
    assert q4["files"] == ["Qwen3-0.6B-Q4_K_M.gguf"]
    assert 300_000_000 < q4["sizeBytes"] < 600_000_000
    assert q4["estimate"]["weightsBytes"] == q4["sizeBytes"]
    assert q4["estimate"]["totalBytes"] > q4["sizeBytes"]
    assert q4["estimate"]["fits"] == "yes"
    assert q4["estimate"]["message"]

    recommended = [v for v in detail["variants"] if v["recommended"]]
    assert len(recommended) == 1
    assert recommended[0]["quant"] in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "BF16", "F16")
    assert all(v["sizeBytes"] > 0 for v in detail["variants"])


@pytest.mark.network
def test_real_details_of_an_mlx_repo(settings: Settings):
    detail = HubClient(settings).details("mlx-community/Qwen3-0.6B-4bit")
    assert detail["arch"] == "qwen3"
    assert detail["contextLength"] == 40960
    assert len(detail["variants"]) == 1
    variant = detail["variants"][0]
    assert variant["quant"] == "4bit"
    assert variant["recommended"] is True
    assert "config.json" in variant["files"] and "model.safetensors" in variant["files"]
