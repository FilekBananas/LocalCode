"""Tests for hub/library.py: discovery, grouping, keys, caching, defaults, deletion.

Every fixture is a synthetic directory tree with real GGUF/safetensors headers,
so the library really parses what it finds.  ``LOCALCODE_HOME`` is redirected to
a temporary directory (``hub_home`` fixture), so the metadata cache and the
per-model load defaults never touch the developer's own ``~/.localcode``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from hub_fixtures import make_hf_cache_repo, make_mlx_dir, write_gguf, write_mmproj, write_split_gguf
from localcode.config import Settings
from localcode.hub import library as lib
from localcode.hub.library import LibraryError, ModelLibrary, display_name_for, model_key


def keys(models) -> list[str]:
    return sorted(m.key for m in models)


def by_key(models) -> dict[str, object]:
    return {m.key: m for m in models}


# --------------------------------------------------------------------------- #
# Naming helpers
# --------------------------------------------------------------------------- #


def test_model_key_of_a_downloaded_file():
    assert model_key("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf") == \
        "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"
    assert model_key("unsloth/X-GGUF", "Q4/x-Q4_K_M-00001-of-00002.gguf") == "unsloth/X-GGUF/x-Q4_K_M"
    assert model_key("mlx-community/Qwen3-0.6B-4bit") == "mlx-community/Qwen3-0.6B-4bit"


@pytest.mark.parametrize("name,quant,expected", [
    ("Qwen3-4B-Instruct-2507-Q4_K_M", "Q4_K_M", "Qwen3 4B Instruct 2507 · Q4_K_M"),
    ("Qwen3-4B-UD-Q4_K_XL", "UD-Q4_K_XL", "Qwen3 4B · UD-Q4_K_XL"),
    ("Qwen3-0.6B-4bit", "4bit", "Qwen3 0.6B · 4bit"),
    ("Qwen3-0.6B-MLX", None, "Qwen3 0.6B"),
])
def test_display_name(name: str, quant: str | None, expected: str):
    assert display_name_for(name, quant) == expected


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #


def test_scan_finds_gguf_and_mlx_models(settings: Settings, models_dir: Path):
    write_gguf(models_dir / "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf", pad_to=4096)
    make_mlx_dir(models_dir / "mlx-community/Qwen3-0.6B-4bit",
                 chat_template="{%- if tools %}{{ tools }}{%- endif %}")
    models = by_key(ModelLibrary(settings).scan())

    assert sorted(models) == ["mlx-community/Qwen3-0.6B-4bit", "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"]
    gguf = models["unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"]
    assert gguf.format == "gguf"
    assert gguf.repo_id == "unsloth/Qwen3-0.6B-GGUF"
    assert gguf.quant == "Q4_K_M"
    assert gguf.source == "localcode"
    assert gguf.display_name == "Qwen3 0.6B · Q4_K_M"
    assert gguf.size_bytes == 4096
    assert gguf.path.endswith("Qwen3-0.6B-Q4_K_M.gguf")
    assert gguf.meta.arch == "qwen3" and gguf.meta.context_length == 40960
    mlx = models["mlx-community/Qwen3-0.6B-4bit"]
    assert mlx.format == "mlx" and mlx.quant == "4bit"
    assert mlx.path == str(models_dir / "mlx-community/Qwen3-0.6B-4bit")
    assert mlx.capabilities == ["tools"]


def test_split_shards_are_one_model(settings: Settings, models_dir: Path):
    repo = models_dir / "unsloth/Big-GGUF"
    write_split_gguf(repo, "Big-Q4_K_M", 3, pad_to=2048)
    models = ModelLibrary(settings).scan()
    assert keys(models) == ["unsloth/Big-GGUF/Big-Q4_K_M"]
    model = models[0]
    assert model.path.endswith("Big-Q4_K_M-00001-of-00003.gguf")
    assert model.size_bytes == 3 * 2048


def test_incomplete_split_models_are_skipped(settings: Settings, models_dir: Path):
    repo = models_dir / "unsloth/Big-GGUF"
    write_split_gguf(repo, "Big-Q4_K_M", 3)
    (repo / "Big-Q4_K_M-00002-of-00003.gguf").unlink()
    assert ModelLibrary(settings).scan() == []


def test_mmproj_is_attached_not_listed(settings: Settings, models_dir: Path):
    repo = models_dir / "Qwen/Qwen3-VL-8B-Instruct-GGUF"
    write_gguf(repo / "Qwen3-VL-8B-Instruct-Q4_K_M.gguf")
    write_mmproj(repo / "mmproj-Qwen3-VL-8B-Instruct-F16.gguf")
    write_mmproj(repo / "mmproj-Qwen3-VL-8B-Instruct-Q8_0.gguf", file_type=7)
    models = ModelLibrary(settings).scan()
    assert keys(models) == ["Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-Q4_K_M"]
    model = models[0]
    assert model.mmproj_path and model.mmproj_path.endswith("mmproj-Qwen3-VL-8B-Instruct-F16.gguf")
    assert "vision" in model.capabilities
    assert model.to_dict()["hasVision"] is True


def test_quant_subfolders_become_separate_models(settings: Settings, models_dir: Path):
    repo = models_dir / "unsloth/Qwen3-4B-GGUF"
    write_gguf(repo / "Q4_K_M/Qwen3-4B-Q4_K_M.gguf")
    write_split_gguf(repo / "BF16", "Qwen3-4B-BF16", 2)
    write_mmproj(repo / "mmproj-F16.gguf")
    models = by_key(ModelLibrary(settings).scan())
    assert sorted(models) == ["unsloth/Qwen3-4B-GGUF/Qwen3-4B-BF16", "unsloth/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M"]
    # the projector lives one level up and is attached to both variants
    assert all(m.mmproj_path for m in models.values())


def test_directories_that_are_not_mlx_are_ignored(settings: Settings, models_dir: Path):
    make_mlx_dir(models_dir / "org/pytorch-model", config={"model_type": "qwen3", "hidden_size": 8,
                                                           "num_hidden_layers": 1})
    assert ModelLibrary(settings).scan() == []


def test_mlx_is_recognised_by_the_readme_front_matter(settings: Settings, models_dir: Path):
    make_mlx_dir(models_dir / "org/Weird-Name", config={"model_type": "qwen3", "hidden_size": 8,
                                                        "num_hidden_layers": 1},
                 readme="---\nlibrary_name: mlx\n---\n# model\n")
    assert keys(ModelLibrary(settings).scan()) == ["org/Weird-Name"]


def test_incomplete_mlx_shards_are_skipped(settings: Settings, models_dir: Path):
    d = make_mlx_dir(models_dir / "org/Sharded-4bit",
                     weights={"model-00001-of-00002.safetensors": None},
                     index={"a.weight": "model-00001-of-00002.safetensors",
                            "b.weight": "model-00002-of-00002.safetensors"})
    assert (d / "model.safetensors.index.json").is_file()
    assert ModelLibrary(settings).scan() == []


def test_embedding_models_are_flagged(settings: Settings, models_dir: Path):
    from hub_fixtures import write_embedding_gguf

    write_embedding_gguf(models_dir / "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q8_0.gguf")
    model = ModelLibrary(settings).scan()[0]
    assert model.is_embedding and model.capabilities == ["embedding"]


def test_models_outside_the_org_layout_get_a_local_key(settings: Settings, models_dir: Path):
    write_gguf(models_dir / "loose-Q4_K_M.gguf")
    write_gguf(models_dir / "somedir/other-Q8_0.gguf")
    assert keys(ModelLibrary(settings).scan()) == ["local/loose-Q4_K_M", "local/somedir/other-Q8_0"]


# --------------------------------------------------------------------------- #
# Other sources
# --------------------------------------------------------------------------- #


def test_lmstudio_models_are_listed_with_their_source(settings: Settings, models_dir: Path, tmp_path: Path):
    lmstudio = tmp_path / "lmstudio-models"
    write_gguf(lmstudio / "lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
    settings = dataclasses.replace(settings, scan_lmstudio=True)
    models = ModelLibrary(settings, lmstudio_dirs=[lmstudio]).scan()
    assert keys(models) == ["lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M"]
    assert models[0].source == "lmstudio"


def test_hf_cache_models_are_listed_with_their_source(settings: Settings, tmp_path: Path):
    staging = tmp_path / "staging"
    write_gguf(staging / "Qwen3-0.6B-Q4_K_M.gguf")
    cache = tmp_path / "hf-cache"
    make_hf_cache_repo(cache, "unsloth/Qwen3-0.6B-GGUF", {"Qwen3-0.6B-Q4_K_M.gguf": staging / "Qwen3-0.6B-Q4_K_M.gguf"})
    settings = dataclasses.replace(settings, scan_hf_cache=True)
    models = ModelLibrary(settings, hf_cache_dir=cache).scan()
    assert keys(models) == ["unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"]
    assert models[0].source == "hf-cache"
    assert models[0].repo_id == "unsloth/Qwen3-0.6B-GGUF"


def test_localcode_wins_when_two_sources_hold_the_same_model(settings: Settings, models_dir: Path,
                                                             tmp_path: Path):
    write_gguf(models_dir / "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf", pad_to=8192)
    lmstudio = tmp_path / "lmstudio-models"
    write_gguf(lmstudio / "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf", pad_to=4096)
    settings = dataclasses.replace(settings, scan_lmstudio=True)
    models = ModelLibrary(settings, lmstudio_dirs=[lmstudio]).scan()
    assert len(models) == 1
    assert models[0].source == "localcode" and models[0].size_bytes == 8192


# --------------------------------------------------------------------------- #
# Cache, queries
# --------------------------------------------------------------------------- #


def test_metadata_is_cached_between_scans(settings: Settings, models_dir: Path, hub_home: Path,
                                          monkeypatch: pytest.MonkeyPatch):
    path = write_gguf(models_dir / "org/repo/model-Q4_K_M.gguf")
    library = ModelLibrary(settings)
    library.scan()
    assert (hub_home / "data" / "library-cache.json").is_file()

    calls: list[str] = []
    real = lib.read_gguf_meta
    monkeypatch.setattr(lib, "read_gguf_meta", lambda p: (calls.append(str(p)), real(p))[1])
    ModelLibrary(settings).scan()
    assert calls == []                      # unchanged file: served from the cache

    path.write_bytes(path.read_bytes() + b"\0" * 16)   # touch it → re-read
    ModelLibrary(settings).scan()
    assert calls == [str(path)]


def test_unreadable_models_are_reported_not_fatal(settings: Settings, models_dir: Path):
    write_gguf(models_dir / "org/repo/good-Q4_K_M.gguf")
    (models_dir / "org/repo/broken-Q4_K_M.gguf").write_bytes(b"NOPE" + b"\0" * 128)
    library = ModelLibrary(settings)
    models = library.scan()
    assert keys(models) == ["org/repo/good-Q4_K_M"]
    assert library.scan_errors and "broken" in library.scan_errors[0]["path"]


def test_get_and_resolve(settings: Settings, models_dir: Path):
    write_gguf(models_dir / "unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M.gguf")
    make_mlx_dir(models_dir / "mlx-community/Qwen3-0.6B-4bit")
    library = ModelLibrary(settings)
    library.scan()
    key = "unsloth/Qwen3-4B-Instruct-2507-GGUF/Qwen3-4B-Instruct-2507-Q4_K_M"
    assert library.get(key).key == key
    assert library.get("nope") is None
    assert library.resolve(key).key == key
    assert library.resolve("mlx-community/Qwen3-0.6B-4bit").key == "mlx-community/Qwen3-0.6B-4bit"
    assert library.resolve("qwen3 4b").key == key
    assert library.resolve("") is None
    assert library.resolve("llama") is None


def test_list_is_scanned_lazily_and_sorted(settings: Settings, models_dir: Path):
    write_gguf(models_dir / "org/b-repo/zeta-Q4_K_M.gguf")
    write_gguf(models_dir / "org/a-repo/alpha-Q4_K_M.gguf")
    library = ModelLibrary(settings)
    assert [m.display_name for m in library.list()] == ["alpha · Q4_K_M", "zeta · Q4_K_M"]


# --------------------------------------------------------------------------- #
# Load defaults
# --------------------------------------------------------------------------- #


def test_load_defaults_round_trip(settings: Settings, models_dir: Path, hub_home: Path):
    write_gguf(models_dir / "org/repo/model-Q4_K_M.gguf")
    library = ModelLibrary(settings)
    library.scan()
    key = "org/repo/model-Q4_K_M"
    assert library.load_defaults(key) is None
    stored = library.set_load_defaults(key, {"nCtx": 8192, "nParallel": 2, "kvCacheType": "q8_0"})
    assert stored == {"nCtx": 8192, "nParallel": 2, "kvCacheType": "q8_0"}

    path = hub_home / "data" / "model-defaults.json"
    assert json.loads(path.read_text()) == {key: stored}
    assert ModelLibrary(settings).load_defaults(key) == stored     # survives a restart

    assert library.set_load_defaults(key, None) is None
    assert library.load_defaults(key) is None
    assert json.loads(path.read_text()) == {}


@pytest.mark.parametrize("bad", [
    {"nCtx": 10},                  # below the minimum
    {"nCtx": "8192"},              # not a number
    {"flashAttn": "yes"},
    {"kvCacheType": "q3_0"},
    {"embedding": "true"},
    {"unknownKey": 1},
])
def test_load_defaults_are_validated(settings: Settings, models_dir: Path, bad: dict):
    write_gguf(models_dir / "org/repo/model-Q4_K_M.gguf")
    library = ModelLibrary(settings)
    library.scan()
    with pytest.raises(LibraryError) as exc:
        library.set_load_defaults("org/repo/model-Q4_K_M", bad)
    assert exc.value.code == "invalid"


def test_load_defaults_for_an_unknown_model(settings: Settings):
    with pytest.raises(LibraryError) as exc:
        ModelLibrary(settings).set_load_defaults("nope", {"nCtx": 4096})
    assert exc.value.code == "not_found"
    assert "Nie znaleziono modelu" in exc.value.message


# --------------------------------------------------------------------------- #
# Deletion
# --------------------------------------------------------------------------- #


def test_delete_removes_the_files_and_prunes_the_directory(settings: Settings, models_dir: Path):
    repo = models_dir / "unsloth/Qwen3-0.6B-GGUF"
    write_split_gguf(repo, "Qwen3-0.6B-Q4_K_M", 2)
    write_gguf(repo / "Qwen3-0.6B-Q8_0.gguf")
    (repo / "README.md").write_text("hello")
    library = ModelLibrary(settings)
    library.scan()

    library.delete("unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M")
    assert not list(repo.glob("*Q4_K_M*"))
    assert (repo / "Qwen3-0.6B-Q8_0.gguf").is_file()
    assert library.get("unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M") is None

    library.delete("unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0")
    assert not repo.exists()                       # bookkeeping files go with the last model
    assert not (models_dir / "unsloth").exists()


def test_delete_also_removes_orphaned_projectors_and_defaults(settings: Settings, models_dir: Path,
                                                              hub_home: Path):
    repo = models_dir / "Qwen/Qwen3-VL-8B-GGUF"
    write_gguf(repo / "Qwen3-VL-8B-Q4_K_M.gguf")
    write_mmproj(repo / "mmproj-F16.gguf")
    library = ModelLibrary(settings)
    library.scan()
    key = "Qwen/Qwen3-VL-8B-GGUF/Qwen3-VL-8B-Q4_K_M"
    library.set_load_defaults(key, {"nCtx": 4096})
    library.delete(key)
    assert not repo.exists()
    assert json.loads((hub_home / "data" / "model-defaults.json").read_text()) == {}


def test_delete_refuses_models_from_other_apps(settings: Settings, tmp_path: Path):
    lmstudio = tmp_path / "lmstudio-models"
    write_gguf(lmstudio / "lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf")
    cache = tmp_path / "hf-cache"
    staging = tmp_path / "staging"
    write_gguf(staging / "Qwen3-0.6B-Q4_K_M.gguf")
    make_hf_cache_repo(cache, "unsloth/Qwen3-0.6B-GGUF",
                       {"Qwen3-0.6B-Q4_K_M.gguf": staging / "Qwen3-0.6B-Q4_K_M.gguf"})
    settings = dataclasses.replace(settings, scan_lmstudio=True, scan_hf_cache=True)
    library = ModelLibrary(settings, lmstudio_dirs=[lmstudio], hf_cache_dir=cache)
    library.scan()

    with pytest.raises(LibraryError) as lm_exc:
        library.delete("lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M")
    assert lm_exc.value.code == "not_owned"
    assert "LM Studio" in lm_exc.value.message
    assert (lmstudio / "lmstudio-community/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf").is_file()

    with pytest.raises(LibraryError) as hf_exc:
        library.delete("unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M")
    assert hf_exc.value.code == "not_owned"
    assert "Hugging Face" in hf_exc.value.message


def test_delete_of_an_mlx_model(settings: Settings, models_dir: Path):
    make_mlx_dir(models_dir / "mlx-community/Qwen3-0.6B-4bit")
    library = ModelLibrary(settings)
    library.scan()
    library.delete("mlx-community/Qwen3-0.6B-4bit")
    assert not (models_dir / "mlx-community").exists()
    with pytest.raises(LibraryError) as exc:
        library.delete("mlx-community/Qwen3-0.6B-4bit")
    assert exc.value.code == "not_found"


def test_delete_keeps_a_directory_with_downloads_in_progress(settings: Settings, models_dir: Path):
    repo = models_dir / "org/repo"
    write_gguf(repo / "model-Q4_K_M.gguf")
    part = repo / ".cache" / "localcode" / "other-Q8_0.gguf.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(b"\0" * 32)
    library = ModelLibrary(settings)
    library.scan()
    library.delete("org/repo/model-Q4_K_M")
    assert part.is_file(), "a running download must not be deleted with the model"


# --------------------------------------------------------------------------- #
# Importing a model the person already has on disk
# --------------------------------------------------------------------------- #


def test_import_copies_a_gguf_file_into_the_library(tmp_path: Path) -> None:
    elsewhere = tmp_path / "pobrane"
    elsewhere.mkdir()
    write_gguf(elsewhere / "MyModel-Q4_K_M.gguf", pad_to=4096)
    library = ModelLibrary(Settings(models_dir=str(tmp_path / "models"),
                                    scan_lmstudio=False, scan_hf_cache=False))

    model = library.import_path(elsewhere / "MyModel-Q4_K_M.gguf")

    assert model.format == "gguf"
    assert model.source == "localcode"
    assert Path(model.path).exists()
    assert (elsewhere / "MyModel-Q4_K_M.gguf").exists()      # a copy, not a move
    assert model.key in {m.key for m in library.list()}


def test_import_can_move_instead_of_copying(tmp_path: Path) -> None:
    """A 40 GB model should not have to exist twice."""
    elsewhere = tmp_path / "pobrane"
    elsewhere.mkdir()
    source = elsewhere / "MyModel-Q4_K_M.gguf"
    write_gguf(source, pad_to=4096)
    library = ModelLibrary(Settings(models_dir=str(tmp_path / "models"),
                                    scan_lmstudio=False, scan_hf_cache=False))

    library.import_path(source, move=True)

    assert not source.exists()


def test_import_takes_an_mlx_directory(tmp_path: Path) -> None:
    source = make_mlx_dir(tmp_path / "pobrane" / "MyMLX")
    library = ModelLibrary(Settings(models_dir=str(tmp_path / "models"),
                                    scan_lmstudio=False, scan_hf_cache=False))

    model = library.import_path(source)

    assert model.format == "mlx"
    assert (Path(model.path) / "config.json").exists()


def test_import_refuses_what_is_not_a_model(tmp_path: Path) -> None:
    junk = tmp_path / "notatki.txt"
    junk.write_text("to nie model", encoding="utf-8")
    library = ModelLibrary(Settings(models_dir=str(tmp_path / "models"),
                                    scan_lmstudio=False, scan_hf_cache=False))

    with pytest.raises(LibraryError) as excinfo:
        library.import_path(junk)
    assert "gguf" in str(excinfo.value).lower()

    empty = tmp_path / "pusty"
    empty.mkdir()
    with pytest.raises(LibraryError):
        library.import_path(empty)


def test_import_refuses_to_overwrite_an_existing_name(tmp_path: Path) -> None:
    elsewhere = tmp_path / "pobrane"
    elsewhere.mkdir()
    write_gguf(elsewhere / "MyModel-Q4_K_M.gguf", pad_to=4096)
    library = ModelLibrary(Settings(models_dir=str(tmp_path / "models"),
                                    scan_lmstudio=False, scan_hf_cache=False))
    library.import_path(elsewhere / "MyModel-Q4_K_M.gguf")

    with pytest.raises(LibraryError) as excinfo:
        library.import_path(elsewhere / "MyModel-Q4_K_M.gguf")
    assert "już" in str(excinfo.value)


def test_an_imported_model_can_be_deleted_again(tmp_path: Path) -> None:
    """It lives under models_dir, so the uninstall button owns it."""
    elsewhere = tmp_path / "pobrane"
    elsewhere.mkdir()
    write_gguf(elsewhere / "MyModel-Q4_K_M.gguf", pad_to=4096)
    library = ModelLibrary(Settings(models_dir=str(tmp_path / "models"),
                                    scan_lmstudio=False, scan_hf_cache=False))
    model = library.import_path(elsewhere / "MyModel-Q4_K_M.gguf")

    library.delete(model.key)

    assert model.key not in {m.key for m in library.list()}


def test_effort_is_stored_per_model(tmp_path) -> None:
    """A reasoning level belongs to the model, so it is saved next to it."""
    from localcode.hub.library import EFFORT_VALUES, ModelLibrary

    assert "effort" in ModelLibrary._validate_defaults({"effort": "high"})
    for level in EFFORT_VALUES:
        assert ModelLibrary._validate_defaults({"effort": level})["effort"] == level


def test_a_nonsense_effort_is_refused_in_polish() -> None:
    from localcode.hub.library import LibraryError, ModelLibrary

    with pytest.raises(LibraryError) as exc:
        ModelLibrary._validate_defaults({"effort": "bardzo-mocno"})
    assert "effort" in str(exc.value) and "high" in str(exc.value)
