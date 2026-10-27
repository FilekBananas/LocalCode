"""``localcode doctor`` — the checklist, its verdicts and the exit code."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from localcode import doctor
from localcode.config import Settings


def test_check_python_is_ok() -> None:
    check = doctor.check_python()
    assert check.status == "ok"
    assert check.detail.startswith(".".join(str(v) for v in __import__("sys").version_info[:2]))


def test_check_python_too_old(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "MIN_PYTHON", (99, 0))
    check = doctor.check_python()
    assert check.status == "fail"
    assert "brew install python" in check.fix


def test_check_platform_on_mac(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(doctor.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(doctor.platform, "mac_ver", lambda: ("26.0", ("", "", ""), ""))
    check = doctor.check_platform()
    assert check.status == "ok"
    assert "Apple Silicon" in check.detail


def test_check_platform_on_intel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(doctor.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(doctor.platform, "mac_ver", lambda: ("26.0", ("", "", ""), ""))
    check = doctor.check_platform()
    assert check.status == "warn"
    assert "MLX" in check.fix


def test_check_platform_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor.platform, "system", lambda: "Linux")
    check = doctor.check_platform()
    assert check.status == "fail" and check.essential


def test_models_dir_is_created(tmp_path: Path) -> None:
    settings = Settings(models_dir=str(tmp_path / "modele"))
    check = doctor.check_models_dir(settings)
    assert check.status == "ok"
    assert (tmp_path / "modele").is_dir()


def test_models_dir_without_write_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(models_dir=str(tmp_path / "ro"))
    (tmp_path / "ro").mkdir()
    monkeypatch.setattr(doctor.os, "access", lambda *a, **k: False)
    check = doctor.check_models_dir(settings)
    assert check.status == "fail" and check.essential
    assert "chmod" in check.fix


def test_engines_need_at_least_one(monkeypatch: pytest.MonkeyPatch) -> None:
    llama = doctor.Check("llama-server (GGUF)", "fail")
    mlx = doctor.Check("mlx-lm (MLX)", "warn")
    check = doctor.check_engines(llama, mlx)
    assert check.status == "fail" and check.essential
    assert "brew install llama.cpp" in check.fix


def test_engines_ok_with_one() -> None:
    check = doctor.check_engines(doctor.Check("a", "ok"), doctor.Check("b", "warn"))
    assert check.status == "ok" and check.detail == "GGUF"


def test_library_warns_when_empty(tmp_path: Path) -> None:
    settings = Settings(models_dir=str(tmp_path / "models"), scan_lmstudio=False,
                        scan_hf_cache=False)
    check = doctor.check_library(settings)
    assert check.status == "warn"
    assert "localcode models curated" in check.fix


async def test_server_check_without_a_server(isolated_home: Path) -> None:
    check = await doctor.check_server()
    assert check.status == "ok"
    assert "nie działa" in check.detail


async def test_server_check_with_a_stale_file(isolated_home: Path) -> None:
    (isolated_home / "server.json").write_text(
        json.dumps({"port": 1, "token": "t", "host": "127.0.0.1"}), encoding="utf-8")
    check = await doctor.check_server()
    assert check.status == "warn"
    assert "rm " in check.fix


def test_report_is_ok_unless_an_essential_check_failed() -> None:
    report = doctor.Report([doctor.Check("a", "warn"), doctor.Check("b", "fail")])
    assert report.ok is True
    report.add(doctor.Check("c", "fail", essential=True))
    assert report.ok is False
    assert len(report.problems) == 3


def test_report_to_dict() -> None:
    report = doctor.Report([doctor.Check("a", "ok", "szczegół", "napraw", True)])
    payload = report.to_dict()
    assert payload["ok"] is True
    assert payload["checks"][0] == {"name": "a", "status": "ok", "detail": "szczegół",
                                    "fix": "napraw", "essential": True}


def test_render_report(console: Console) -> None:
    report = doctor.Report([
        doctor.Check("System", "ok", "macOS 26 · Apple Silicon"),
        doctor.Check("Modele", "warn", "brak modeli", "localcode models curated"),
    ])
    doctor.render_report(report, console)
    out = console.export_text()
    assert "diagnostyka" in out
    assert "✔ System" in out
    assert "! Modele" in out
    assert "Co zrobić:" in out
    assert "localcode models curated" in out
    assert "Wszystko gotowe" in out


def test_render_report_of_a_broken_setup(console: Console) -> None:
    report = doctor.Report([doctor.Check("Silnik modeli", "fail", "brak", "brew install llama.cpp",
                                         True)])
    doctor.render_report(report, console)
    out = console.export_text()
    assert "✗ Silnik modeli" in out
    assert "Brakuje czegoś niezbędnego" in out


async def test_collect_runs_everything(tmp_path: Path, isolated_home: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    async def offline(settings):
        return doctor.Check("Hugging Face", "warn", "brak połączenia", "Sprawdź internet.")

    monkeypatch.setattr(doctor, "check_huggingface", offline)
    settings = Settings(models_dir=str(tmp_path / "models"), scan_lmstudio=False,
                        scan_hf_cache=False)
    report = await doctor.collect(settings)
    names = [c.name for c in report.checks]
    for expected in ("System", "Python", "Pamięć", "Wolne miejsce", "Silnik modeli",
                     "llama-server (GGUF)", "mlx-lm (MLX)", "Katalog modeli", "Modele",
                     "Serwer LocalCode", "Hugging Face", "Budżet modeli"):
        assert expected in names


async def test_run_doctor_returns_an_exit_code(console: Console, tmp_path: Path,
                                               isolated_home: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_collect(settings=None):
        return doctor.Report([doctor.Check("x", "fail", essential=True)])

    monkeypatch.setattr(doctor, "collect", fake_collect)
    assert await doctor.run_doctor(console) == 1


def test_plural() -> None:
    assert doctor._plural(1, "model", "modele", "modeli") == "model"
    assert doctor._plural(3, "model", "modele", "modeli") == "modele"
    assert doctor._plural(12, "model", "modele", "modeli") == "modeli"
    assert doctor._plural(22, "model", "modele", "modeli") == "modele"


@pytest.mark.network
async def test_huggingface_probe_really_works() -> None:
    check = await doctor.check_huggingface(Settings())
    assert check.status in ("ok", "warn")


# --------------------------------------------------------------------------- #
# Image backends
# --------------------------------------------------------------------------- #


def test_check_images_reports_the_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: a fresh `./scripts/install.sh` could not generate images.

    The installer only installed the `mlx` extra, so `localcode image …` failed
    with "Generowanie obrazów wymaga pakietu …" while `localcode doctor` said
    everything was fine. Doctor now says which backend is missing.
    """
    monkeypatch.setattr(doctor, "_module_version", lambda name: None)
    check = doctor.check_images()
    assert check.status == "warn"
    assert not check.essential                    # images are optional, not fatal
    assert "localcode[images]" in check.fix


def test_check_images_is_ok_with_one_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_module_version",
                        lambda name: "0.20" if name == "mflux" else None)
    check = doctor.check_images()
    assert check.status == "ok"
    assert "mflux" in check.detail


async def test_collect_includes_the_image_check(tmp_path: Path, isolated_home: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    async def offline(settings):
        return doctor.Check("Hugging Face", "warn", "brak", "Sprawdź internet.")

    monkeypatch.setattr(doctor, "check_huggingface", offline)
    report = await doctor.collect(Settings(models_dir=str(tmp_path / "models"),
                                           scan_lmstudio=False, scan_hf_cache=False))
    assert "Generowanie obrazów" in [c.name for c in report.checks]


def test_store_index_check_reports_the_number_of_models(tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    from localcode.hub import store_data

    monkeypatch.setattr(store_data, "count", lambda: 6000)
    monkeypatch.setattr(store_data, "generated_at", lambda: "2026-09-26T00:00:00Z")
    check = doctor.check_store_index()
    assert check.status == "ok"
    assert "6000" in check.detail and "2026-09-26" in check.detail


def test_store_index_check_warns_when_the_catalogue_is_missing(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """An installation without the generated file still works — with sixty
    models instead of thousands — so this is a warning with a way out."""
    from localcode.hub import store_data

    monkeypatch.setattr(store_data, "count", lambda: 0)
    check = doctor.check_store_index()
    assert check.status == "warn"
    assert "build_catalog" in (check.fix or "")


async def test_collect_includes_the_store_index_check(tmp_path: Path, isolated_home: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    async def offline(settings):
        return doctor.Check("Hugging Face", "warn", "brak", "Sprawdź internet.")

    monkeypatch.setattr(doctor, "check_huggingface", offline)
    report = await doctor.collect(Settings(models_dir=str(tmp_path / "models"),
                                           scan_lmstudio=False, scan_hf_cache=False))
    assert "Sklep z modelami" in [c.name for c in report.checks]
