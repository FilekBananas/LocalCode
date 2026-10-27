"""``localcode video`` — CLI wiring, argument validation and progress plumbing.

The real generation runtime is replaced with a fake (the way
``tests/tui/test_images_cli.py`` fakes the image client): what these tests
promise is the command's own behaviour — parsing, the estimate-before-start
call, threading every flag through to ``VideoRuntime.generate``, reporting a
:class:`~localcode.video.base.VideoError` on stderr with the right exit code,
and driving the progress bar through a realistic sequence of progress events
without crashing. What a real generation actually produces is not this
module's business (see ``tests/video/test_video_runtime.py`` and
``docs/VIDEO.md`` for what is and is not verified).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

import localcode.video as video_pkg
from localcode import cli
from localcode.tui import render
from localcode.video.base import INSTALL_HINT, VideoError, VideoProgress, VideoResult

PROMPT = "kot w kapeluszu idzie ulicą"


@pytest.fixture
def console() -> Console:
    """A recording console, no colours, wide enough that paths are not wrapped."""
    return Console(record=True, width=200, color_system=None, theme=render.THEME,
                   highlight=False, legacy_windows=False, force_terminal=False)


class FakeVideoRuntime:
    """Stands in for :class:`~localcode.video.runtime.VideoRuntime`."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.estimate_calls: list[dict[str, Any]] = []
        self.generate_calls: list[tuple[str, dict[str, Any]]] = []
        self.progress_seen: list[VideoProgress] = []
        #: fed to ``on_progress`` during ``generate`` — a realistic sequence
        #: (loading, several denoising steps, decoding).
        self.progress_events: list[VideoProgress] = [
            VideoProgress(stage="loading", step=0, steps=0, message="Wczytuję model…"),
            VideoProgress(stage="denoising", step=1, steps=3, message="Odszumianie: krok 1/3"),
            VideoProgress(stage="denoising", step=2, steps=3, message="Odszumianie: krok 2/3"),
            VideoProgress(stage="denoising", step=3, steps=3, message="Odszumianie: krok 3/3"),
            VideoProgress(stage="decoding", step=3, steps=3, message="Dekoduję klatki…"),
        ]
        self.estimate_error: VideoError | None = None
        self.generate_error: VideoError | None = None
        self.result: VideoResult | None = None

    def estimate(self, **kwargs: Any) -> dict[str, Any]:
        self.estimate_calls.append(kwargs)
        if self.estimate_error:
            raise self.estimate_error
        return {"seconds": 90.0, "message": "41 klatek — szacunkowo ~2 min liczenia (niezmierzone).",
                **{k: v for k, v in kwargs.items() if v is not None}}

    async def generate(self, prompt: str, *, on_progress: Any = None, **kwargs: Any) -> VideoResult:
        self.generate_calls.append((prompt, kwargs))
        for progress in self.progress_events:
            self.progress_seen.append(progress)
            if on_progress is not None:
                on_progress(progress)
        if self.generate_error:
            raise self.generate_error
        assert self.result is not None
        return self.result


@pytest.fixture
def video_cli(monkeypatch: pytest.MonkeyPatch, console: Console, tmp_path: Path) -> FakeVideoRuntime:
    fake = FakeVideoRuntime()
    fake.result = VideoResult(
        id="deadbeef", path=str(tmp_path / "wideo.mp4"), width=704, height=480, num_frames=41,
        fps=24, seed=123, steps=30, guidance=3.0, model_key="calcuis/ltxv-gguf/ltxv",
        prompt=PROMPT, negative_prompt="worst quality", elapsed_ms=125_000, size_bytes=4_000_000,
        estimated_seconds=90.0,
    )
    Path(fake.result.path).parent.mkdir(parents=True, exist_ok=True)
    Path(fake.result.path).write_bytes(b"\x00fake-mp4")

    monkeypatch.setattr(video_pkg, "VideoRuntime", lambda *a, **kw: fake)
    monkeypatch.setattr(cli, "_console", lambda: console)
    return fake


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def test_video_is_a_known_subcommand() -> None:
    assert "video" in cli.SUBCOMMANDS


async def test_video_command_needs_a_prompt(video_cli: FakeVideoRuntime) -> None:
    with pytest.raises(SystemExit):        # argparse: prompt is a required positional
        await cli._cmd_video([])
    assert video_cli.generate_calls == []


async def test_video_command_rejects_a_bad_size(video_cli: FakeVideoRuntime,
                                                capsys: pytest.CaptureFixture[str]) -> None:
    code = await cli._cmd_video([PROMPT, "--size", "ogromny"])
    assert code == 2
    assert "Nieprawidłowy rozmiar" in capsys.readouterr().err
    assert video_cli.generate_calls == []


# --------------------------------------------------------------------------- #
# Happy path
# --------------------------------------------------------------------------- #


async def test_video_command_estimates_before_generating(video_cli: FakeVideoRuntime,
                                                          console: Console) -> None:
    code = await cli._cmd_video([PROMPT])
    assert code == 0
    assert video_cli.estimate_calls, "the honest time estimate must run before generation starts"
    out = console.export_text()
    assert "szacunkowo" in out or "niezmierzone" in out       # the pre-run estimate line


async def test_video_command_reports_the_result(video_cli: FakeVideoRuntime, console: Console) -> None:
    code = await cli._cmd_video([PROMPT])
    assert code == 0
    out = console.export_text()
    assert "Gotowe" in out
    assert "704" in out and "480" in out
    assert "41" in out


async def test_video_command_threads_every_flag(video_cli: FakeVideoRuntime) -> None:
    code = await cli._cmd_video([
        PROMPT, "-m", "calcuis/ltxv-gguf", "--size", "512x384", "--frames", "200",
        "--fps", "12", "--steps", "10", "--guidance", "4.5", "--seed", "99",
        "--negative", "rozmyte", "-o", "/tmp/klip.mp4", "--force",
    ])
    assert code == 0
    prompt, kwargs = video_cli.generate_calls[0]
    assert prompt == PROMPT
    assert kwargs["model_key"] == "calcuis/ltxv-gguf"
    assert (kwargs["width"], kwargs["height"]) == (512, 384)
    assert kwargs["num_frames"] == 200
    assert kwargs["fps"] == 12
    assert kwargs["steps"] == 10
    assert kwargs["guidance"] == 4.5
    assert kwargs["seed"] == 99
    assert kwargs["negative_prompt"] == "rozmyte"
    assert kwargs["output"] == "/tmp/klip.mp4"
    assert kwargs["force"] is True


async def test_video_command_defaults_are_left_to_the_runtime(video_cli: FakeVideoRuntime) -> None:
    """Unset flags stay ``None`` — ``VideoRuntime``/``VideoParams`` own the defaults."""
    await cli._cmd_video([PROMPT])
    _prompt, kwargs = video_cli.generate_calls[0]
    for key in ("model_key", "width", "height", "num_frames", "fps", "steps", "guidance", "seed",
                "negative_prompt", "output"):
        assert kwargs[key] is None
    assert kwargs["force"] is False


async def test_video_progress_events_reach_the_bar_without_crashing(video_cli: FakeVideoRuntime) -> None:
    """A realistic loading → denoising × N → decoding sequence must not blow up
    the progress bar wiring (this is the "progress events per step" contract)."""
    code = await cli._cmd_video([PROMPT])
    assert code == 0
    assert len(video_cli.progress_seen) == 5
    assert [p.stage for p in video_cli.progress_seen] == \
        ["loading", "denoising", "denoising", "denoising", "decoding"]


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


async def test_video_command_reports_a_frame_cap_error(video_cli: FakeVideoRuntime,
                                                        console: Console) -> None:
    video_cli.generate_error = VideoError(
        "300 klatek przekracza domyślny limit 81. Użyj --force.", "frame_cap_exceeded")
    code = await cli._cmd_video([PROMPT, "--frames", "300"])
    assert code == 1
    assert "przekracza domyślny limit" in console.export_text()


async def test_video_command_reports_the_missing_extra(video_cli: FakeVideoRuntime,
                                                        console: Console) -> None:
    video_cli.generate_error = VideoError(INSTALL_HINT, "missing_extra", 501)
    code = await cli._cmd_video([PROMPT])
    assert code == 1
    assert "localcode[video]" in console.export_text()


async def test_video_command_stops_before_generating_when_the_estimate_fails(
        video_cli: FakeVideoRuntime, console: Console) -> None:
    video_cli.estimate_error = VideoError(
        "9999 klatek przekracza bezwzględny limit 161.", "frame_hard_cap")
    code = await cli._cmd_video([PROMPT, "--frames", "9999", "--force"])
    assert code == 1
    assert "bezwzględny limit" in console.export_text()
    assert video_cli.generate_calls == []


async def test_video_command_reports_no_model_installed(video_cli: FakeVideoRuntime,
                                                         console: Console) -> None:
    video_cli.generate_error = VideoError(
        "Nie masz jeszcze modelu do generowania wideo (LTX-Video 2B). "
        "Zainstaluj go w sklepie (kategoria „Wideo”).", "no_model")
    code = await cli._cmd_video([PROMPT])
    assert code == 1
    assert "Wideo" in console.export_text()


# --------------------------------------------------------------------------- #
# Subcommand dispatch
# --------------------------------------------------------------------------- #


async def test_video_dispatches_through_run_subcommand(video_cli: FakeVideoRuntime) -> None:
    assert await cli._run_subcommand("video", [PROMPT]) == 0
    assert video_cli.generate_calls


# --------------------------------------------------------------------------- #
# The HTTP route may not write wherever the caller asks
# --------------------------------------------------------------------------- #


class _OutputRuntime:
    def __init__(self, root) -> None:
        self._root = root

    def output_dir(self):
        return self._root


def test_a_plain_name_lands_in_the_video_directory(tmp_path) -> None:
    from localcode.server.routes_video import _safe_output

    assert _safe_output(_OutputRuntime(tmp_path), "klip.mp4") == str(tmp_path / "klip.mp4")
    assert _safe_output(_OutputRuntime(tmp_path), "  ") is None
    assert _safe_output(_OutputRuntime(tmp_path), None) is None


@pytest.mark.parametrize("escape", [
    "/etc/passwd", "../../../.zshrc", "~/.ssh/id_rsa", "podkatalog/../../uciekło.mp4",
])
def test_the_route_refuses_to_write_outside_the_video_directory(tmp_path, escape: str) -> None:
    """The terminal writes where the user points it; the server does not.

    Anyone who can reach the port would otherwise be able to overwrite a file in
    the home directory by naming it in the request body.
    """
    from localcode.server.app import ApiError
    from localcode.server.routes_video import _safe_output

    with pytest.raises(ApiError) as exc:
        _safe_output(_OutputRuntime(tmp_path), escape)
    assert "output" in str(exc.value.reason if hasattr(exc.value, "reason") else exc.value)
