"""Tests for ``localcode.images.base`` and the two backend engines.

The worker protocol is exercised against a *real* subprocess: a tiny aiohttp
program (written to ``tmp_path`` and spawned by :class:`FakeImageEngine`) speaks
the same ``/health``, ``/props``, ``/generate``, ``/cancel``, ``/memory``
surface as the mflux and stable-diffusion.cpp workers.  That covers the loading
phase (503), a crash during start, SSE progress/image/error frames, soft and
hard cancellation and process teardown — without any model.

The tests at the end check the *installed* third-party APIs the workers call
(mflux 0.20, stable_diffusion_cpp 0.4) so a breaking upgrade fails here instead
of at generation time; they skip when the package is not installed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json

import sys
import time
from pathlib import Path
from typing import Any

import psutil
import pytest

from localcode.hub.types import LocalModel, ModelMeta
from localcode.images.base import (
    IMAGE_FAMILIES,
    ImageEngineError,
    ImageEngineStartError,
    ImageEngineUnavailableError,
    ImageGenerationCancelled,
    ImageParams,
    ImageProgress,
    ImageResult,
    WorkerImageEngine,
    backend_missing_message,
    default_steps_for,
    family_defaults,
    new_image_id,
)
from localcode.images.mflux_engine import MfluxEngine
from localcode.images.sdcpp_engine import SdcppEngine

# --------------------------------------------------------------------------- #
# Fake worker
# --------------------------------------------------------------------------- #

FAKE_WORKER = r'''
import argparse, asyncio, base64, json, os, sys, time
from pathlib import Path
from aiohttp import web

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--parent-pid", type=int, default=None)
ap.add_argument("--mode", default="ok")          # ok | crash | slow | error | stall
ap.add_argument("--load-delay", type=float, default=0.2)
args, _unknown = ap.parse_known_args()

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==")

if args.mode == "crash":
    for i in range(40):
        print(f"worker log line {i}", flush=True)
    print("RuntimeError: nie udalo sie wczytac modelu", flush=True)
    sys.exit(3)

state = {"ready": False, "cancel": False, "generating": False}

async def health(request):
    if not state["ready"]:
        return web.json_response({"error": {"code": 503, "message": "Loading model"}}, status=503)
    return web.json_response({"status": "ok"})

async def props(request):
    return web.json_response({"backend": "fake", "family": "flux", "hardCancel": args.mode == "hard",
                              "defaults": {"steps": 4, "guidance": None, "sizes": ["1024x1024"]}})

async def memory(request):
    return web.json_response({"bytes": 123456789})

async def cancel(request):
    state["cancel"] = True
    return web.json_response({"ok": True, "hard": args.mode == "hard"})

async def generate(request):
    body = await request.json()
    outputs = body["outputs"]
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)

    async def send(frame):
        await resp.write(b"data: " + json.dumps(frame).encode() + b"\n\n")

    state["generating"] = True
    if args.mode == "error":
        await send({"type": "error", "message": "Backend padl"})
        await resp.write_eof()
        return resp
    steps = int(body.get("steps") or 4)
    for index, out in enumerate(outputs):
        for step in range(1, steps + 1):
            if state["cancel"]:
                await send({"type": "error", "cancelled": True, "message": "Przerwano"})
                await resp.write_eof()
                return resp
            await send({"type": "progress", "index": index, "step": step, "steps": steps,
                        "count": len(outputs)})
            await asyncio.sleep(0.05 if args.mode == "stall" else 0.005)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(PNG)
        await send({"type": "image", "index": index, "path": out, "seed": 1000 + index,
                    "width": body["width"], "height": body["height"], "steps": steps,
                    "guidance": body.get("guidance"), "elapsedMs": 12})
    await send({"type": "done"})
    await resp.write_eof()
    state["generating"] = False
    return resp

async def main():
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/props", props)
    app.router.add_get("/memory", memory)
    app.router.add_post("/generate", generate)
    app.router.add_post("/cancel", cancel)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, args.host, args.port).start()
    print("fake image worker listening", flush=True)
    await asyncio.sleep(args.load_delay)
    state["ready"] = True
    print("fake image worker ready", flush=True)
    await asyncio.Event().wait()

asyncio.run(main())
'''


def make_model(key: str = "org/flux-test", fmt: str = "mlx", family: str | None = "flux",
               path: str | None = None, components: dict[str, str] | None = None) -> LocalModel:
    meta = ModelMeta(quant="4bit")
    meta.image_family = family
    if components:
        meta.image_components = dict(components)
    return LocalModel(
        key=key, format=fmt, path=path or f"/models/{key}", display_name=key.split("/")[-1],
        size_bytes=4_000_000_000, repo_id=key, quant="4bit", capabilities=["image"], meta=meta)


class FakeImageEngine(WorkerImageEngine):
    """Runs the fake worker above instead of a real backend."""

    backend = "fake"
    format = "mlx"
    startup_timeout_base_s = 20.0
    startup_timeout_per_gb_s = 0.0

    def __init__(self, script: Path, *, mode: str = "ok", load_delay: float = 0.2,
                 model: LocalModel | None = None) -> None:
        super().__init__("fake-1", model or make_model())
        self.script = script
        self.mode = mode
        self.load_delay = load_delay

    def _worker_module(self) -> str:  # pragma: no cover - not used, _build_command is overridden
        return "fake"

    def _worker_args(self) -> list[str]:
        return ["--mode", self.mode, "--load-delay", str(self.load_delay)]

    def _build_command(self, port: int) -> list[str]:
        return [sys.executable, str(self.script), "--host", self.host, "--port", str(port),
                *self._worker_args()]


@pytest.fixture
def worker_script(tmp_path: Path) -> Path:
    script = tmp_path / "fake_image_worker.py"
    script.write_text(FAKE_WORKER)
    return script


@pytest.fixture
def home(tmp_path, monkeypatch) -> Path:
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    monkeypatch.setenv("LOCALCODE_HOME", str(home_dir))
    return home_dir


async def run_engine(engine: WorkerImageEngine, params: ImageParams, out: Path,
                     ids: list[str] | None = None) -> tuple[list[ImageProgress], list[ImageResult]]:
    progress: list[ImageProgress] = []
    images: list[ImageResult] = []
    stream = engine.generate(params, output_dir=out, ids=ids)
    async with contextlib.aclosing(stream):
        async for item in stream:
            (progress if isinstance(item, ImageProgress) else images).append(item)
    return progress, images


# --------------------------------------------------------------------------- #
# ImageParams / helpers
# --------------------------------------------------------------------------- #


def test_params_normalization_rounds_and_clamps():
    p = ImageParams(prompt="  kot  ", width=1000, height=1029, steps=999, count=99).normalized(multiple=64)
    assert p.prompt == "kot"
    assert p.width == 960 and p.height == 1024        # rounded down to the 64 grid
    assert p.steps == 150 and p.count == 8            # clamped
    assert ImageParams(prompt="x", width=10, height=10).normalized(multiple=16).width == 128


def test_params_reject_empty_prompt_and_missing_init_image(tmp_path):
    with pytest.raises(ImageEngineError) as exc:
        ImageParams(prompt="   ").normalized()
    assert "prompt" in str(exc.value).lower()
    with pytest.raises(ImageEngineError) as exc:
        ImageParams(prompt="kot", init_image=str(tmp_path / "nope.png")).normalized()
    assert "Nie znaleziono" in str(exc.value)


def test_params_seeds_are_deterministic_or_random():
    fixed = ImageParams(prompt="kot", seed=7, count=3).seeds()
    assert fixed == [7, 8, 9]
    random_seeds = ImageParams(prompt="kot", count=3).seeds()
    assert len(random_seeds) == 3 and all(0 <= s for s in random_seeds)


def test_params_payload_round_trip():
    p = ImageParams(prompt="kot", negative_prompt="rozmycie", width=768, height=512, steps=6,
                    guidance=3.5, seed=42, count=2, init_image=None, strength=0.4)
    assert ImageParams.from_payload(p.to_payload()) == p


def test_progress_fraction_and_dto():
    assert ImageProgress(steps=0).fraction is None
    assert ImageProgress(index=1, step=2, steps=4, count=2).fraction == pytest.approx(0.75)
    dto = ImageProgress(index=0, step=1, steps=4, message="Generowanie…").to_dict()
    assert dto["step"] == 1 and dto["message"] == "Generowanie…" and dto["fraction"] == 0.25


def test_result_dto_is_camel_case():
    result = ImageResult(id="a", path="/tmp/a.png", width=1024, height=768, seed=5,
                         model_key="org/model", prompt="kot", steps=4, guidance=None,
                         elapsed_ms=120, created_at=1_700_000_000.0)
    dto = result.to_dict()
    assert dto["modelKey"] == "org/model" and dto["createdAt"] == 1_700_000_000_000
    assert dto["elapsedMs"] == 120 and dto["guidance"] is None


def test_family_defaults_and_steps():
    assert set(IMAGE_FAMILIES) >= {"flux", "flux2", "qwen-image", "sd1", "sdxl", "sd3"}
    assert family_defaults("sd1")["steps"] == 20
    assert family_defaults("nic-takiego")["steps"] == family_defaults("unknown")["steps"]
    assert default_steps_for("flux", "dhairyashil/FLUX.1-schnell-mflux-4bit") == 4
    assert default_steps_for("sdxl", "org/sdxl-base") == 25
    assert default_steps_for("flux2", "Runpod/FLUX.2-klein-4B-mflux-4bit") == 4


def test_backend_missing_message_is_polish():
    msg = backend_missing_message("mflux")
    assert "mflux" in msg and "pip install" in msg
    assert "stable-diffusion-cpp-python" in backend_missing_message("sdcpp")


def test_new_image_id_is_unique_and_sortable():
    ids = [new_image_id() for _ in range(5)]
    assert len(set(ids)) == 5
    assert ids == sorted(ids) or all(len(i) == len(ids[0]) for i in ids)


# --------------------------------------------------------------------------- #
# Worker protocol (real subprocess)
# --------------------------------------------------------------------------- #


async def test_engine_starts_streams_and_stops(worker_script, tmp_path, home):
    engine = FakeImageEngine(worker_script, load_delay=0.3)
    try:
        await engine.start()
        assert engine.running and engine.pid and engine.port
        assert engine.props["family"] == "flux"
        assert await engine.memory_bytes() == 123456789
        progress, images = await run_engine(
            engine, ImageParams(prompt="kot w kapeluszu", width=512, height=512, steps=3, count=2),
            tmp_path / "out", ids=["a", "b"])
        assert [p.step for p in progress if p.index == 0] == [1, 2, 3]
        assert [i.id for i in images] == ["a", "b"]
        assert all(Path(i.path).is_file() and i.size_bytes > 0 for i in images)
        assert images[0].model_key == engine.model.key and images[0].prompt == "kot w kapeluszu"
        pid = engine.pid
    finally:
        await engine.stop()
    assert not engine.running
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


async def test_engine_start_failure_reports_log_tail(worker_script, home):
    engine = FakeImageEngine(worker_script, mode="crash")
    with pytest.raises(ImageEngineStartError) as exc:
        await engine.start()
    assert "nie udalo sie wczytac modelu" in exc.value.log_tail
    assert "Nie udało się wczytać modelu" in str(exc.value)
    assert not engine.running


async def test_engine_reports_backend_error_frame(worker_script, tmp_path, home):
    engine = FakeImageEngine(worker_script, mode="error")
    try:
        await engine.start()
        with pytest.raises(ImageEngineError) as exc:
            await run_engine(engine, ImageParams(prompt="kot"), tmp_path / "out")
        assert "Backend padl" in str(exc.value)
    finally:
        await engine.stop()


async def test_soft_cancel_ends_the_stream(worker_script, tmp_path, home):
    engine = FakeImageEngine(worker_script, mode="stall")
    try:
        await engine.start()
        params = ImageParams(prompt="kot", steps=40, count=1)
        stream = engine.generate(params, output_dir=tmp_path / "out")
        collected: list[Any] = []
        with pytest.raises(ImageGenerationCancelled):
            async with contextlib.aclosing(stream):
                async for item in stream:
                    collected.append(item)
                    if len(collected) == 2:
                        assert await engine.cancel() is True
        assert engine.running          # a soft cancel keeps the worker alive
    finally:
        await engine.stop()


async def test_hard_cancel_stops_the_process(worker_script, home):
    engine = FakeImageEngine(worker_script, mode="hard")
    try:
        await engine.start()
        pid = engine.pid
        assert await engine.cancel() is False        # backend cannot abort -> process killed
        assert not engine.running
        assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        await engine.stop()


async def test_generate_without_start_raises(tmp_path, home, worker_script):
    engine = FakeImageEngine(worker_script)
    with pytest.raises(ImageEngineUnavailableError):
        await run_engine(engine, ImageParams(prompt="kot"), tmp_path / "out")


async def test_stop_is_idempotent(worker_script, home):
    engine = FakeImageEngine(worker_script)
    await engine.start()
    await asyncio.gather(engine.stop(), engine.stop())
    await engine.stop()
    assert not engine.running


# --------------------------------------------------------------------------- #
# Command building
# --------------------------------------------------------------------------- #


def test_mflux_engine_command(home):
    model = make_model("Runpod/FLUX.2-klein-4B-mflux-4bit", fmt="mlx", family="flux2",
                       path="/models/Runpod/FLUX.2-klein-4B-mflux-4bit")
    engine = MfluxEngine("i-1", model, base_model="klein-4b", default_steps=4)
    cmd = engine._build_command(1234)
    assert cmd[1:3] == ["-m", "localcode.images.mflux_worker"]
    assert "--model" in cmd and "/models/Runpod/FLUX.2-klein-4B-mflux-4bit" in cmd
    assert cmd[cmd.index("--family") + 1] == "flux2"
    assert cmd[cmd.index("--base-model") + 1] == "klein-4b"
    assert cmd[cmd.index("--steps") + 1] == "4"
    assert cmd[cmd.index("--port") + 1] == "1234"
    assert engine.size_multiple() == 16


def test_sdcpp_engine_command_self_contained(home):
    model = make_model("second-state/sd15/sd15-Q4_0", fmt="gguf", family="sd1",
                       path="/models/sd15-Q4_0.gguf")
    engine = SdcppEngine("i-2", model, default_steps=20, threads=4, flash_attn=True)
    cmd = engine._build_command(4321)
    assert cmd[1:3] == ["-m", "localcode.images.sdcpp_worker"]
    assert cmd[cmd.index("--model") + 1] == "/models/sd15-Q4_0.gguf"
    assert "--diffusion-model" not in cmd
    assert cmd[cmd.index("--family") + 1] == "sd1"
    assert "--flash-attn" in cmd and cmd[cmd.index("--threads") + 1] == "4"
    assert engine.size_multiple() == 64


def test_sdcpp_engine_command_split_setup(home):
    model = make_model(
        "second-state/FLUX.1-schnell-GGUF/flux1-schnell-Q4_0", fmt="gguf", family="flux",
        path="/models/flux1-schnell-Q4_0.gguf",
        components={"unet": "/models/flux1-schnell-Q4_0.gguf", "clip_l": "/models/clip_l.gguf",
                    "t5xxl": "/models/t5xxl-Q4_0.gguf", "vae": "/models/ae.safetensors"})
    cmd = SdcppEngine("i-3", model)._build_command(9)
    assert cmd[cmd.index("--diffusion-model") + 1] == "/models/flux1-schnell-Q4_0.gguf"
    assert cmd[cmd.index("--clip-l") + 1] == "/models/clip_l.gguf"
    assert cmd[cmd.index("--t5xxl") + 1] == "/models/t5xxl-Q4_0.gguf"
    assert cmd[cmd.index("--vae") + 1] == "/models/ae.safetensors"
    assert "--model" not in cmd


async def test_engines_refuse_to_start_without_the_backend(monkeypatch, home):
    monkeypatch.setattr("localcode.images.mflux_engine.mflux_available", lambda *_a, **_k: False)
    monkeypatch.setattr("localcode.images.sdcpp_engine.sdcpp_available", lambda *_a, **_k: False)
    with pytest.raises(ImageEngineError) as exc:
        await MfluxEngine("i", make_model()).start()
    assert "mflux" in str(exc.value) and "pip install" in str(exc.value)
    with pytest.raises(ImageEngineError) as exc:
        await SdcppEngine("i", make_model(fmt="gguf", family="sd1")).start()
    assert "stable-diffusion-cpp-python" in str(exc.value)


# --------------------------------------------------------------------------- #
# Installed third-party APIs the workers depend on
# --------------------------------------------------------------------------- #

try:  # the image extra is optional: these tests check it only when installed
    import mflux  # noqa: F401
except ImportError:  # pragma: no cover - environment without localcode[images]
    mflux = None


@pytest.mark.skipif(mflux is None, reason="mflux not installed")
def test_mflux_model_classes_and_signatures_match_the_worker():
    """The worker's class table and ``generate_image`` kwargs must exist in mflux."""
    import inspect

    from localcode.images.mflux_worker import _MODEL_CLASSES, MfluxWorker

    keys = ["schnell", "dev", "krea-dev", "flux2-klein-4b", "qwen-image", "qwen-image-2.1"]
    for key in keys:
        cls = MfluxWorker._model_class(key)
        assert inspect.isclass(cls), key
        init = set(inspect.signature(cls.__init__).parameters)
        assert {"model_config", "quantize", "model_path"} <= init, (key, init)
        gen = set(inspect.signature(cls.generate_image).parameters)
        assert {"seed", "prompt", "num_inference_steps", "width", "height", "guidance"} <= gen, key
        assert {"image_path", "image_strength"} <= gen, key
    # every entry of the table must import
    for _prefix, module_name, class_name in _MODEL_CLASSES:
        module = __import__(module_name, fromlist=[class_name])
        assert hasattr(module, class_name), (module_name, class_name)


@pytest.mark.skipif(mflux is None, reason="mflux not installed")
def test_mflux_resolves_the_curated_repo_names():
    """``ModelConfig`` must infer the right base model from our curated names."""
    from mflux.models.common.config.model_config import ModelConfig
    from mflux.models.common.resolution.config_resolution import ConfigResolution

    from localcode.images.mflux_worker import family_for_key

    cases = {
        "Runpod/FLUX.2-klein-4B-mflux-4bit": ("flux2-klein-4b", "flux2"),
        "mlx-community/flux2-klein-9b-4bit": ("flux2-klein-9b", "flux2"),
        "dhairyashil/FLUX.1-schnell-mflux-4bit": ("schnell", "flux"),
        "filipstrand/FLUX.1-Krea-dev-mflux-4bit": ("krea-dev", "flux"),
        "mlx-community/Qwen-Image-2512-4bit": ("qwen-image", "qwen-image"),
    }
    for name, (expected_key, expected_family) in cases.items():
        assert ModelConfig.from_name(model_name=name) is not None
        key = ConfigResolution.resolve_key(model_name=name)
        assert key == expected_key, (name, key)
        assert family_for_key(key) == expected_family


@pytest.mark.skipif(mflux is None, reason="mflux not installed")
def test_mflux_progress_callback_plumbing_and_cancellation():
    """Our callback really receives mflux's per-step calls and can cancel."""
    from mflux.callbacks.callback_registry import CallbackRegistry

    from localcode.images.base import CancelledJob
    from localcode.images.mflux_worker import _ProgressCallback

    class Recorder:
        def __init__(self) -> None:
            self.frames: list[dict[str, Any]] = []
            self._cancel = False

        def emit_progress(self, *, step: int, steps: int, message: str = "") -> None:
            self.frames.append({"step": step, "steps": steps, "message": message})

        def cancelled(self) -> bool:
            return self._cancel

    class Config:
        num_inference_steps = 4
        time_steps = range(4)

    recorder = Recorder()
    registry = CallbackRegistry()
    registry.register(_ProgressCallback(recorder))
    ctx = registry.start(seed=1, prompt="kot", config=Config())
    ctx.before_loop(latents=None)
    ctx.in_loop(t=0, latents=None)
    ctx.in_loop(t=1, latents=None)
    assert [f["step"] for f in recorder.frames] == [0, 1, 2]
    assert [f["steps"] for f in recorder.frames] == [4, 4, 4]
    recorder._cancel = True
    with pytest.raises(CancelledJob):
        ctx.in_loop(t=2, latents=None)
    ctx.after_loop(latents=None)
    assert recorder.frames[-1]["message"] == "Dekodowanie obrazu…"


def test_sdcpp_generate_image_accepts_the_worker_kwargs():
    """Every kwarg ``SdcppWorker.run_job`` passes must exist in the installed binding."""
    sd = pytest.importorskip("stable_diffusion_cpp", reason="localcode[images] not installed")
    import inspect

    init = set(inspect.signature(sd.StableDiffusion.__init__).parameters)
    assert {"model_path", "diffusion_model_path", "clip_l_path", "clip_g_path", "t5xxl_path",
            "vae_path", "taesd_path", "n_threads", "flash_attn", "diffusion_flash_attn",
            "vae_decode_only", "verbose"} <= init
    gen = set(inspect.signature(sd.StableDiffusion.generate_image).parameters)
    assert {"prompt", "negative_prompt", "width", "height", "cfg_scale", "guidance",
            "sample_steps", "sample_method", "seed", "batch_count", "init_image", "strength",
            "progress_callback"} <= gen
    assert hasattr(sd.StableDiffusion, "close")


def test_sdcpp_worker_saves_png_metadata(tmp_path):
    """``_save`` writes a valid PNG with the generation parameters embedded."""
    pytest.importorskip("PIL", reason="localcode[images] not installed")
    from PIL import Image

    from localcode.images.sdcpp_worker import SdcppWorker

    worker = SdcppWorker(model="/models/sd15.gguf", family="sd1", name="SD 1.5")
    image = Image.new("RGB", (8, 8), (10, 120, 200))
    target = tmp_path / "out.png"
    params = ImageParams(prompt="kot w kapeluszu", negative_prompt="rozmycie", width=8, height=8)
    worker._save(image, target, params, seed=7, steps=20, guidance=7.0)
    assert target.is_file()
    with Image.open(target) as saved:
        assert saved.size == (8, 8)
        assert "kot w kapeluszu" in saved.text["parameters"]
        assert json.loads(saved.text["localcode"])["seed"] == 7


@pytest.mark.parametrize("name,expected", [
    ("FLUX.1 Fill dev pure · Q4_0", True),
    ("/m/gpustack/FLUX.1-Fill-dev-GGUF/FLUX.1-Fill-dev-pure-Q4_0.gguf", True),
    ("sd-v1-5-inpainting.gguf", True),
    ("FLUX.1 dev Q4_0", False),
    ("stable-diffusion-v1-5-pruned-emaonly-Q8_0.gguf", False),
    ("refill", False),
    ("backfilled-model.gguf", False),
    ("FLUX.1 Krea dev Q4_0", False),
])
def test_is_inpaint_model_detects_fill_and_inpainting_names(name, expected):
    from localcode.images.sdcpp_worker import is_inpaint_model

    assert is_inpaint_model(name) is expected
    assert is_inpaint_model(None, "", name) is expected


class _FakeSd:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def generate_image(self, **kwargs):
        from PIL import Image

        self.calls.append(kwargs)
        return [Image.new("RGB", (kwargs["width"], kwargs["height"]), (1, 2, 3))]


@pytest.mark.parametrize("name,inpaint,init_is_image,strength", [
    ("FLUX.1 Fill dev pure · Q4_0", True, True, 1.0),
    ("FLUX.1 dev Q4_0", False, False, 0.75),
])
def test_sdcpp_worker_run_job_blank_canvas_only_for_inpaint_models(
        tmp_path, name, inpaint, init_is_image, strength):
    pytest.importorskip("PIL", reason="localcode[images] not installed")
    from PIL import Image

    from localcode.images.sdcpp_worker import SdcppWorker

    worker = SdcppWorker(model="/models/m.gguf", family="flux", name=name)
    assert worker.inpaint is inpaint
    worker._sd = fake = _FakeSd()
    params = ImageParams(prompt="jablko", width=64, height=128, steps=2)
    events: list[dict[str, Any]] = []
    worker.run_job(params, [str(tmp_path / "o.png")], events.append)
    (call,) = fake.calls
    if init_is_image:
        assert isinstance(call["init_image"], Image.Image)
        assert call["init_image"].size == (64, 128)
    else:
        assert call["init_image"] is None
    assert call["strength"] == strength
    assert (tmp_path / "o.png").is_file()


def test_sdcpp_worker_load_keeps_the_vae_encoder(tmp_path, monkeypatch):
    sd = pytest.importorskip("stable_diffusion_cpp", reason="localcode[images] not installed")
    from localcode.images.sdcpp_worker import SdcppWorker

    seen: dict[str, Any] = {}

    class Recorder:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    monkeypatch.setattr(sd, "StableDiffusion", Recorder)
    model = tmp_path / "m.gguf"
    model.write_bytes(b"x")
    SdcppWorker(model=str(model), family="sd1").load()
    assert seen["vae_decode_only"] is False
    assert seen["model_path"] == str(model)


def test_worker_servers_declare_their_cancellation_ability():
    from localcode.images.mflux_worker import MfluxWorker
    from localcode.images.sdcpp_worker import SdcppWorker

    assert MfluxWorker.hard_cancel is False       # can stop between denoising steps
    assert SdcppWorker.hard_cancel is True        # C library has no abort callback


def test_mflux_worker_resolution_name_uses_the_repo_directory(tmp_path):
    from localcode.images.mflux_worker import MfluxWorker

    repo = tmp_path / "Runpod" / "FLUX.2-klein-4B-mflux-4bit"
    (repo / "transformer").mkdir(parents=True)
    worker = MfluxWorker(str(repo))
    assert worker._resolution_name() == "Runpod/FLUX.2-klein-4B-mflux-4bit"
    assert MfluxWorker("dhairyashil/FLUX.1-schnell-mflux-4bit")._resolution_name() == \
        "dhairyashil/FLUX.1-schnell-mflux-4bit"


def test_worker_cli_parsers_accept_the_engine_arguments():
    """The engines' argv must parse in the workers (they are built independently)."""
    from localcode.images.mflux_worker import build_parser as mflux_parser
    from localcode.images.sdcpp_worker import build_parser as sdcpp_parser

    model = make_model("Runpod/FLUX.2-klein-4B-mflux-4bit", family="flux2")
    args = MfluxEngine("i", model, base_model="klein-4b", default_steps=4)._build_command(5000)[3:]
    parsed = mflux_parser().parse_args(args)
    assert parsed.port == 5000 and parsed.family == "flux2" and parsed.base_model == "klein-4b"

    gguf = make_model("x/sd3/sd3-Q4_0", fmt="gguf", family="sd3", path="/models/sd3-Q4_0.gguf",
                      components={"clip_l": "/m/clip_l.gguf", "t5xxl": "/m/t5.gguf"})
    args = SdcppEngine("i", gguf, threads=6)._build_command(5001)[3:]
    parsed = sdcpp_parser().parse_args(args)
    assert parsed.port == 5001 and parsed.family == "sd3" and parsed.t5xxl == "/m/t5.gguf"
    assert parsed.threads == 6


def test_time_and_ids_do_not_collide_under_load():
    start = time.time()
    ids = {new_image_id() for _ in range(500)}
    assert len(ids) == 500 and time.time() - start < 5
