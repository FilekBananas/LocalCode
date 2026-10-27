"""Worker subprocess: runs one LTX-Video generation, reports progress on stdout.

Invoked as ``python -u -m localcode.video.worker`` by
:mod:`localcode.video.runtime`. Protocol (see the module docstring of
``runtime.py`` for the client side):

* stdin: one JSON object (params — see :func:`_read_payload`), then EOF.
* stdout: one JSON object per line —
  ``{"type": "progress", "stage": ..., "step": ..., "steps": ..., "message": ...}``,
  finally either
  ``{"type": "result", "path": ..., "width": ..., "height": ..., "numFrames": ...,
    "fps": ..., "seed": ..., "steps": ..., "guidance": ..., "sizeBytes": ...}``
  or ``{"type": "error", "message": "...", "code": "..."}`` (exit code 1).

Everything heavy (``torch``, ``diffusers``, ``transformers``) is imported here,
never in :mod:`localcode.video.base` or :mod:`localcode.video.runtime`, so the
rest of LocalCode stays importable without ``pip install "localcode[video]"``.

This file runs in its own process specifically so that a crash, a memory
allocation failure or a hung Metal kernel takes down one generation and not
the LocalCode server — the same reasoning as every other engine subprocess in
this codebase (``runtime/http_engine.py``, ``images/mflux_worker.py``).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from localcode.video.base import INSTALL_HINT


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _progress(stage: str, step: int, steps: int, message: str, started: float) -> None:
    _emit({"type": "progress", "stage": stage, "step": step, "steps": steps,
           "elapsedMs": int((time.monotonic() - started) * 1000), "message": message})


def _read_payload() -> dict[str, Any]:
    raw = sys.stdin.read()
    data = json.loads(raw) if raw.strip() else {}
    if not isinstance(data, dict):
        raise ValueError("payload musi być obiektem JSON")
    return data


def _fail(message: str, code: str = "generate_failed") -> None:
    _emit({"type": "error", "message": message, "code": code})
    sys.exit(1)


def main() -> None:
    started = time.monotonic()
    try:
        payload = _read_payload()
    except ValueError as exc:
        _fail(f"Nie udało się odczytać parametrów generowania: {exc}", "bad_payload")
        return

    model_path = str(payload.get("modelPath") or "")
    base_repo = str(payload.get("baseRepo") or "Lightricks/LTX-Video")
    output_path = str(payload.get("outputPath") or "")
    prompt = str(payload.get("prompt") or "")
    negative_prompt = str(payload.get("negativePrompt") or "")
    width = int(payload.get("width") or 704)
    height = int(payload.get("height") or 480)
    num_frames = int(payload.get("numFrames") or 41)
    fps = int(payload.get("fps") or 24)
    steps = int(payload.get("steps") or 30)
    guidance = float(payload.get("guidance") or 3.0)
    seed = int(payload.get("seed") or 0)

    if not model_path or not Path(model_path).is_file():
        _fail(f"Nie znaleziono pliku modelu: {model_path}", "model_not_found")
        return
    if not output_path:
        _fail("Brak ścieżki wyjściowej.", "missing_output")
        return

    _progress("loading", 0, 0, "Wczytuję LTX-Video (transformer GGUF + enkoder tekstu T5)…", started)
    try:
        import torch
        from diffusers import GGUFQuantizationConfig, LTXPipeline, LTXVideoTransformer3DModel
        from diffusers.utils import export_to_video
    except ImportError as exc:
        _fail(f"{INSTALL_HINT} (brakujący moduł: {exc.name or exc})", "missing_extra")
        return

    device = "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
    # bfloat16 on the GPU, float32 on the CPU. bfloat16 is what diffusers' own
    # LTX example uses and it costs about 15% more time than float16 here
    # (measured: 88 s vs 76 s for the same clip), so it is the safer default.
    # It is *not*, however, a fix for anything: the same clip came out as
    # coloured noise under both dtypes, pixel pattern nearly unchanged — see
    # docs/VIDEO.md for what is still unexplained about this pipeline.
    dtype = torch.bfloat16 if device == "mps" else torch.float32

    try:
        transformer = LTXVideoTransformer3DModel.from_single_file(
            model_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        pipe = LTXPipeline.from_pretrained(base_repo, transformer=transformer, torch_dtype=dtype)
        try:
            pipe = pipe.to(device)
        except Exception as exc:  # noqa: BLE001 - MPS op coverage is not guaranteed; fall back
            if device != "cpu":
                _progress("loading", 0, 0,
                          f"MPS odmówiło ({exc}); przechodzę na CPU (dużo wolniej)…", started)
                device = "cpu"
                pipe = pipe.to("cpu")
            else:
                raise
    except Exception as exc:  # noqa: BLE001 - reported to the caller, not a crash
        _fail(f"Nie udało się wczytać modelu LTX-Video: {exc}", "load_failed")
        return

    # The random generator is deliberately kept on CPU even when the pipeline
    # runs on MPS: several diffusers pipelines document this as the reliable
    # way to get reproducible noise on Apple Silicon (MPS RNG parity with
    # CUDA/CPU is not guaranteed). Unverified on real hardware here either.
    generator = torch.Generator("cpu").manual_seed(seed)

    def on_step_end(_pipe: Any, step_index: int, _timestep: Any,
                    callback_kwargs: dict[str, Any]) -> dict[str, Any]:
        _progress("denoising", step_index + 1, steps, f"Odszumianie: krok {step_index + 1}/{steps}",
                  started)
        return callback_kwargs

    try:
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            callback_on_step_end=on_step_end,
        )
    except Exception as exc:  # noqa: BLE001 - reported to the caller, not a crash
        _fail(f"Generowanie nie powiodło się: {exc}", "generate_failed")
        return

    _progress("decoding", steps, steps, "Dekoduję klatki i zapisuję plik wideo…", started)
    try:
        frames = output.frames[0]
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(target), fps=fps)
    except Exception as exc:  # noqa: BLE001
        _fail(f"Nie udało się zapisać pliku wideo: {exc}", "export_failed")
        return

    size_bytes = target.stat().st_size if target.exists() else 0
    _emit({
        "type": "result", "path": str(target), "width": width, "height": height,
        "numFrames": num_frames, "fps": fps, "seed": seed, "steps": steps,
        "guidance": guidance, "sizeBytes": size_bytes,
    })


if __name__ == "__main__":
    main()
