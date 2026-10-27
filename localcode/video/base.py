"""Contract of the video engine: parameters, progress, results and errors.

LocalCode targets exactly one model family today: **LTX-Video 2B**
(``Lightricks/LTX-Video``), the smallest text-to-video model in the store's
"wideo" shelf that has any realistic chance of running on a 32 GB Apple
Silicon Mac (see ``docs/VIDEO.md`` for why the other 90-odd catalogue entries
— Wan, HunyuanVideo, MiniMax, the 13B+ LTX-2 line — are out of scope).

The catalogue only ships **GGUF** checkpoints for video models (community
requantizations built for ComfyUI-GGUF, e.g. ``city96/LTX-Video-gguf``), so
this module loads them the same way ``diffusers`` documents for that exact
file shape: :class:`~diffusers.LTXVideoTransformer3DModel.from_single_file`
with a ``GGUFQuantizationConfig``, then :class:`~diffusers.LTXPipeline` from
the base ``Lightricks/LTX-Video`` repo with that transformer substituted in
(the other pipeline components — VAE, T5 text encoder, scheduler — come from
the base repo and are fetched by ``diffusers``/``huggingface_hub`` on first
use, *outside* LocalCode's own download manager; see ``docs/VIDEO.md``).
There is no MLX engine for this model: the MLX video ports that exist target
the newer, much larger LTX-2 (13-19B) family, not the 2B model this targets.

Everything heavy (``torch``, ``diffusers``, ``transformers``) lives behind the
``video`` extra and is only imported inside the worker subprocess
(:mod:`localcode.video.worker`); this module and :mod:`localcode.video.runtime`
stay importable without it, exactly like ``localcode.speech``.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from localcode.config import install_command

__all__ = [
    "DEFAULT_FPS",
    "DEFAULT_GUIDANCE",
    "DISTILLED_GUIDANCE",
    "default_guidance_for",
    "is_distilled",
    "DEFAULT_HEIGHT",
    "DEFAULT_NEGATIVE_PROMPT",
    "DEFAULT_NUM_FRAMES",
    "DEFAULT_STEPS",
    "DEFAULT_WIDTH",
    "INSTALL_HINT",
    "MAX_DIMENSION",
    "MAX_FRAMES_DEFAULT",
    "MAX_FRAMES_HARD",
    "MAX_STEPS",
    "MIN_DIMENSION",
    "MIN_FRAMES",
    "VideoError",
    "VideoParams",
    "VideoProgress",
    "VideoResult",
    "classify_video_model",
    "estimate_seconds",
    "new_video_id",
    "plan_message",
]

# Ścieżka interpretera, nie samo „pip” — patrz localcode.config.install_command.
INSTALL_HINT = (
    "Brakuje dodatku do wideo (diffusers + torch — generowanie działa przez "
    f"PyTorch/MPS, nie MLX). Zainstaluj: {install_command('video')}"
)

#: LTX-Video works on a fixed spatial/temporal compression grid: width and
#: height must be multiples of 32, frame counts must be ``8k + 1``
#: (verified against ``LTXPipeline.check_inputs`` in diffusers).
_DIM_MULTIPLE = 32
_FRAME_STEP = 8
MIN_DIMENSION = 256
MAX_DIMENSION = 768          # Lightricks' own recommended ceiling for the 2B model
MIN_FRAMES = 9               # 8*1 + 1
#: Soft cap: generation above this needs an explicit override (``force=True``).
#: 81 frames @24fps ≈ 3.4s of footage — already several minutes of compute here.
MAX_FRAMES_DEFAULT = 81
#: Absolute ceiling, not overridable — the model card's own "full sequence"
#: length. Nothing about this runtime has been measured beyond that point.
MAX_FRAMES_HARD = 161
MAX_STEPS = 100
DEFAULT_WIDTH = 704
DEFAULT_HEIGHT = 480
DEFAULT_FPS = 24
DEFAULT_STEPS = 30           # diffusers' example uses 50; lower by default because
                              # every step costs real minutes on this hardware.
DEFAULT_GUIDANCE = 3.0
#: A *distilled* LTX checkpoint is trained to run without classifier-free
#: guidance; feeding it the usual 3.0 does not sharpen the result, it destroys
#: it. Measured here: 448x256, 8 steps, guidance 3.0 on
#: ``LTX-Video-0.9.6-distilled`` produced a valid 25-frame H.264 file whose every
#: frame was coloured noise. Distilled checkpoints therefore default to 1.0.
DISTILLED_GUIDANCE = 1.0
DEFAULT_NUM_FRAMES = 41      # ≈1.7s @24fps — a first run should finish, not stall for an hour.
#: Lightricks' own recommended negative prompt (diffusers example, docs/VIDEO.md).
DEFAULT_NEGATIVE_PROMPT = "worst quality, inconsistent motion, blurry, jittery, distorted"
SEED_MAX = 2**31 - 1


class VideoError(Exception):
    """A video failure with a Polish, user-presentable message."""

    def __init__(self, message: str, code: str = "video_error", status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


def new_video_id() -> str:
    """Short, sortable, unique id of one generated clip."""
    return f"{int(time.time() * 1000):013d}-{uuid.uuid4().hex[:8]}"


def _round_to_multiple(value: int, multiple: int, lo: int, hi: int) -> int:
    value = max(lo, min(int(value), hi))
    rounded = int(round(value / multiple)) * multiple
    return max(lo, min(rounded, hi))


def _round_frames(value: int, lo: int, hi: int) -> int:
    """Nearest valid ``8k + 1`` frame count within ``[lo, hi]``."""
    value = max(lo, min(int(value), hi))
    k = round((value - 1) / _FRAME_STEP)
    rounded = k * _FRAME_STEP + 1
    return max(lo, min(rounded, hi))


# --------------------------------------------------------------------------- #
# Parameters / progress / results
# --------------------------------------------------------------------------- #


@dataclass
class VideoParams:
    """One generation request (see ``docs/VIDEO.md``)."""

    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    num_frames: int = DEFAULT_NUM_FRAMES
    fps: int = DEFAULT_FPS
    steps: int = DEFAULT_STEPS
    guidance: float = DEFAULT_GUIDANCE
    seed: int | None = None              # None = random, echoed back in the result
    force: bool = False                  # override MAX_FRAMES_DEFAULT (never MAX_FRAMES_HARD)

    def normalized(self, *, require_prompt: bool = True, enforce_frame_cap: bool = True) -> "VideoParams":
        """A validated copy with sizes/frames snapped to the model's grid.

        Raises :class:`VideoError` (Polish) for values that cannot be fixed
        silently: an empty prompt (unless ``require_prompt`` is False — used
        by the cost estimate, which may run before the user has typed one),
        or a frame count over the *default* cap without ``force`` (unless
        ``enforce_frame_cap`` is False, same reason). ``MAX_FRAMES_HARD`` is
        never bypassed by either flag — nothing about this runtime has been
        measured beyond it, so even a bare cost estimate refuses to guess.
        """
        prompt = (self.prompt or "").strip()
        if not prompt and require_prompt:
            raise VideoError("Podaj opis materiału wideo (prompt).", "empty_prompt")
        width = _round_to_multiple(self.width, _DIM_MULTIPLE, MIN_DIMENSION, MAX_DIMENSION)
        height = _round_to_multiple(self.height, _DIM_MULTIPLE, MIN_DIMENSION, MAX_DIMENSION)
        requested_frames = max(MIN_FRAMES, int(self.num_frames or DEFAULT_NUM_FRAMES))
        if requested_frames > MAX_FRAMES_DEFAULT and not self.force and enforce_frame_cap:
            raise VideoError(
                f"{requested_frames} klatek przekracza domyślny limit "
                f"{MAX_FRAMES_DEFAULT} (to już kilka minut liczenia na tym sprzęcie). "
                f"Użyj --force (albo \"force\": true), żeby wygenerować więcej, "
                f"maksymalnie {MAX_FRAMES_HARD}.",
                "frame_cap_exceeded",
            )
        if requested_frames > MAX_FRAMES_HARD:
            raise VideoError(
                f"{requested_frames} klatek przekracza bezwzględny limit {MAX_FRAMES_HARD} "
                "— tyle klatek nigdy nie było testowane z tym silnikiem.",
                "frame_hard_cap",
            )
        num_frames = _round_frames(requested_frames, MIN_FRAMES, MAX_FRAMES_HARD)
        steps = max(1, min(int(self.steps or DEFAULT_STEPS), MAX_STEPS))
        guidance = float(self.guidance if self.guidance is not None else DEFAULT_GUIDANCE)
        fps = max(1, min(int(self.fps or DEFAULT_FPS), 60))
        seed = random.randint(0, SEED_MAX) if self.seed is None else int(self.seed) % (SEED_MAX + 1)
        return VideoParams(
            prompt=prompt,
            negative_prompt=(self.negative_prompt or "").strip() or DEFAULT_NEGATIVE_PROMPT,
            width=width, height=height, num_frames=num_frames, fps=fps, steps=steps,
            guidance=guidance, seed=seed, force=self.force,
        )

    def to_payload(self) -> dict[str, Any]:
        """JSON body sent to the worker subprocess over stdin."""
        return {
            "prompt": self.prompt, "negativePrompt": self.negative_prompt,
            "width": self.width, "height": self.height, "numFrames": self.num_frames,
            "fps": self.fps, "steps": self.steps, "guidance": self.guidance, "seed": self.seed,
        }


@dataclass
class VideoProgress:
    """One progress tick of a running generation."""

    stage: str = "loading"    # "loading" | "denoising" | "decoding" | "encoding"
    step: int = 0
    steps: int = 0
    elapsed_ms: int = 0
    message: str = ""         # Polish status line

    @property
    def fraction(self) -> float | None:
        """Progress in 0…1, or None for indeterminate progress (loading/decoding)."""
        if self.steps <= 0:
            return None
        return max(0.0, min(1.0, self.step / self.steps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage, "step": self.step, "steps": self.steps,
            "elapsedMs": self.elapsed_ms, "message": self.message, "fraction": self.fraction,
        }


@dataclass
class VideoResult:
    """One finished clip on disk."""

    id: str
    path: str
    width: int
    height: int
    num_frames: int
    fps: int
    seed: int
    steps: int
    guidance: float
    model_key: str
    prompt: str
    negative_prompt: str = ""
    elapsed_ms: int = 0
    created_at: float = field(default_factory=time.time)
    size_bytes: int = 0
    estimated_seconds: float = 0.0     # the pre-run estimate, echoed back for reference

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "path": self.path, "width": self.width, "height": self.height,
            "numFrames": self.num_frames, "fps": self.fps, "seed": self.seed,
            "steps": self.steps, "guidance": self.guidance, "modelKey": self.model_key,
            "prompt": self.prompt, "negativePrompt": self.negative_prompt,
            "elapsedMs": self.elapsed_ms, "createdAt": int(self.created_at * 1000),
            "sizeBytes": self.size_bytes, "estimatedSeconds": round(self.estimated_seconds, 1),
        }


# --------------------------------------------------------------------------- #
# Time estimate
# --------------------------------------------------------------------------- #

#: Seconds per denoising step, per frame, at the reference resolution
#: (704x480) on Apple Silicon. **Measured**, on an M-series Mac with 32 GB and
#: the Q4_K_M checkpoint, from two runs that differ only in resolution:
#:
#:     512x320, 25 frames,  8 steps -> 73,9 s
#:     704x480, 25 frames,  8 steps -> 86,7 s
#:
#: Solving those two for a fixed overhead plus a per-pixel-per-frame-per-step
#: term gives the constants below; they reproduce both measurements to within
#: 0,2 s. The first guess here was 0,55 with 110 s of overhead, which promised
#: "~4 min" for a job that takes a minute and a half.
#:
#: Still an estimate, not a promise: one machine, one checkpoint, one quant.
_SECONDS_PER_STEP_PER_FRAME_REFERENCE = 0.124
_REFERENCE_PIXELS = DEFAULT_WIDTH * DEFAULT_HEIGHT
_MODEL_LOAD_OVERHEAD_S = 45.0        # load of transformer + T5 text encoder + VAE
_DECODE_OVERHEAD_S = 17.0            # VAE decode + video encode, roughly flat


def estimate_seconds(*, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
                     num_frames: int = DEFAULT_NUM_FRAMES, steps: int = DEFAULT_STEPS) -> float:
    """A coarse, clearly-labelled-as-rough time estimate in seconds.

    Scales linearly with pixel count, frame count and step count from one
    unverified reference constant. See the module docstring above the
    constant it reads — this is for setting expectations, not for a
    trustworthy ETA.
    """
    pixel_ratio = (max(1, width) * max(1, height)) / _REFERENCE_PIXELS
    per_step = _SECONDS_PER_STEP_PER_FRAME_REFERENCE * pixel_ratio * max(1, num_frames)
    return _MODEL_LOAD_OVERHEAD_S + per_step * max(1, steps) + _DECODE_OVERHEAD_S


def plan_message(params: VideoParams, seconds: float) -> str:
    """Polish sentence combining the request shape and the rough time estimate."""
    minutes = seconds / 60.0
    duration_s = params.num_frames / max(1, params.fps)
    when = f"~{minutes:.0f} min" if minutes >= 1.5 else f"~{seconds:.0f} s"
    return (
        f"{params.num_frames} klatek ({duration_s:.1f} s materiału przy {params.fps} kl/s), "
        f"{params.width}x{params.height}, {params.steps} kroków — szacunkowo {when} liczenia. "
        "To bardzo przybliżona wartość, nikt jeszcze nie zmierzył tego na tym sprzęcie."
    )


# --------------------------------------------------------------------------- #
# Which downloaded model is this?
# --------------------------------------------------------------------------- #

#: Repos this runtime has actually verified against the diffusers GGUF-loading
#: pattern documented for LTX-Video (see docs/VIDEO.md): small (~2GB Q8_0),
#: 2B-parameter checkpoints of the base ``Lightricks/LTX-Video`` model.
#: Only checkpoints whose **version matches the base pipeline** they are loaded
#: into. This is not a formality: a 0.9.6 transformer inside the 0.9.0-era
#: ``Lightricks/LTX-Video`` pipeline runs to completion, reports no error and
#: writes a valid H.264 file in which every frame is coloured noise. Measured
#: here across four runs (two quantisations, two compute dtypes, three
#: resolutions) before the version turned out to be the answer: with
#: ``city96/LTX-Video-gguf`` (0.9.0) and nothing else changed, the same prompt
#: produced a recognisable banana on a wooden table in 87 s.
#:
#: Lightricks does not publish a diffusers-format base repository for 0.9.6, so
#: those checkpoints stay out until one exists (they are reported as
#: ``"ltx-other"`` with a message, not silently run).
KNOWN_LTX_2B_REPOS: dict[str, str] = {
    "city96/ltx-video-gguf": "Lightricks/LTX-Video",
    "calcuis/ltxv-gguf": "Lightricks/LTX-Video",
}
#: A checkpoint bigger than this cannot be the 2B model no matter what its name
#: says — a defensive check against a mislabeled or renamed file. The ceiling
#: has to clear the *unquantised* weights: two billion parameters at bf16 are
#: about 4 GB, and the first cut of this constant (3,5 GB) rejected the model's
#: own BF16 file while accepting every quantisation of it. The next line up
#: (13B) starts around 7 GB even heavily quantised, so 5 GB still separates them.
_MAX_2B_SIZE_BYTES = 5_000_000_000

#: Name fragment that identifies the LTX *family* at all (broad on purpose —
#: catalogue names for this family range from "LTX Video" to "LTXv" to a bare
#: "LTX 2.3"; ``_LTX_OTHER_LINE_MARKERS`` and the size check below are what
#: actually separate the verified 2B checkpoint from everything else called
#: LTX, so this only decides "family recognised" vs. "not LTX at all").
_LTX_FAMILY_MARKER = "ltx"
#: Name fragments that clearly mean a *different* LTX line (the 13B+ LTX-2
#: family, NSFW community finetunes) or a 2B revision this pipeline cannot
#: decode — never the 2B target, however close the name looks.
#:
#: ``0.9.6``/``0.9.7`` are here for the same reason the map above is short: the
#: only diffusers base repository is the 0.9.0-era ``Lightricks/LTX-Video``, and
#: feeding it a newer transformer produces a valid video file full of noise
#: rather than an error. Better to say "nieobsługiwany wariant" than to hand
#: somebody a clip of static after a minute and a half of computing.
_LTX_OTHER_LINE_MARKERS = ("ltx-2", "ltx2", "ltx 2", "2.3", "2.5", "13b", "22b",
                          "10eros", "uncensored", "0.9.6", "0.9.7", "0.9.8")
#: Other catalogue "wideo" families LocalCode recognises by name but cannot
#: run yet (see scripts/build_catalog.py's own name heuristic for the same
#: vocabulary — kept in sync by hand, not imported, since hub/** is a
#: different owner's file).
_OTHER_VIDEO_MARKERS = ("wan2", "wan-video", "wan video", "hunyuanvideo", "hunyuan video",
                        "hunyuan_video", "cogvideo", "mochi", "minimax", "animatediff",
                        "cosmos", "longcat")


def _haystack(model: Any) -> str:
    parts = [str(getattr(model, "repo_id", "") or ""), str(getattr(model, "display_name", "") or ""),
              str(getattr(model, "key", "") or ""), str(getattr(model, "path", "") or "")]
    return " ".join(parts).lower()


def is_distilled(model: Any) -> bool:
    """True for a checkpoint whose name marks it as distilled.

    Distilled LTX models are trained for few steps and no guidance; the
    difference is not cosmetic (see :data:`DISTILLED_GUIDANCE`).
    """
    return "distill" in _haystack(model)


def default_guidance_for(model: Any) -> float:
    """The guidance value to use when the caller did not choose one."""
    return DISTILLED_GUIDANCE if is_distilled(model) else DEFAULT_GUIDANCE


def classify_video_model(model: Any) -> tuple[str, str | None]:
    """What kind of video model ``model`` looks like.

    Returns ``(kind, base_repo)``:

    * ``"ltx-2b"`` — a checkpoint this runtime knows how to load;
      ``base_repo`` is the ``diffusers`` repo to pull the rest of the
      pipeline from (``Lightricks/LTX-Video``).
    * ``"ltx-other"`` — clearly LTX, but not the verified 2B checkpoint
      (wrong size, or the name says LTX-2/2.3/2.5/13B/22B/a finetune).
    * ``"other-video"`` — a different video family entirely (Wan,
      HunyuanVideo, MiniMax, …).
    * ``"unknown"`` — nothing about the name or size suggests a video model
      at all (hub/library.py does not tag video capabilities today — see
      docs/VIDEO.md — so this is a best-effort name/size guess, the same way
      ``scripts/build_catalog.py`` recognises a video repo by name).

    ``base_repo`` is only set for ``"ltx-2b"``.
    """
    repo_id = str(getattr(model, "repo_id", "") or "").strip().lower()
    known_base = KNOWN_LTX_2B_REPOS.get(repo_id)
    size_bytes = int(getattr(model, "size_bytes", 0) or 0)
    haystack = _haystack(model)
    if known_base and (size_bytes <= 0 or size_bytes <= _MAX_2B_SIZE_BYTES):
        return "ltx-2b", known_base
    looks_ltx = _LTX_FAMILY_MARKER in haystack
    if looks_ltx:
        other_line = any(marker in haystack for marker in _LTX_OTHER_LINE_MARKERS)
        too_big = size_bytes > _MAX_2B_SIZE_BYTES
        if not other_line and not too_big:
            # Looks like the right family and the right size, but is not one
            # of the exact repos we have actually checked against the
            # diffusers loading pattern — treated as supported, since the
            # signal is strong (name + size both match the 2B checkpoint).
            return "ltx-2b", "Lightricks/LTX-Video"
        return "ltx-other", None
    if any(marker in haystack for marker in _OTHER_VIDEO_MARKERS):
        return "other-video", None
    return "unknown", None
