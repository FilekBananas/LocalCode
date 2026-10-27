"""``localcode/video/base.py``: parameter validation, the time estimate and model
classification — no subprocess, no ``torch``/``diffusers``, all pure Python.

This is the layer every other test in this directory (CLI, routes, runtime)
exercises only indirectly through a fake; here the real
:meth:`~localcode.video.base.VideoParams.normalized` and
:func:`~localcode.video.base.classify_video_model` run, in particular the one
invariant the module docstring calls out explicitly: ``MAX_FRAMES_HARD`` is
never bypassed, not even by the cost estimate's own ``enforce_frame_cap=False``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from localcode.video.base import (
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_FRAMES,
    DEFAULT_STEPS,
    DEFAULT_WIDTH,
    MAX_DIMENSION,
    MAX_FRAMES_DEFAULT,
    MAX_FRAMES_HARD,
    MAX_STEPS,
    MIN_DIMENSION,
    MIN_FRAMES,
    VideoError,
    VideoParams,
    classify_video_model,
    estimate_seconds,
    new_video_id,
    plan_message,
)

# --------------------------------------------------------------------------- #
# VideoParams.normalized() — prompt
# --------------------------------------------------------------------------- #


def test_empty_prompt_is_rejected_by_default() -> None:
    with pytest.raises(VideoError) as excinfo:
        VideoParams(prompt="   ").normalized()
    assert excinfo.value.code == "empty_prompt"
    assert "opis materiału wideo" in str(excinfo.value)


def test_empty_prompt_is_allowed_when_not_required() -> None:
    """The cost estimate runs before the user has necessarily typed a prompt."""
    params = VideoParams(prompt="").normalized(require_prompt=False)
    assert params.prompt == ""


def test_prompt_is_stripped() -> None:
    params = VideoParams(prompt="  kot w kapeluszu  ").normalized()
    assert params.prompt == "kot w kapeluszu"


def test_blank_negative_prompt_falls_back_to_the_default() -> None:
    params = VideoParams(prompt="kot", negative_prompt="   ").normalized()
    assert params.negative_prompt == DEFAULT_NEGATIVE_PROMPT


# --------------------------------------------------------------------------- #
# VideoParams.normalized() — size / frame / step rounding
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value, expected", [
    (0, MIN_DIMENSION), (10_000, MAX_DIMENSION),
    (700, 704),                          # nearest multiple of 32
    (DEFAULT_WIDTH, DEFAULT_WIDTH),
])
def test_width_is_snapped_to_the_32_pixel_grid(value: int, expected: int) -> None:
    params = VideoParams(prompt="kot", width=value).normalized()
    assert params.width == expected
    assert params.width % 32 == 0


@pytest.mark.parametrize("value, expected", [
    (1, MIN_FRAMES),                     # clamped up to the minimum (0 means "unset", see below)
    (42, 41),                            # nearest 8k+1
])
def test_frame_count_is_snapped_to_the_8k_plus_1_grid(value: int, expected: int) -> None:
    """Both values stay under ``MAX_FRAMES_DEFAULT`` — the cap itself is
    covered separately below, since anything past it raises instead of
    rounding (see ``test_hard_cap_is_never_lifted_even_with_force``)."""
    params = VideoParams(prompt="kot", num_frames=value).normalized()
    assert params.num_frames == expected
    assert (params.num_frames - 1) % 8 == 0


def test_a_zero_frame_count_means_unset_not_the_minimum() -> None:
    """``num_frames`` (like ``fps``/``steps``) is read with ``... or DEFAULT``,
    so ``0`` falls back to :data:`DEFAULT_NUM_FRAMES` rather than clamping to
    :data:`MIN_FRAMES` — a caller who wants the floor must ask for it with a
    positive value below it (``1``, covered above), not with ``0``."""
    assert VideoParams(prompt="kot", num_frames=0).normalized().num_frames == DEFAULT_NUM_FRAMES


def test_steps_are_clamped_to_the_maximum() -> None:
    params = VideoParams(prompt="kot", steps=10_000).normalized()
    assert params.steps == MAX_STEPS


def test_fps_is_clamped_between_1_and_60() -> None:
    # fps=0 is "unset" (falls back to DEFAULT_FPS, like num_frames above) —
    # a negative value is what actually exercises the lower clamp.
    assert VideoParams(prompt="kot", fps=-5).normalized().fps == 1
    assert VideoParams(prompt="kot", fps=999).normalized().fps == 60
    assert VideoParams(prompt="kot", fps=0).normalized().fps == DEFAULT_FPS


def test_seed_is_random_when_not_given_and_stable_when_given() -> None:
    a = VideoParams(prompt="kot").normalized()
    b = VideoParams(prompt="kot").normalized()
    assert a.seed is not None and b.seed is not None
    # not asserting a != b (a random draw could collide) — just that a seed is
    # always produced, and that an explicit one is preserved:
    assert VideoParams(prompt="kot", seed=42).normalized().seed == 42


# --------------------------------------------------------------------------- #
# VideoParams.normalized() — the frame cap (soft, then absolute)
# --------------------------------------------------------------------------- #


def test_default_frame_cap_is_enforced_without_force() -> None:
    with pytest.raises(VideoError) as excinfo:
        VideoParams(prompt="kot", num_frames=MAX_FRAMES_DEFAULT + 8).normalized()
    assert excinfo.value.code == "frame_cap_exceeded"
    assert str(MAX_FRAMES_DEFAULT) in str(excinfo.value)


def test_force_lifts_the_default_cap_up_to_the_hard_cap() -> None:
    params = VideoParams(prompt="kot", num_frames=MAX_FRAMES_DEFAULT + 8, force=True).normalized()
    assert params.num_frames > MAX_FRAMES_DEFAULT


def test_hard_cap_is_never_lifted_even_with_force() -> None:
    with pytest.raises(VideoError) as excinfo:
        VideoParams(prompt="kot", num_frames=MAX_FRAMES_HARD + 100, force=True).normalized()
    assert excinfo.value.code == "frame_hard_cap"


def test_hard_cap_survives_enforce_frame_cap_false_too() -> None:
    """The cost estimate (``enforce_frame_cap=False``) skips the *soft* cap so it
    can quote a price for an over-budget request — but never the hard one:
    nothing about this runtime has been measured past it, so even a bare
    estimate refuses to guess (see the module docstring above the constant)."""
    params = VideoParams(prompt="", num_frames=MAX_FRAMES_DEFAULT + 8)
    # under the soft cap's own default requirement this would already raise;
    # confirm enforce_frame_cap=False actually lifts it here:
    lifted = params.normalized(require_prompt=False, enforce_frame_cap=False)
    assert lifted.num_frames > MAX_FRAMES_DEFAULT

    with pytest.raises(VideoError) as excinfo:
        VideoParams(prompt="", num_frames=MAX_FRAMES_HARD + 100).normalized(
            require_prompt=False, enforce_frame_cap=False)
    assert excinfo.value.code == "frame_hard_cap"


def test_frame_cap_message_mentions_the_force_flag() -> None:
    with pytest.raises(VideoError) as excinfo:
        VideoParams(prompt="kot", num_frames=200).normalized()
    assert "--force" in str(excinfo.value) or "force" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# to_payload()
# --------------------------------------------------------------------------- #


def test_to_payload_is_the_worker_wire_shape() -> None:
    params = VideoParams(prompt="kot", width=512, height=384, num_frames=41, fps=24,
                         steps=10, guidance=4.5, seed=7).normalized()
    payload = params.to_payload()
    assert payload == {
        "prompt": "kot", "negativePrompt": params.negative_prompt,
        "width": 512, "height": 384, "numFrames": 41, "fps": 24, "steps": 10,
        "guidance": 4.5, "seed": 7,
    }


# --------------------------------------------------------------------------- #
# estimate_seconds() / plan_message()
# --------------------------------------------------------------------------- #


def test_estimate_scales_with_steps_and_frames() -> None:
    baseline = estimate_seconds(steps=DEFAULT_STEPS, num_frames=DEFAULT_NUM_FRAMES)
    more_steps = estimate_seconds(steps=DEFAULT_STEPS * 2, num_frames=DEFAULT_NUM_FRAMES)
    more_frames = estimate_seconds(steps=DEFAULT_STEPS, num_frames=DEFAULT_NUM_FRAMES * 2)
    assert more_steps > baseline
    assert more_frames > baseline


def test_estimate_scales_with_pixel_count() -> None:
    small = estimate_seconds(width=256, height=256)
    big = estimate_seconds(width=768, height=768)
    assert big > small


def test_estimate_never_goes_below_the_fixed_overheads() -> None:
    """Even a degenerate (1-step, tiny) request pays model load + decode."""
    from localcode.video.base import _DECODE_OVERHEAD_S, _MODEL_LOAD_OVERHEAD_S

    seconds = estimate_seconds(width=1, height=1, num_frames=1, steps=1)
    assert seconds >= _MODEL_LOAD_OVERHEAD_S + _DECODE_OVERHEAD_S


@pytest.mark.parametrize("width, height, frames, steps, measured", [
    (512, 320, 25, 8, 73.9),
    (704, 480, 25, 8, 86.7),
])
def test_the_estimate_reproduces_the_runs_it_was_calibrated_on(
        width: int, height: int, frames: int, steps: int, measured: float) -> None:
    """Two real generations on an M-series Mac with 32 GB, weights already cached.

    The constants were solved from exactly these two points, so this test is what
    keeps somebody from "tidying" them back into the guess they replaced — which
    promised four minutes for a job that takes a minute and a half.
    """
    seconds = estimate_seconds(width=width, height=height, num_frames=frames, steps=steps)
    assert abs(seconds - measured) < 1.0, f"{seconds:.1f} s vs zmierzone {measured} s"


def test_plan_message_is_polish_and_mentions_the_shape() -> None:
    params = VideoParams(prompt="kot").normalized()
    message = plan_message(params, estimate_seconds())
    assert "klatek" in message and "kroków" in message
    assert str(params.width) in message and str(params.height) in message


def test_plan_message_switches_between_seconds_and_minutes() -> None:
    params = VideoParams(prompt="kot").normalized()
    short = plan_message(params, 10.0)                # well under the 1.5 min threshold
    long = plan_message(params, 600.0)                 # 10 minutes
    assert "~10 s" in short and "min" not in short
    assert "~10 min" in long


# --------------------------------------------------------------------------- #
# new_video_id()
# --------------------------------------------------------------------------- #


def test_video_ids_are_unique_and_sortable_by_time() -> None:
    first = new_video_id()
    second = new_video_id()
    assert first != second
    assert first.split("-")[0].isdigit()


# --------------------------------------------------------------------------- #
# classify_video_model()
# --------------------------------------------------------------------------- #


def _model(**fields: object) -> SimpleNamespace:
    defaults = {"repo_id": "", "display_name": "", "key": "", "path": "", "size_bytes": 0}
    defaults.update(fields)
    return SimpleNamespace(**defaults)


def test_a_known_repo_is_ltx_2b() -> None:
    kind, base_repo = classify_video_model(_model(repo_id="city96/ltx-video-gguf"))
    assert kind == "ltx-2b"
    assert base_repo == "Lightricks/LTX-Video"


def test_known_repo_matching_is_case_insensitive() -> None:
    kind, _ = classify_video_model(_model(repo_id="City96/LTX-Video-GGUF"))
    assert kind == "ltx-2b"


def test_an_unlisted_but_right_shaped_ltx_repo_is_still_ltx_2b() -> None:
    """Strong signal (name says LTX, size fits) even without an exact repo match."""
    kind, base_repo = classify_video_model(
        _model(repo_id="someone-else/ltx-video-q8", size_bytes=2_000_000_000))
    assert kind == "ltx-2b"
    assert base_repo == "Lightricks/LTX-Video"


def test_an_oversized_ltx_named_checkpoint_is_ltx_other() -> None:
    """Name says LTX, but the file is far bigger than any known 2B quant."""
    kind, base_repo = classify_video_model(
        _model(repo_id="someone-else/ltx-video-fp16", size_bytes=10_000_000_000))
    assert kind == "ltx-other"
    assert base_repo is None


def test_the_ltx_2_line_is_ltx_other_not_ltx_2b() -> None:
    kind, _ = classify_video_model(_model(display_name="LTX-2 13B"))
    assert kind == "ltx-other"


def test_a_known_repo_oversized_beyond_the_2b_ceiling_is_ltx_other() -> None:
    """A mislabeled/renamed file: right repo id, wrong size — never trusted blindly."""
    kind, _ = classify_video_model(
        _model(repo_id="city96/ltx-video-gguf", size_bytes=10_000_000_000))
    assert kind == "ltx-other"


@pytest.mark.parametrize("name", ["Wan2.1 T2V 14B", "HunyuanVideo", "MiniMax Video-01",
                                  "CogVideoX-5B", "AnimateDiff"])
def test_other_known_video_families_are_recognised_but_unsupported(name: str) -> None:
    kind, base_repo = classify_video_model(_model(display_name=name))
    assert kind == "other-video"
    assert base_repo is None


def test_an_unrelated_model_is_unknown() -> None:
    kind, base_repo = classify_video_model(
        _model(repo_id="unsloth/Qwen3-0.6B-GGUF", display_name="Qwen3 0.6B"))
    assert kind == "unknown"
    assert base_repo is None


# --------------------------------------------------------------------------- #
# A distilled checkpoint wants no guidance (measured, not guessed)
# --------------------------------------------------------------------------- #


class _NamedModel:
    def __init__(self, name: str) -> None:
        self.repo_id = name
        self.display_name = name
        self.key = name
        self.path = f"/models/{name}"
        self.size_bytes = 2_040_000_000


@pytest.mark.parametrize("name, distilled", [
    ("city96/LTX-Video-0.9.6-distilled-gguf", True),
    ("calcuis/ltxv0.9.6-gguf", False),
    ("Lightricks/LTX-Video", False),
    ("wsbagnsv1/ltxv-13b-0.9.7-distilled-GGUF", True),
])
def test_a_distilled_checkpoint_is_recognised(name: str, distilled: bool) -> None:
    from localcode.video.base import is_distilled

    assert is_distilled(_NamedModel(name)) is distilled


def test_a_distilled_checkpoint_defaults_to_no_guidance() -> None:
    """Measured: guidance 3.0 on a distilled model returns coloured noise.

    The file it wrote was a valid 25-frame H.264 video and every frame was
    static — which is why this is a default and not a note in the docs.
    """
    from localcode.video.base import DEFAULT_GUIDANCE, DISTILLED_GUIDANCE, default_guidance_for

    assert default_guidance_for(_NamedModel("LTX-Video-0.9.6-distilled")) == DISTILLED_GUIDANCE
    assert default_guidance_for(_NamedModel("LTX-Video")) == DEFAULT_GUIDANCE
    assert DISTILLED_GUIDANCE == 1.0


def test_the_models_own_unquantised_file_is_still_the_2b_model() -> None:
    """Two billion parameters at bf16 are ~4 GB, so the ceiling must clear that.

    The first cut of ``_MAX_2B_SIZE_BYTES`` (3,5 GB) rejected the checkpoint's own
    BF16 file while accepting every quantisation of it — "wybierz najwyższą
    jakość" then produced a model the engine refused to run.
    """
    from localcode.video.base import classify_video_model

    class _Sized(_NamedModel):
        def __init__(self, name: str, size: int) -> None:
            super().__init__(name)
            self.repo_id = "city96/LTX-Video-gguf"
            self.size_bytes = size

    assert classify_video_model(_Sized("ltx-video-2b-v0.9-BF16", 3_600_000_000))[0] == "ltx-2b"
    assert classify_video_model(_Sized("ltx-video-2b-v0.9-Q4_K_M", 1_330_000_000))[0] == "ltx-2b"
    # ...and the next model up is still refused
    assert classify_video_model(_Sized("ltxv-13b-0.9.7-distilled-Q4_K_M", 7_540_000_000))[0] != "ltx-2b"


# --------------------------------------------------------------------------- #
# Only a checkpoint whose version matches the base pipeline
# --------------------------------------------------------------------------- #


class _Repo(_NamedModel):
    def __init__(self, repo: str, name: str, size: int = 1_330_000_000) -> None:
        super().__init__(name)
        self.repo_id = repo
        self.size_bytes = size


def test_the_version_matched_checkpoint_is_the_one_we_run() -> None:
    """0.9.0 GGUF + the 0.9.0-era base repo: this pair produced a real clip."""
    from localcode.video.base import classify_video_model

    kind, base = classify_video_model(_Repo("city96/LTX-Video-gguf", "ltx-video-2b-v0.9-Q4_K_M"))
    assert (kind, base) == ("ltx-2b", "Lightricks/LTX-Video")


@pytest.mark.parametrize("repo, name", [
    ("city96/LTX-Video-0.9.6-distilled-gguf", "ltxv-2b-0.9.6-distilled-04-25-Q4_K_M"),
    ("calcuis/ltxv0.9.6-gguf", "ltxv-0.9.6-Q8_0"),
])
def test_a_newer_revision_without_a_matching_pipeline_is_refused(repo: str, name: str) -> None:
    """It does not fail loudly — it writes a minute and a half of noise.

    With the 0.9.6 transformer in the 0.9.0 pipeline the run completes, reports
    no error and produces a valid H.264 file in which every frame is static.
    Refusing it up front is the only honest answer until Lightricks publishes a
    diffusers base repository for that revision.
    """
    from localcode.video.base import classify_video_model

    kind, base = classify_video_model(_Repo(repo, name))
    assert kind == "ltx-other"
    assert base is None
