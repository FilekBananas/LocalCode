"""Transcription and synthesis (``localcode.speech``).

The backends (mlx-whisper, mlx-audio) are injected, so the whole path — picking
a model, validating the input, shaping the answer, writing subtitles — is
covered without either package installed. What the real backends do with a
model is their business; what this module promises is everything around it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from localcode.config import Settings
from localcode.hub.types import LocalModel, ModelMeta
from localcode.speech import SpeechError, SpeechRuntime, Transcript


class FakeLibrary:
    def __init__(self, models: list[LocalModel]) -> None:
        self.models = models

    def list(self) -> list[LocalModel]:
        return list(self.models)

    def get(self, key: str) -> LocalModel | None:
        return next((m for m in self.models if m.key == key), None)

    def resolve(self, text: str) -> LocalModel | None:
        return next((m for m in self.models if text.lower() in m.key.lower()), None)


def model(key: str, *, caps: list[str], path: str = "/models/x") -> LocalModel:
    return LocalModel(key=key, format="mlx", path=path, display_name=key.split("/")[-1],
                      size_bytes=500_000_000, capabilities=caps, meta=ModelMeta())


STT = model("mlx-community/whisper-small", caps=["stt"])
TTS = model("mlx-community/Kokoro-82M", caps=["tts"])


def transcript_backend(**answer: Any):
    calls: list[tuple[Any, ...]] = []

    def backend(audio: str, model_path: str, language: str | None) -> dict[str, Any]:
        calls.append((audio, model_path, language))
        return {"text": "Cześć, to jest test.", "language": language or "pl", "duration": 3.5,
                "segments": [{"start": 0.0, "end": 1.8, "text": "Cześć,"},
                             {"start": 1.8, "end": 3.5, "text": "to jest test."}],
                **answer}

    backend.calls = calls  # type: ignore[attr-defined]
    return backend


def voice_backend(**answer: Any):
    calls: list[tuple[Any, ...]] = []

    def backend(text: str, model_path: str, output: str, voice: str, speed: float) -> dict[str, Any]:
        calls.append((text, model_path, output, voice, speed))
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"RIFF" + b"\0" * 100)
        return {"path": output, "voice": voice, "duration": 1.2, **answer}

    backend.calls = calls  # type: ignore[attr-defined]
    return backend


@pytest.fixture
def wav(tmp_path: Path) -> Path:
    path = tmp_path / "nagranie.wav"
    path.write_bytes(b"RIFF" + b"\0" * 1000)
    return path


def runtime(tmp_path: Path, *, models: list[LocalModel] | None = None, **backends: Any) -> SpeechRuntime:
    settings = Settings(models_dir=str(tmp_path / "models"))
    return SpeechRuntime(settings, FakeLibrary(models if models is not None else [STT, TTS]),
                         **backends)


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #


async def test_transcribing_a_recording(tmp_path: Path, wav: Path) -> None:
    backend = transcript_backend()
    result = await runtime(tmp_path, transcriber=backend).transcribe(wav)

    assert isinstance(result, Transcript)
    assert result.text == "Cześć, to jest test."
    assert result.language == "pl"
    assert result.model_key == STT.key
    assert [s.text for s in result.segments] == ["Cześć,", "to jest test."]
    assert backend.calls[0][0] == str(wav)


async def test_the_language_can_be_forced(tmp_path: Path, wav: Path) -> None:
    backend = transcript_backend()
    await runtime(tmp_path, transcriber=backend).transcribe(wav, language="en")
    assert backend.calls[0][2] == "en"


async def test_segments_can_be_skipped(tmp_path: Path, wav: Path) -> None:
    result = await runtime(tmp_path, transcriber=transcript_backend()).transcribe(
        wav, segments=False)
    assert result.segments == []
    assert result.text


async def test_a_transcript_becomes_subtitles(tmp_path: Path, wav: Path) -> None:
    result = await runtime(tmp_path, transcriber=transcript_backend()).transcribe(wav)
    srt = result.to_srt()
    assert srt.startswith("1\n00:00:00,000 --> 00:00:01,800\nCześć,")
    assert "2\n00:00:01,800 --> 00:00:03,500\nto jest test." in srt


async def test_a_missing_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(SpeechError) as excinfo:
        await runtime(tmp_path, transcriber=transcript_backend()).transcribe(tmp_path / "nie-ma.wav")
    assert excinfo.value.status == 404


async def test_a_file_that_is_not_audio_is_refused(tmp_path: Path) -> None:
    junk = tmp_path / "notatki.txt"
    junk.write_text("nie audio", encoding="utf-8")
    with pytest.raises(SpeechError) as excinfo:
        await runtime(tmp_path, transcriber=transcript_backend()).transcribe(junk)
    assert excinfo.value.code == "unsupported_audio"


async def test_without_a_speech_model_the_store_is_named(tmp_path: Path, wav: Path) -> None:
    with pytest.raises(SpeechError) as excinfo:
        await runtime(tmp_path, models=[TTS], transcriber=transcript_backend()).transcribe(wav)
    assert "sklepie" in str(excinfo.value)


async def test_a_missing_backend_names_the_extra(tmp_path: Path, wav: Path) -> None:
    def backend(*_: Any) -> dict[str, Any]:
        raise ImportError("No module named 'mlx_whisper'")

    with pytest.raises(SpeechError) as excinfo:
        await runtime(tmp_path, transcriber=backend).transcribe(wav)
    assert "localcode[speech]" in str(excinfo.value)
    assert excinfo.value.status == 501


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #


async def test_reading_text_aloud(tmp_path: Path) -> None:
    backend = voice_backend()
    target = tmp_path / "out" / "mowa.wav"
    result = await runtime(tmp_path, synthesizer=backend).speak(
        "Dzień dobry.", output=target, voice="af_bella", speed=1.2)

    assert Path(result.path) == target
    assert target.exists()
    assert result.voice == "af_bella"
    assert result.model_key == TTS.key
    assert result.size_bytes > 0
    assert backend.calls[0][3:] == ("af_bella", 1.2)


async def test_empty_text_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SpeechError) as excinfo:
        await runtime(tmp_path, synthesizer=voice_backend()).speak("   ")
    assert excinfo.value.code == "empty_text"


async def test_without_a_voice_model_the_store_is_named(tmp_path: Path) -> None:
    with pytest.raises(SpeechError) as excinfo:
        await runtime(tmp_path, models=[STT], synthesizer=voice_backend()).speak("Cześć")
    assert "Mowa" in str(excinfo.value)


async def test_a_named_model_is_honoured(tmp_path: Path, wav: Path) -> None:
    other = model("mlx-community/whisper-large", caps=["stt"], path="/models/large")
    backend = transcript_backend()
    await runtime(tmp_path, models=[STT, other], transcriber=backend).transcribe(
        wav, model_key="whisper-large")
    assert backend.calls[0][1] == "/models/large"


def test_the_voice_list_needs_a_model(tmp_path: Path) -> None:
    assert runtime(tmp_path).voices()
    with pytest.raises(SpeechError):
        runtime(tmp_path, models=[STT]).voices()


# --------------------------------------------------------------------------- #
# What the app shows next to a transcript and next to the player
# --------------------------------------------------------------------------- #


def test_the_length_of_a_recording_is_reported_even_when_the_backend_omits_it() -> None:
    """The app shows the length next to the transcript, so it must not be blank.

    mlx-whisper fills ``duration`` for some inputs and not others; the end of the
    last segment is the same number to within the length of a pause.
    """
    from localcode.speech.runtime import _audio_duration

    raw = {"segments": [{"start": 0.0, "end": 1.4}, {"start": 1.4, "end": 2.8}]}
    assert _audio_duration(raw, []) == 2.8


def test_a_recording_with_no_segments_at_all_reports_nothing() -> None:
    from localcode.speech.runtime import _audio_duration

    assert _audio_duration({}, []) is None
    assert _audio_duration({"segments": []}, []) is None


def test_the_length_of_generated_speech_comes_from_the_file(tmp_path) -> None:
    """mlx-audio does not report it and the app shows it next to the player."""
    import wave

    from localcode.speech.runtime import _wave_duration

    path = tmp_path / "mowa.wav"
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(24_000)
        fh.writeframes(b"\x00\x00" * 24_000)        # exactly one second

    assert _wave_duration(path) == 1.0


def test_a_missing_or_broken_file_reports_no_length(tmp_path) -> None:
    from localcode.speech.runtime import _wave_duration

    assert _wave_duration(tmp_path / "nie-ma.wav") is None
    broken = tmp_path / "zepsuty.wav"
    broken.write_bytes(b"to nie jest wave")
    assert _wave_duration(broken) is None


async def test_speaking_that_produces_no_file_is_an_error_not_a_success(tmp_path) -> None:
    """mlx-audio prints its import errors and returns normally.

    Reproduced with Kokoro and no ``misaki`` installed: the route answered 201
    with a path to a file that did not exist. A missing file has to be an error.
    """
    from localcode.speech.base import SpeechError
    from localcode.speech.runtime import _mlx_audio_speak

    fake = tmp_path / "cisza.wav"
    import sys
    import types

    module = types.ModuleType("mlx_audio.tts.generate")
    module.generate_audio = lambda **kwargs: None          # writes nothing, raises nothing
    pkg = types.ModuleType("mlx_audio")
    tts = types.ModuleType("mlx_audio.tts")
    sys.modules.update({"mlx_audio": pkg, "mlx_audio.tts": tts,
                        "mlx_audio.tts.generate": module})
    try:
        with pytest.raises(SpeechError) as exc:
            _mlx_audio_speak("tekst", "/model", str(fake), "af_heart", 1.0)
        assert "nie wygenerował pliku" in str(exc.value)
    finally:
        for name in ("mlx_audio.tts.generate", "mlx_audio.tts", "mlx_audio"):
            sys.modules.pop(name, None)
