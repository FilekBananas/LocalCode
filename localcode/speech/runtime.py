"""Transcription and synthesis, one call at a time.

Both jobs here are short and self-contained, so there is no resident process
and no slot accounting: the backend loads the model, does the work and lets it
go, all inside one ``asyncio.to_thread`` call.  That is a deliberate difference
from :mod:`localcode.runtime` — a chat model has to stay warm between messages,
a transcription does not, and keeping speech out of the RAM budget means a
dictation never evicts the model somebody is talking to.

The backends are injected (``transcriber`` / ``synthesizer``) so the tests can
run the whole path without mlx-whisper or mlx-audio installed; in production
they are resolved lazily, and a missing package produces a Polish sentence
naming the extra to install rather than an ImportError.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from localcode.config import install_command
from localcode.speech.base import (
    AUDIO_SUFFIXES,
    SpeechError,
    Transcript,
    TranscriptSegment,
    Voice,
    VoiceResult,
)

log = logging.getLogger(__name__)

# Ścieżka interpretera, nie samo „pip” — patrz localcode.config.install_command.
INSTALL_HINT = ("Brakuje dodatku do mowy (mlx-whisper rozpoznaje, mlx-audio czyta na głos). "
                f"Zainstaluj: {install_command('speech')}")

def _audio_duration(raw: dict[str, Any], segments: "list[TranscriptSegment]") -> float | None:
    """How long the recording was, when the backend did not say.

    mlx-whisper reports ``duration`` only for some inputs, and the app shows the
    length next to the transcript — so fall back to where the last segment ends,
    which is the same number to within the length of a pause.
    """
    ends = [float(s.get("end") or 0.0) for s in (raw.get("segments") or ())]
    ends += [float(s.end) for s in segments]
    longest = max(ends, default=0.0)
    return round(longest, 2) or None


#: Kokoro's default voice; overridden per call or per model.
DEFAULT_VOICE = "af_heart"
DEFAULT_SAMPLE_RATE = 24_000


class SpeechRuntime:
    """Speech-to-text and text-to-speech for one machine."""

    def __init__(self, settings: Any, library: Any = None, *,
                 transcriber: Callable[..., dict[str, Any]] | None = None,
                 synthesizer: Callable[..., dict[str, Any]] | None = None) -> None:
        self.settings = settings
        self._library = library
        self._transcriber = transcriber
        self._synthesizer = synthesizer

    # plumbing ----------------------------------------------------------------
    @property
    def library(self) -> Any:
        if self._library is None:
            from localcode.hub.library import ModelLibrary

            self._library = ModelLibrary(self.settings)
        return self._library

    def _model(self, key: str | None, *, want: str) -> Any:
        """The model to use, by key or by capability."""
        models = list(self.library.list())
        if key:
            model = self.library.get(key) or self.library.resolve(key)
            if model is None:
                raise SpeechError(f"Nie znaleziono modelu „{key}”.", "model_not_found", 404)
            return model
        for model in models:
            if want in (model.capabilities or []):
                return model
        raise SpeechError(
            "Nie masz jeszcze modelu do " + ("rozpoznawania mowy" if want == "stt"
                                             else "czytania na głos")
            + ". Zainstaluj któryś w sklepie (kategoria „Mowa”).",
            "no_model", 400)

    @staticmethod
    def _audio_path(path: str | Path) -> Path:
        audio = Path(path).expanduser()
        if not audio.exists():
            raise SpeechError(f"Nie znaleziono pliku „{audio}”.", "not_found", 404)
        if audio.suffix.lower() not in AUDIO_SUFFIXES:
            raise SpeechError(
                f"Nie umiem czytać plików „{audio.suffix or '?'}”. "
                f"Obsługiwane: {', '.join(AUDIO_SUFFIXES[:8])}…", "unsupported_audio")
        return audio

    # transcription -----------------------------------------------------------
    async def transcribe(self, audio: str | Path, *, model_key: str | None = None,
                         language: str | None = None, segments: bool = True) -> Transcript:
        """Recording → text (plus timed segments, which is what subtitles need)."""
        path = self._audio_path(audio)
        model = self._model(model_key, want="stt")
        started = time.monotonic()
        backend = self._transcriber or _mlx_whisper_transcribe
        try:
            raw = await asyncio.to_thread(backend, str(path), model.path, language)
        except SpeechError:
            raise
        except ImportError as exc:
            raise SpeechError(INSTALL_HINT, "missing_extra", 501) from exc
        except Exception as exc:  # noqa: BLE001 - one bad recording must not kill the server
            raise SpeechError(f"Nie udało się rozpoznać mowy: {exc}", "transcribe_failed",
                              500) from exc
        parts = [
            TranscriptSegment(float(s.get("start") or 0.0), float(s.get("end") or 0.0),
                              str(s.get("text") or "").strip())
            for s in (raw.get("segments") or ())
        ] if segments else []
        return Transcript(
            text=str(raw.get("text") or "").strip(),
            language=raw.get("language") or language,
            duration_s=(float(raw.get("duration") or 0.0) or _audio_duration(raw, parts)),
            segments=parts,
            model_key=model.key,
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )

    # synthesis ---------------------------------------------------------------
    async def speak(self, text: str, *, model_key: str | None = None, voice: str | None = None,
                    speed: float = 1.0, output: str | Path | None = None) -> VoiceResult:
        """Text → a wave file on disk."""
        words = (text or "").strip()
        if not words:
            raise SpeechError("Nie ma czego przeczytać — tekst jest pusty.", "empty_text")
        model = self._model(model_key, want="tts")
        target = Path(output).expanduser() if output else self._default_output()
        target.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        backend = self._synthesizer or _mlx_audio_speak
        try:
            raw = await asyncio.to_thread(backend, words, model.path, str(target),
                                          voice or DEFAULT_VOICE, float(speed))
        except SpeechError:
            raise
        except ImportError as exc:
            raise SpeechError(INSTALL_HINT, "missing_extra", 501) from exc
        except Exception as exc:  # noqa: BLE001
            raise SpeechError(f"Nie udało się przeczytać tekstu: {exc}", "speak_failed",
                              500) from exc
        path = Path(str(raw.get("path") or target))
        return VoiceResult(
            path=str(path),
            text=words,
            voice=raw.get("voice") or voice or DEFAULT_VOICE,
            duration_s=float(raw.get("duration") or 0.0) or _wave_duration(path),
            sample_rate=int(raw.get("sampleRate") or DEFAULT_SAMPLE_RATE),
            model_key=model.key,
            elapsed_ms=int((time.monotonic() - started) * 1000),
            size_bytes=path.stat().st_size if path.exists() else 0,
        )

    def voices(self, model_key: str | None = None) -> list[Voice]:
        """Voices a synthesis model offers (Kokoro ships a fixed set)."""
        self._model(model_key, want="tts")
        return [Voice(v, label, lang) for v, label, lang in _KOKORO_VOICES]

    def _default_output(self) -> Path:
        from localcode.config import home

        stamp = time.strftime("%Y%m%d-%H%M%S")
        return home() / "data" / "speech" / f"mowa-{stamp}.wav"


#: The voices Kokoro ships with, in the order its documentation lists them.
_KOKORO_VOICES: Sequence[tuple[str, str, str]] = (
    ("af_heart", "Heart (kobiecy, US)", "en-US"),
    ("af_bella", "Bella (kobiecy, US)", "en-US"),
    ("af_nicole", "Nicole (kobiecy, US)", "en-US"),
    ("am_michael", "Michael (męski, US)", "en-US"),
    ("am_fenrir", "Fenrir (męski, US)", "en-US"),
    ("bf_emma", "Emma (kobiecy, UK)", "en-GB"),
    ("bm_george", "George (męski, UK)", "en-GB"),
)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #


def _mlx_whisper_transcribe(audio: str, model_path: str, language: str | None) -> dict[str, Any]:
    """Whisper-family transcription through ``mlx-whisper``."""
    import mlx_whisper  # noqa: PLC0415 - optional dependency, imported on use

    return mlx_whisper.transcribe(
        audio, path_or_hf_repo=model_path, language=language, word_timestamps=False)


def _wave_duration(path: Path) -> float | None:
    """How long a generated wave file plays, read from its own header.

    mlx-audio does not report it, and the app shows the length next to the
    player; the header has the two numbers needed and costs nothing to read.
    """
    import wave  # noqa: PLC0415 - only needed right here

    try:
        with wave.open(str(path), "rb") as fh:
            rate = fh.getframerate()
            frames = fh.getnframes()
    except (OSError, wave.Error):
        return None
    return round(frames / rate, 2) if rate and frames else None


def _mlx_audio_speak(text: str, model_path: str, output: str, voice: str,
                     speed: float) -> dict[str, Any]:
    """Text-to-speech through ``mlx-audio`` (Kokoro and compatible models).

    The keyword for the model changed between releases (``model_path`` →
    ``model``) and newer versions write nothing unless asked (``save=True``), so
    the call is assembled from the signature actually installed. Getting this
    wrong is quiet: ``generate_audio`` prints "Error loading model" and returns
    normally, which is why this function insists on seeing the file afterwards.
    """
    import inspect  # noqa: PLC0415 - only needed on this path

    from mlx_audio.tts.generate import generate_audio  # noqa: PLC0415 - optional dependency

    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    params = set(inspect.signature(generate_audio).parameters)
    kwargs: dict[str, Any] = {
        "text": text, "voice": voice, "speed": speed,
        "file_prefix": str(target.with_suffix("")), "audio_format": "wav",
        "join_audio": True, "verbose": False,
    }
    kwargs["model_path" if "model_path" in params else "model"] = model_path
    if "save" in params:
        kwargs["save"] = True            # newer releases return audio instead of writing it
    if "output_path" in params:
        kwargs["output_path"] = str(target.parent)
    generate_audio(**kwargs)
    # mlx-audio appends its own suffix when it splits long text; take whatever
    # it actually wrote next to the prefix we asked for.
    if not target.exists():
        produced = sorted(target.parent.glob(target.stem + "*.wav"))
        if produced:
            produced[0].replace(target)
    if not target.exists() or target.stat().st_size == 0:
        raise SpeechError(
            "Model nie wygenerował pliku dźwiękowego. Sprawdź, czy wybrany model to model "
            "do czytania na głos (Kokoro) i czy pakiet mlx-audio jest w aktualnej wersji.",
            "speak_failed", 500)
    return {"path": str(target), "voice": voice}
