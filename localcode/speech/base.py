"""Shapes and errors shared by transcription and synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class SpeechError(Exception):
    """A speech failure with a Polish, user-presentable message."""

    def __init__(self, message: str, code: str = "speech_error", status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


#: Audio containers the transcriber accepts (ffmpeg reads the rest, when present).
AUDIO_SUFFIXES = (".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm",
                  ".mp4", ".mov", ".mkv", ".aiff", ".aif", ".caf")


@dataclass
class TranscriptSegment:
    """One timed chunk of a transcript, as the subtitle writers need it."""

    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "text": self.text}


@dataclass
class Transcript:
    text: str
    language: str | None = None
    duration_s: float | None = None
    segments: list[TranscriptSegment] = field(default_factory=list)
    model_key: str = ""
    elapsed_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "durationSeconds": round(self.duration_s, 2) if self.duration_s else None,
            "segments": [s.to_dict() for s in self.segments],
            "modelKey": self.model_key,
            "elapsedMs": self.elapsed_ms,
        }

    def to_srt(self) -> str:
        """The transcript as a subtitle file."""
        def stamp(value: float) -> str:
            ms = int(round(value * 1000))
            hours, ms = divmod(ms, 3_600_000)
            minutes, ms = divmod(ms, 60_000)
            seconds, ms = divmod(ms, 1000)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"

        lines: list[str] = []
        for index, segment in enumerate(self.segments or [], 1):
            lines.append(str(index))
            lines.append(f"{stamp(segment.start)} --> {stamp(segment.end)}")
            lines.append(segment.text.strip())
            lines.append("")
        return "\n".join(lines).strip() + ("\n" if lines else "")


@dataclass
class Voice:
    """One voice a synthesis model offers."""

    id: str
    label: str = ""
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label or self.id, "language": self.language}


@dataclass
class VoiceResult:
    path: str
    text: str
    voice: str | None = None
    duration_s: float | None = None
    sample_rate: int = 24_000
    model_key: str = ""
    elapsed_ms: int = 0
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "text": self.text,
            "voice": self.voice,
            "durationSeconds": round(self.duration_s, 2) if self.duration_s else None,
            "sampleRate": self.sample_rate,
            "modelKey": self.model_key,
            "elapsedMs": self.elapsed_ms,
            "sizeBytes": self.size_bytes,
        }
