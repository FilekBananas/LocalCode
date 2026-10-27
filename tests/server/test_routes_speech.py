"""The speech endpoints (``localcode/server/routes_speech.py``).

The runtime is replaced with a stub, because what these tests are about is the
HTTP contract: which fields are required, what a bad value answers, and that a
speech failure keeps its own status code instead of collapsing into a 500.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from aiohttp.test_utils import TestClient

from localcode.speech import SpeechError
from localcode.speech.base import Transcript, TranscriptSegment, Voice, VoiceResult


class StubSpeech:
    def __init__(self, *, error: SpeechError | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def transcribe(self, path: str, **kwargs: Any) -> Transcript:
        self.calls.append(("transcribe", {"path": path, **kwargs}))
        if self.error:
            raise self.error
        return Transcript(text="Cześć.", language="pl", duration_s=1.5,
                          segments=[TranscriptSegment(0.0, 1.5, "Cześć.")],
                          model_key="mlx-community/whisper-small", elapsed_ms=120)

    async def speak(self, text: str, **kwargs: Any) -> VoiceResult:
        self.calls.append(("speak", {"text": text, **kwargs}))
        if self.error:
            raise self.error
        return VoiceResult(path="/tmp/mowa.wav", text=text, voice=kwargs.get("voice") or "af_heart",
                           duration_s=1.1, model_key="mlx-community/Kokoro-82M", elapsed_ms=90,
                           size_bytes=1234)

    def voices(self, key: str | None = None) -> list[Voice]:
        self.calls.append(("voices", {"key": key}))
        if self.error:
            raise self.error
        return [Voice("af_heart", "Heart", "en-US")]


@pytest.fixture
def speech(core) -> StubSpeech:
    stub = StubSpeech()
    core._speech = stub          # noqa: SLF001 - the route's own lazy attribute
    return stub


async def test_transcribe_returns_the_text_and_the_segments(client: TestClient,
                                                            speech: StubSpeech) -> None:
    response = await client.post("/api/speech/transcribe", json={"path": "/tmp/a.wav"})
    assert response.status == 200
    body = await response.json()
    assert body["text"] == "Cześć."
    assert body["language"] == "pl"
    assert body["segments"] == [{"start": 0.0, "end": 1.5, "text": "Cześć."}]
    assert body["modelKey"].endswith("whisper-small")
    assert "srt" not in body


async def test_transcribe_can_return_subtitles(client: TestClient, speech: StubSpeech) -> None:
    response = await client.post("/api/speech/transcribe",
                                 json={"path": "/tmp/a.wav", "srt": True})
    body = await response.json()
    assert body["srt"].startswith("1\n00:00:00,000 --> 00:00:01,500")


async def test_transcribe_needs_a_path(client: TestClient, speech: StubSpeech) -> None:
    response = await client.post("/api/speech/transcribe", json={})
    assert response.status == 400
    assert (await response.json())["code"] == "missing_path"


async def test_a_speech_error_keeps_its_status(client: TestClient, core) -> None:
    core._speech = StubSpeech(  # noqa: SLF001
        error=SpeechError("Nie znaleziono pliku.", "not_found", 404))
    response = await client.post("/api/speech/transcribe", json={"path": "/tmp/nie-ma.wav"})
    assert response.status == 404
    assert (await response.json())["code"] == "not_found"


async def test_speak_writes_a_file_and_says_where(client: TestClient, speech: StubSpeech) -> None:
    response = await client.post("/api/speech/speak",
                                 json={"text": "Dzień dobry", "voice": "af_bella"})
    assert response.status == 201
    body = await response.json()
    assert body["path"].endswith(".wav")
    assert body["voice"] == "af_bella"
    assert speech.calls[0][1]["text"] == "Dzień dobry"


@pytest.mark.parametrize("speed", [0.1, 5, "szybko"])
async def test_speak_rejects_an_impossible_speed(client: TestClient, speech: StubSpeech,
                                                 speed: Any) -> None:
    response = await client.post("/api/speech/speak", json={"text": "test", "speed": speed})
    assert response.status == 400


async def test_the_voice_list_is_served(client: TestClient, speech: StubSpeech) -> None:
    response = await client.get("/api/speech/voices")
    assert response.status == 200
    assert (await response.json())[0]["id"] == "af_heart"
