"""Speech: turning recordings into text, and text into a voice.

Two jobs that look like one shelf in the store and are nothing alike inside:

* **transcription** (``stt``) runs an encoder-decoder over audio — on Apple
  Silicon that is ``mlx-whisper`` (Whisper, Parakeet-style models),
* **synthesis** (``tts``) runs a much smaller model over text and writes a wave
  file — ``mlx-audio`` (Kokoro and friends).

Both are one-shot jobs of a few seconds, so unlike the language runtime there is
no resident process and no slot bookkeeping: a model is loaded, used and
released inside one call, in a worker thread.  That keeps the whole module
small enough to read, and keeps memory free for the models that *do* need to
stay resident.

The optional dependencies live in the ``speech`` extra; without them every
entry point explains what to install instead of failing with an ImportError.
"""

from localcode.speech.base import (
    SpeechError,
    Transcript,
    TranscriptSegment,
    Voice,
    VoiceResult,
)
from localcode.speech.runtime import SpeechRuntime

__all__ = [
    "SpeechError",
    "SpeechRuntime",
    "Transcript",
    "TranscriptSegment",
    "Voice",
    "VoiceResult",
]
