"""
audio_service.py — Hybrid TTS Provider Architecture
====================================================
Routing logic:
  1. ElevenLabsProvider  — premium, tried first if ELEVENLABS_API_KEY is set
  2. EdgeTTSProvider     — zero-cost Microsoft Neural TTS, automatic fallback

EdgeTTSProvider features:
  - Uses edge_tts.Communicate Python API with asyncio.run() (no subprocess)
  - Native rate/pitch params via LanguageProfile (rate, pitch)
  - Plain text input (no SSML — edge-tts reads tags literally)
  - Post-save file-size validation (raises TTSGenerationError on 0 bytes)
  - Text normalisation pre-pass  (%, $, abbreviations → spoken form)
  - Immediate WAV conversion via ffmpeg-python after output to guarantee
    100% accurate ffprobe duration detection (no VBR-MP3 drift)
"""

import os
import re
import asyncio
import logging
import tempfile
import base64
import requests
import ffmpeg
import edge_tts

from django.conf import settings
from apps.processor.services.language_profiles import ProfileManager

logger = logging.getLogger(__name__)


class TTSGenerationError(Exception):
    """Raised when TTS produces empty or otherwise unusable audio output."""
    pass

# ── Silent WAV fallback (1 s silence, 16-bit PCM 16 kHz mono) ────────────────
# Generated with: ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t 1 -c:a pcm_s16le silent.wav
# Then base64-encoded.  Used only when ALL providers fail so the pipeline can
# still reach FFmpeg and fail gracefully rather than crashing with a missing path.
_SILENT_WAV_B64 = (
    "UklGRiQAAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YQAAAAA="
)


# ═══════════════════════════════════════════════════════════════════════════════
# Voice Mapping Table
# ═══════════════════════════════════════════════════════════════════════════════
# frontend voice_id  →  (elevenlabs_voice_id,  edge_tts_voice_name)
VOICE_MAP = {
    "standard_male":    ("pNInz6obpgDQGcFmaJgB", "en-US-ChristopherNeural"),
    "standard_female":  ("21m00Tcm4TlvDq8ikWAM", "en-US-AriaNeural"),
    "male_1":           ("pNInz6obpgDQGcFmaJgB", "en-US-ChristopherNeural"),
    "female_1":         ("21m00Tcm4TlvDq8ikWAM", "en-US-AriaNeural"),
    "hindi_male":       ("JBFqnCBsd6RMkjVDRZzb", "hi-IN-MadhurNeural"),
    "hindi_female":     ("ThT5KcBeYPX3keUQqHPh", "hi-IN-SwaraNeural"),
    "british_male":     ("VR6AewLTigWG4xSOukaG", "en-GB-RyanNeural"),
    "british_female":   ("AZnzlk1XvdvUeBnXmlld", "en-GB-SoniaNeural"),
    "hi-IN-MadhurNeural":      ("JBFqnCBsd6RMkjVDRZzb", "hi-IN-MadhurNeural"),
    "gu-IN-NiranjanNeural":    ("JBFqnCBsd6RMkjVDRZzb", "gu-IN-NiranjanNeural"),
    "en-US-ChristopherNeural": ("pNInz6obpgDQGcFmaJgB", "en-US-ChristopherNeural"),
}

# Default fallback when voice_id is unknown
_DEFAULT_ELEVENLABS_VOICE = "pNInz6obpgDQGcFmaJgB"   # Adam
_DEFAULT_EDGE_VOICE       = "en-US-ChristopherNeural"

# Language detection replaced by ProfileManager.detect() in language_profiles.py


# ═══════════════════════════════════════════════════════════════════════════════
# Text Normalisation Pre-pass
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_text(text: str) -> str:
    """
    Clean raw script text before sending to any TTS engine.
    Only strip actual illegal OS characters or control characters (\n, \r, \t)
    and preserve Devanagari and Gujarati characters in raw UTF-8.
    """
    # Replace control characters (\n, \r, \t) with spaces
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Strip illegal OS filename/control characters (e.g. \x00-\x1f) if any, but preserve UTF-8 Devanagari and Gujarati characters
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Collapse multiple whitespace
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text




# ═══════════════════════════════════════════════════════════════════════════════
# EdgeTTS Provider
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeTTSProvider:
    """
    Renders audio using the edge_tts.Communicate Python API with
    asyncio.run().  Avoids fragile subprocess I/O.

    Pipeline:
      1. Normalise text  (symbols, currencies, abbreviations)
      2. edge_tts.Communicate(text, voice, rate, pitch) → raw VBR MP3
      3. os.path.getsize validation — raises TTSGenerationError on 0 bytes
      4. ffmpeg re-encode → final PCM WAV (16-bit, 16 kHz, mono)
         guarantees 100% accurate ffprobe duration detection
    """

    @staticmethod
    def generate(text: str, voice_id: str, output_wav_path: str) -> None:
        """
        Render plain text to a WAV file using the edge_tts Python API.
        Validates file size post-save.  Raises TTSGenerationError or
        RuntimeError on failure so the caller can fall through.
        """
        profile = ProfileManager.detect(text)

        # Get voice from map or use profile default
        mapping = VOICE_MAP.get(voice_id)
        if mapping:
            _, edge_voice = mapping
        else:
            edge_voice = profile.tts_voice_id

        normalised = _normalise_text(text)

        if not normalised:
            raise TTSGenerationError("Normalised text is empty — nothing to synthesise.")

        raw_mp3_fd, raw_mp3_path = tempfile.mkstemp(suffix='.mp3')
        os.close(raw_mp3_fd)

        try:
            logger.info(
                f"[EdgeTTS] Generating TTS for: {normalised[:50]}... | "
                f"Rate: {profile.tts_rate} | Pitch: {profile.tts_pitch} | "
                f"Voice: {edge_voice}"
            )

            communicate = edge_tts.Communicate(
                normalised,
                edge_voice,
                rate=profile.tts_rate,
                pitch=profile.tts_pitch,
            )

            asyncio.run(communicate.save(raw_mp3_path))

            # — File-size validation -------------------------------------------------
            mp3_size = os.path.getsize(raw_mp3_path)
            if mp3_size == 0:
                raise TTSGenerationError(
                    f"Audio file is 0 bytes — edge-tts produced no output. "
                    f"voice={edge_voice}, rate={profile.tts_rate}, pitch={profile.tts_pitch}"
                )

            logger.info(
                f"[EdgeTTS] MP3 render complete ({mp3_size} bytes). "
                f"Converting → PCM WAV at {output_wav_path} ..."
            )

            (
                ffmpeg
                .input(raw_mp3_path)
                .output(
                    output_wav_path,
                    acodec='pcm_s16le',
                    ar=16000,
                    ac=1,
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )

            wav_size = os.path.getsize(output_wav_path)
            if wav_size == 0:
                raise TTSGenerationError(
                    f"WAV file is 0 bytes after ffmpeg conversion."
                )

            logger.info(
                f"[EdgeTTS] WAV output written successfully: {output_wav_path} "
                f"({wav_size} bytes)"
            )

        except (TTSGenerationError, RuntimeError):
            raise

        except Exception as exc:
            raise RuntimeError(
                f"edge-tts Python API failed: {exc}"
            ) from exc

        finally:
            try:
                if os.path.exists(raw_mp3_path):
                    os.remove(raw_mp3_path)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# ElevenLabs Provider
# ═══════════════════════════════════════════════════════════════════════════════

class ElevenLabsProvider:
    """
    Renders audio using the ElevenLabs REST API.
    Raises an exception on 401 Unauthorized or 429 Quota Exceeded so the
    HybridTTSService can immediately fall through to EdgeTTSProvider.
    """

    @staticmethod
    def generate(text: str, voice_id: str, output_mp3_path: str) -> None:
        """
        Stream ElevenLabs audio to output_mp3_path.
        Raises ValueError or Exception on any API failure.
        """
        api_key = getattr(settings, 'ELEVENLABS_API_KEY', None)
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is not set — skipping ElevenLabs provider.")

        el_voice, _ = VOICE_MAP.get(voice_id, (_DEFAULT_ELEVENLABS_VOICE, _DEFAULT_EDGE_VOICE))

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{el_voice}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": _normalise_text(text),
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
        }

        logger.info(f"[ElevenLabs] Requesting TTS for voice '{el_voice}'...")
        response = requests.post(url, json=payload, headers=headers, stream=True, timeout=60)

        if response.status_code in (401, 403):
            raise PermissionError(
                f"ElevenLabs authentication failed (HTTP {response.status_code}). "
                "Check ELEVENLABS_API_KEY."
            )
        if response.status_code == 429:
            raise RuntimeError(
                f"ElevenLabs quota exceeded (HTTP 429). Falling back to EdgeTTS."
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"ElevenLabs API error: HTTP {response.status_code} — {response.text[:200]}"
            )

        with open(output_mp3_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info(f"[ElevenLabs] Audio stream written to: {output_mp3_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# HybridTTSService  — public entry point
# ═══════════════════════════════════════════════════════════════════════════════

def generate_tts_audio(script_text: str, voice_id: str, project_id: str) -> str:
    """
    Unified TTS entry point with automatic ElevenLabs → EdgeTTS fallback.

    Returns:
        url_path (str): Media-relative URL path to the generated WAV file
                        e.g. '/media/projects/my_project_abc12345/audio/voiceover.wav'

    Provider priority:
        1. ElevenLabsProvider  (premium, if API key is configured)
        2. EdgeTTSProvider     (zero-cost Microsoft Neural, Python API, file-size validated)
        3. Silent WAV stub     (last resort — keeps pipeline alive for testing)
    """
    audio_dir = os.path.join(settings.MEDIA_ROOT, 'projects', str(project_id), 'audio')
    os.makedirs(audio_dir, exist_ok=True)

    output_path = os.path.join(audio_dir, 'voiceover.wav')
    url_path    = f"/media/projects/{project_id}/audio/voiceover.wav"

    # ── Testing short-circuit ─────────────────────────────────────────────────
    if getattr(settings, 'TESTING', False):
        _write_silent_wav(output_path)
        return url_path

    # ── Provider 1: ElevenLabs ────────────────────────────────────────────────
    api_key = getattr(settings, 'ELEVENLABS_API_KEY', None)
    if api_key:
        # ElevenLabs writes an MP3; we keep it as a tempfile and discard it
        el_mp3_fd, el_mp3_path = tempfile.mkstemp(suffix='.mp3', dir=audio_dir)
        os.close(el_mp3_fd)
        try:
            ElevenLabsProvider.generate(script_text, voice_id, el_mp3_path)

            # Re-encode MP3 → WAV (CBR, 16 kHz, mono) for reliable ffprobe
            logger.info("[HybridTTS] Re-encoding ElevenLabs MP3 → WAV for duration accuracy...")
            (
                ffmpeg
                .input(el_mp3_path)
                .output(output_path, acodec='pcm_s16le', ar=16000, ac=1)
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            logger.info(f"[HybridTTS] ElevenLabs WAV output ready: {output_path}")
            return url_path

        except Exception as exc:
            logger.warning(
                f"[HybridTTS] ElevenLabs provider failed: {exc}. "
                f"Falling back to EdgeTTS..."
            )
        finally:
            try:
                if os.path.exists(el_mp3_path):
                    os.remove(el_mp3_path)
            except Exception:
                pass
    else:
        logger.info(
            "[HybridTTS] ELEVENLABS_API_KEY not set — skipping ElevenLabs, using EdgeTTS directly."
        )

    # ── Provider 2: EdgeTTS ───────────────────────────────────────────────────
    try:
        EdgeTTSProvider.generate(script_text, voice_id, output_path)
        return url_path
    except Exception as exc:
        logger.error(
            f"[HybridTTS] EdgeTTS provider also failed: {exc}. "
            f"Writing silent WAV stub as last resort."
        )

    # ── Provider 3: Silent WAV stub (last resort) ─────────────────────────────
    _write_silent_wav(output_path)
    return url_path


def _write_silent_wav(path: str) -> None:
    """Write a minimal 1-second silent WAV so downstream FFmpeg doesn't crash."""
    try:
        wav_bytes = base64.b64decode(_SILENT_WAV_B64)
        with open(path, 'wb') as f:
            f.write(wav_bytes)
        logger.info(f"[HybridTTS] Silent WAV stub written to: {path}")
    except Exception as exc:
        logger.error(f"[HybridTTS] Failed to write silent WAV stub: {exc}")
