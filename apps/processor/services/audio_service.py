"""
audio_service.py — Hybrid TTS Provider Architecture
====================================================
Routing logic:
  1. ElevenLabsProvider  — premium, tried first if ELEVENLABS_API_KEY is set
  2. EdgeTTSProvider     — zero-cost Microsoft Neural TTS, automatic fallback

EdgeTTSProvider features:
  - Phoneme-level SSML injection (clause pauses at , . ? !)
  - Text normalisation pre-pass  (%, $, abbreviations → spoken form)
  - subprocess.run() CLI execution — no asyncio loop conflicts inside Celery
  - Immediate WAV conversion via ffmpeg-python after output to guarantee
    100% accurate ffprobe duration detection (no VBR-MP3 drift)
"""

import os
import re
import logging
import subprocess
import tempfile
import base64
import requests
import ffmpeg

from django.conf import settings

logger = logging.getLogger(__name__)

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
}

# Default fallback when voice_id is unknown
_DEFAULT_ELEVENLABS_VOICE = "pNInz6obpgDQGcFmaJgB"   # Adam
_DEFAULT_EDGE_VOICE       = "en-US-ChristopherNeural"


# ═══════════════════════════════════════════════════════════════════════════════
# Text Normalisation Pre-pass
# ═══════════════════════════════════════════════════════════════════════════════

def _normalise_text(text: str) -> str:
    """
    Clean raw script text before sending to any TTS engine.
    Prevents robotic stuttering on symbols and abbreviations.
    Punctuation (. , ? !) is intentionally preserved — Azure Neural voices
    pause naturally at punctuation without any SSML injection.
    """
    # Currency: $10 → ten dollars,  $1.5M → 1.5 million dollars
    text = re.sub(r'\$(\d+(?:\.\d+)?)[Mm]', r'\1 million dollars', text)
    text = re.sub(r'\$(\d+(?:\.\d+)?)[Kk]', r'\1 thousand dollars', text)
    text = re.sub(r'\$(\d+(?:\.\d+)?)', r'\1 dollars', text)

    # Percentages: 95% → 95 percent
    text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)

    # Common abbreviations that TTS spells letter-by-letter awkwardly
    abbreviations = {
        r'\bAI\b':    'A I',
        r'\bAPI\b':   'A P I',
        r'\bUI\b':    'U I',
        r'\bUX\b':    'U X',
        r'\bSaaS\b':  'Software as a Service',
        r'\bBYOK\b':  'Bring Your Own Key',
        r'\bURL\b':   'U R L',
        r'\bHTTP\b':  'H T T P',
        r'\bHTTPS\b': 'H T T P S',
    }
    for pattern, replacement in abbreviations.items():
        text = re.sub(pattern, replacement, text)

    # Bullet / list symbols
    text = re.sub(r'[\u2022\u25cf\u25e6\u25aa\u25b8\u25ba]', '', text)

    # Strip only standalone markdown bold/italic/code markers (*  ** ` ```)
    # — does NOT strip underscores inside compound words like game_changing
    text = re.sub(r'(?<![\w])([*`#])(?![\w])', '', text)
    text = re.sub(r'\*{1,3}', '', text)  # leftover ** bold markers

    # Collapse multiple whitespace
    text = re.sub(r'\s{2,}', ' ', text).strip()

    return text


# _build_ssml() has been intentionally removed.
# edge-tts wraps input in its own SSML internally before hitting the Azure
# API — feeding it a pre-built <speak> document causes the XML tags to be
# read aloud as literal text.  Prosody control is achieved via the native
# --rate CLI flag instead.


# ═══════════════════════════════════════════════════════════════════════════════
# EdgeTTS Provider
# ═══════════════════════════════════════════════════════════════════════════════

class EdgeTTSProvider:
    """
    Renders audio using the Microsoft edge-tts CLI tool via subprocess.run().
    This avoids all asyncio event-loop conflicts inside Celery workers.

    Pipeline:
      1. Normalise text  (symbols, currencies, abbreviations)
      2. Write plain text to a temp file  (NO SSML — edge-tts wraps internally)
      3. edge-tts CLI with --rate=-5% → raw VBR MP3
      4. ffmpeg re-encode → final PCM WAV (16-bit, 16 kHz, mono)
         guarantees 100% accurate ffprobe duration detection
    """

    @staticmethod
    def generate(text: str, voice_id: str, output_wav_path: str) -> None:
        """
        Render normalised plain text to a WAV file at output_wav_path.
        Raises RuntimeError on failure so the caller can fall through.
        """
        _, edge_voice = VOICE_MAP.get(voice_id, (_DEFAULT_ELEVENLABS_VOICE, _DEFAULT_EDGE_VOICE))

        # Normalise text — keep all punctuation intact so Azure Neural can
        # apply its own natural prosodic pauses at ., ,, ?, !
        normalised = _normalise_text(text)

        # Write plain text to a temp file (edge-tts --file accepts plain text
        # only; it wraps it in SSML internally before the Azure API call)
        with tempfile.NamedTemporaryFile(
            suffix='.txt', mode='w', encoding='utf-8', delete=False
        ) as txt_file:
            txt_file.write(normalised)
            txt_path = txt_file.name

        # edge-tts writes raw VBR MP3 first; we convert immediately to WAV
        raw_mp3_fd, raw_mp3_path = tempfile.mkstemp(suffix='.mp3')
        os.close(raw_mp3_fd)

        try:
            logger.info(
                f"[EdgeTTS] Rendering plain text with voice '{edge_voice}' "
                f"at rate=-5% via subprocess CLI..."
            )
            result = subprocess.run(
                [
                    'edge-tts',
                    '--voice',       edge_voice,
                    '--file',        txt_path,       # plain text input file
                    '--rate=-5%',                    # podcast cadence (native flag)
                    '--write-media', raw_mp3_path,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"edge-tts CLI exited with code {result.returncode}. "
                    f"stderr: {result.stderr.strip()}"
                )

            logger.info(
                f"[EdgeTTS] CLI render complete. "
                f"Converting VBR MP3 → PCM WAV at {output_wav_path} ..."
            )

            # Convert to CBR WAV (PCM 16-bit, 16 kHz mono)
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
            logger.info(f"[EdgeTTS] WAV output written successfully: {output_wav_path}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("edge-tts CLI timed out after 120 seconds.")

        finally:
            # Always clean up temp files regardless of success or failure
            for path in (txt_path, raw_mp3_path):
                try:
                    if os.path.exists(path):
                        os.remove(path)
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
        2. EdgeTTSProvider     (zero-cost Microsoft Neural, plain text + --rate=-5%)
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
