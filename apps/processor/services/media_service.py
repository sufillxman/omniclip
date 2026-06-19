import os
import time
import shutil
import logging
import base64
import requests
import tempfile
from django.conf import settings
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

# Base64 encoded 1-second silent MP3 (valid audio file structure for fallback)
SILENT_MP3_B64 = (
    "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjU2LjM2LjEwMAAAAAAAAAAAAAAA//OEAAAAAAAAAAAAAAAAAAAAAAAASW5mbwAAAA8AAAAEAAABIADAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDV1dXV1dXV1dXV1dXV1dXV1dXV1dXV1dXV6urq6urq6urq6urq6urq6urq6urq6urq6v///g=="
)

def generate_tts_audio(script_text: str, voice_id: str, project_id: str) -> str:
    """
    Generates text-to-speech audio using ElevenLabs API.
    Saves the output to the media/projects/{project_id}/audio/ directory.
    Falls back to a default voice strategy (silent audio) if API fails or is unavailable.
    """
    # Establish project-specific audio directory
    audio_dir = os.path.join(settings.MEDIA_ROOT, 'projects', str(project_id), 'audio')
    os.makedirs(audio_dir, exist_ok=True)
    output_path = os.path.join(audio_dir, 'voiceover.mp3')

    # Return path representation matching standard URL routing structure
    url_path = f"/media/projects/{project_id}/audio/voiceover.mp3"

    if getattr(settings, 'TESTING', False):
        try:
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(SILENT_MP3_B64))
        except Exception as write_err:
            logger.error(f"Failed to write testing silent audio file: {write_err}")
        return url_path

    api_key = getattr(settings, 'ELEVENLABS_API_KEY', None)
    
    # Voice ID mapping
    voice_mapping = {
        "standard_male": "pNInz6obpgDQGcFmaJgB",   # Adam
        "standard_female": "21m00Tcm4TlvDq8ikWAM", # Rachel
    }
    actual_voice_id = voice_mapping.get(voice_id, voice_id or "pNInz6obpgDQGcFmaJgB")

    try:
        if not api_key:
            raise ValueError("ELEVENLABS_API_KEY is not configured in settings.")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{actual_voice_id}"
        headers = {
            "xi-api-key": api_key,
            "Content-Type": "application/json"
        }
        data = {
            "text": script_text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        logger.info(f"Initiating ElevenLabs TTS request (Voice: {actual_voice_id}, Model: eleven_multilingual_v2)...")
        response = requests.post(url, json=data, headers=headers, stream=True)

        if response.status_code != 200:
            raise Exception(f"ElevenLabs API error: HTTP {response.status_code} - {response.text}")

        # Stream response contents to disk
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        logger.info(f"Successfully generated ElevenLabs TTS audio at {output_path}")
        return url_path

    except Exception as exc:
        logger.warning(
            f"ElevenLabs TTS generation failed: {exc}. "
            f"Falling back to default voice strategy (silent audio)."
        )
        try:
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(SILENT_MP3_B64))
            return url_path
        except Exception as write_err:
            logger.error(f"Failed to write fallback silent audio file: {write_err}")
            return url_path

def fetch_background_video(keyword: str, project_id: str, chunk_index: int, layout: str = None) -> str:
    """
    Fetches a background video clip matching a keyword from the Pexels Video Search API.
    Saves the video stream to the media/projects/{project_id}/clips/ directory as chunk_{chunk_index}.mp4.

    Robustness guarantees:
      - Outer retry loop: 3 attempts with exponential backoff (1s, 2s) around the
        entire download — catches IncompleteRead, connection resets, and transient
        Pexels 5xx errors that the urllib3 adapter can't handle at stream level.
      - urllib3 Retry adapter: auto-retries on connection drops and 5xx/429 errors
        (3 attempts, 1.5x exponential backoff) for the search request.
      - Strict timeouts: 15s connect + 60s read prevents Celery worker hangs.
      - Chunked streaming: 8192-byte chunks keep memory flat regardless of file size.
      - Atomic write via tempfile + os.replace(): a mid-download crash never
        leaves a corrupt partial .mp4 on disk.
      - Smart fallback: if ALL retries fail and chunk_index > 0, duplicate the
        previous chunk rather than creating an empty file that crashes FFmpeg.
        If chunk_index == 0 fails, raise immediately to abort and refund the user.
    """
    if getattr(settings, 'TESTING', False):
        return f"/media/projects/{project_id}/clips/chunk_{chunk_index}.mp4"

    api_key = getattr(settings, 'PEXELS_API_KEY', None)

    clips_dir = os.path.join(settings.MEDIA_ROOT, 'projects', str(project_id), 'clips')
    os.makedirs(clips_dir, exist_ok=True)
    output_path = os.path.join(clips_dir, f"chunk_{chunk_index}.mp4")
    url_path = f"/media/projects/{project_id}/clips/chunk_{chunk_index}.mp4"

    # Build a resilient session: retries on connection drops and transient server errors.
    retry_strategy = Retry(
        total=3,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # ── Outer retry loop: 3 full attempts for the entire download ─────────────
    # Handles IncompleteRead, connection resets, and stream-level errors that
    # the urllib3 adapter cannot intercept after the response has already started.
    MAX_ATTEMPTS = 3
    last_exc = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            if not api_key:
                raise ValueError("PEXELS_API_KEY is not configured in settings.")

            # ── Step 1: Search Pexels ─────────────────────────────────────────
            params = {"query": keyword, "per_page": 5}
            if layout == 'landscape':
                params['orientation'] = 'landscape'
            elif layout == 'vertical':
                params['orientation'] = 'portrait'
            search_url = f"https://api.pexels.com/videos/search?{urlencode(params)}"
            logger.info(
                f"[PexelsFetch] Attempt {attempt}/{MAX_ATTEMPTS}: "
                f"searching for '{keyword}'..."
            )
            search_response = session.get(
                search_url,
                headers={"Authorization": api_key},
                timeout=(15, 60),
            )
            if search_response.status_code != 200:
                raise Exception(
                    f"Pexels API error: HTTP {search_response.status_code} - "
                    f"{search_response.text[:200]}"
                )

            data = search_response.json()
            videos = data.get("videos", [])
            if not videos:
                raise Exception(f"No videos found on Pexels for keyword: '{keyword}'")

            # ── Step 2: Select best HD (<=1080p) MP4 ─────────────────────────
            HD_MAX_PIXELS = 1920 * 1080

            def _is_mp4(vf):
                return (
                    "mp4" in (vf.get("file_type") or "").lower()
                    or (vf.get("link") or "").lower().endswith(".mp4")
                )

            def _resolution(vf):
                return (vf.get("width") or 0) * (vf.get("height") or 0)

            best_file = None
            for video in videos:
                hd_mp4_files = [
                    vf for vf in video.get("video_files", [])
                    if _is_mp4(vf) and 0 < _resolution(vf) <= HD_MAX_PIXELS
                ]
                if hd_mp4_files:
                    best_file = max(hd_mp4_files, key=_resolution)
                    break

            if not best_file:
                all_files = videos[0].get("video_files", [])
                mp4_files = [vf for vf in all_files if _is_mp4(vf)]
                best_file = (
                    min(mp4_files, key=_resolution)
                    if mp4_files
                    else (all_files[0] if all_files else None)
                )

            if not best_file:
                raise Exception("No suitable video file found in Pexels search results.")

            download_url = best_file.get("link")
            if not download_url:
                raise Exception("No download link found inside selected Pexels video file.")

            res_w = best_file.get("width", "?")
            res_h = best_file.get("height", "?")
            logger.info(
                f"[PexelsFetch] Downloading {res_w}x{res_h} clip from {download_url}..."
            )

            # ── Step 3: Chunked streaming download with atomic write ──────────
            with session.get(download_url, stream=True, timeout=(15, 60)) as r:
                if r.status_code != 200:
                    raise Exception(
                        f"Failed to download Pexels video stream: HTTP {r.status_code}"
                    )

                fd, tmp_path = tempfile.mkstemp(dir=clips_dir)
                bytes_written = 0
                try:
                    with os.fdopen(fd, "wb") as f:
                        for data_chunk in r.iter_content(chunk_size=8192):
                            if data_chunk:
                                f.write(data_chunk)
                                bytes_written += len(data_chunk)
                    os.replace(tmp_path, output_path)  # Atomic rename
                except Exception:
                    try:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except Exception:
                        pass
                    raise

            logger.info(
                f"[PexelsFetch] Successfully downloaded "
                f"{bytes_written / 1_048_576:.1f} MB → {output_path}"
            )
            return url_path

        except Exception as exc:
            last_exc = exc
            logger.warning(
                f"[PexelsFetch] Attempt {attempt}/{MAX_ATTEMPTS} failed for "
                f"chunk_{chunk_index} (keyword='{keyword}'): {exc}"
            )
            # Clean up any stale .tmp files from this failed attempt
            try:
                for f_name in os.listdir(clips_dir):
                    if f_name.endswith(".tmp"):
                        os.remove(os.path.join(clips_dir, f_name))
            except Exception:
                pass

            if attempt < MAX_ATTEMPTS:
                sleep_secs = 2 ** (attempt - 1)   # 1s, 2s (exponential)
                logger.info(
                    f"[PexelsFetch] Retrying chunk_{chunk_index} in {sleep_secs}s..."
                )
                time.sleep(sleep_secs)

    # ── All retries exhausted — smart fallback ────────────────────────────────
    logger.error(
        f"[PexelsFetch] All {MAX_ATTEMPTS} download attempts failed for "
        f"chunk_{chunk_index} (keyword='{keyword}'). Last error: {last_exc}"
    )

    # chunk_0 failure: we have no prior clip to clone — abort so the task can
    # trigger a credit refund rather than building a broken video.
    if chunk_index == 0:
        raise RuntimeError(
            f"[PexelsFetch] chunk_0 download failed after {MAX_ATTEMPTS} attempts "
            f"and no fallback clip is available. Aborting render to trigger refund. "
            f"Original error: {last_exc}"
        )

    # chunk_N (N > 0) failure: duplicate the previous chunk so FFmpeg receives a
    # valid MP4 file rather than an empty 0-byte stub.
    prev_chunk_path = os.path.join(clips_dir, f"chunk_{chunk_index - 1}.mp4")
    if not os.path.exists(prev_chunk_path):
        raise RuntimeError(
            f"[PexelsFetch] chunk_{chunk_index} failed and fallback chunk "
            f"chunk_{chunk_index - 1} does not exist either. Aborting render."
        )

    shutil.copy(prev_chunk_path, output_path)
    logger.warning(
        f"[PexelsFetch] Cloned chunk_{chunk_index - 1}.mp4 as chunk_{chunk_index}.mp4 "
        f"after all download attempts failed. Visual content will repeat for this segment."
    )
    return url_path


def fetch_background_video_with_status(keyword: str, project_id: str, chunk_index: int, layout: str = None) -> tuple:
    """
    Thin wrapper around fetch_background_video() that additionally returns a boolean
    flag indicating whether the returned clip is a clone of the previous chunk.

    The flag is used by assemble_final_video() to apply -stream_loop -1 on cloned
    clips, preventing black frames or freeze artifacts when the cloned MP4's native
    duration is shorter than the Whisper-derived target duration.

    Returns:
        (url_path: str, was_cloned: bool)
            url_path   — media-relative URL to the downloaded or cloned MP4
            was_cloned — True if the file is a copy of chunk_{chunk_index-1}.mp4
    """
    clips_dir       = os.path.join(settings.MEDIA_ROOT, 'projects', str(project_id), 'clips')
    output_path     = os.path.join(clips_dir, f"chunk_{chunk_index}.mp4")
    prev_chunk_path = os.path.join(clips_dir, f"chunk_{chunk_index - 1}.mp4") if chunk_index > 0 else None

    # Snapshot mtime of previous chunk before calling fetch, so we can detect
    # whether a clone occurred (the file will be identical bytes to prev_chunk).
    prev_mtime_before = None
    if prev_chunk_path and os.path.exists(prev_chunk_path):
        prev_mtime_before = os.path.getmtime(prev_chunk_path)

    url_path = fetch_background_video(keyword, project_id, chunk_index, layout=layout)

    # Detect clone: if output_path now has the same size as prev_chunk_path,
    # and prev_chunk_path was not touched during the fetch, it was cloned.
    was_cloned = False
    if (
        chunk_index > 0
        and prev_chunk_path
        and os.path.exists(output_path)
        and os.path.exists(prev_chunk_path)
        and os.path.getsize(output_path) == os.path.getsize(prev_chunk_path)
        and os.path.getmtime(prev_chunk_path) == prev_mtime_before
    ):
        was_cloned = True
        logger.info(
            f"[PexelsFetch] Clone detected for chunk_{chunk_index}: "
            f"same size as chunk_{chunk_index - 1} ({os.path.getsize(output_path)} bytes). "
            f"stream_loop will be applied in FFmpeg."
        )

    return url_path, was_cloned
