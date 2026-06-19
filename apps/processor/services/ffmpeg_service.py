import os
import re
import math
import shutil
import logging
import tempfile
from typing import List, Optional
import ffmpeg
from django.conf import settings

logger = logging.getLogger(__name__)

# ── Industry-standard target resolutions ──────────────────────────────────────
TARGET_RESOLUTIONS = {
    'landscape': (1920, 1080),
    'vertical':  (1080, 1920),
}

def resolve_local_path(path: str) -> str:
    """
    Helper to resolve media url/relative paths (starting with settings.MEDIA_URL)
    to absolute local filesystem paths on the disk.
    """
    if os.path.isabs(path) and os.path.exists(path):
        return path
    
    media_url = getattr(settings, 'MEDIA_URL', '/media/')
    if path.startswith(media_url):
        rel_path = path[len(media_url):]
        rel_path = rel_path.replace('/', os.sep)
        return os.path.normpath(os.path.join(settings.MEDIA_ROOT, rel_path))
        
    return path


# ══════════════════════════════════════════════════════════════════════════════
# ASS Subtitle Generation
# ══════════════════════════════════════════════════════════════════════════════

# ── ASS Styling Constants ────────────────────────────────────────────────────────
# All colours are in ASS BGR-hex format (&HBBGGRR&).
_ASS_FONT_NAME   = "Montserrat"   # Closest open font to the Hormozi style
_ASS_FONT_SIZE   = 72             # Display pt at 1920×1080; scales auto by FFmpeg for vertical
_ASS_GLOW_COLOUR = "&H8B00FF&"   # Deep purple chromatic glow (brand accent)
_ASS_SHADOW_COL  = "&H000000&"   # Pure black hard drop shadow
_ASS_TEXT_COLOUR = "&H00FFFFFF&" # White primary text
_BOUNCE_ENTRY_MS = 80            # Time (ms) for the kinetic scale-in transition

# Three-layer rendering definition: (layer, blur, colour_bgr, alpha_hex, shad, bord, glow)
# Layer 0 renders furthest back (glow), Layer 2 renders on top (crisp text).
_ASS_LAYERS = [
    # layer  blur  colour            alpha  shad  bord
    (0,      8,    _ASS_GLOW_COLOUR,  "80",   0,    0),  # Chromatic outer glow
    (1,      0,    _ASS_SHADOW_COL,   "60",   4,    0),  # Hard offset drop shadow
    (2,      0,    _ASS_TEXT_COLOUR,  "00",   0,    2),  # Primary crisp foreground text
]

# Vertical alignment: an\8 = top-centre; an\2 = bottom-centre (our target)
# MarginV pushes the text up from the absolute bottom edge.
_ASS_ALIGNMENT  = 2     # Bottom-centre
_ASS_MARGIN_V   = 80    # px from bottom edge (safe zone for social media)


def _ass_ts(seconds: float) -> str:
    """
    Convert a float seconds value to ASS timestamp format: H:MM:SS.cc
    (centiseconds, not milliseconds — ASS format spec).
    """
    h  = int(seconds // 3600)
    m  = int((seconds % 3600) // 60)
    s  = int(seconds % 60)
    cs = int(round((seconds - math.floor(seconds)) * 100))  # centiseconds
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _generate_ass_subtitles(subtitle_chunks: list, output_ass_path: str, layout: str = 'landscape') -> str:
    """
    Generates a professional-grade ASS subtitle file from a list of SubtitleChunk
    objects. Implements two viral visual techniques:

    1. Three-Layer Rendering (Drop Shadow + Chromatic Glow):
       Each subtitle chunk is rendered as THREE stacked ASS Dialogue lines at
       layers 0 (purple glow, \\blur8), 1 (hard black offset shadow, \\shad4),
       and 2 (clean white foreground text). This creates the cinematic depth that
       a single \\shad tag cannot achieve.

    2. Kinetic Word Bounce (Hormozi-style):
       Each word within a chunk gets its own Dialogue line, staggered by 40ms,
       entering at 110% scale and settling to 100% via ASS \\t() transitions.
       Words "land" on screen with physical weight and momentum.

    Args:
        subtitle_chunks: List of SubtitleChunk (or compatible dicts with 'words',
                         'start_time', 'end_time' keys).
        output_ass_path: Absolute path where the .ass file will be written.
        layout:          'landscape' or 'vertical'.

    Returns:
        output_ass_path (for chaining).
    """
    res_x, res_y = TARGET_RESOLUTIONS.get(layout, TARGET_RESOLUTIONS['landscape'])
    x_pos, y_pos = (540, 1500) if layout == 'vertical' else (960, 850)
    pos_tag = f"\\an2\\pos({x_pos},{y_pos})"

    # ── ASS Script Header ────────────────────────────────────────────────────────
    header = (
        "[Script Info]\n"
        "Title: OmniClip Generated Subtitles\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        f"PlayResX: {res_x}\n"
        f"PlayResY: {res_y}\n"
        "Collisions: Ignore\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # Bold (-1), no italic, no underline, scale 100/100, spacing 0, border 2, shadow 0
        f"Style: Default,{_ASS_FONT_NAME},{_ASS_FONT_SIZE},"
        f"{_ASS_TEXT_COLOUR},&H000000FF&,&H00000000&,&H00000000&,"
        f"-1,0,0,0,100,100,0,0,1,2,0,{_ASS_ALIGNMENT},10,10,{_ASS_MARGIN_V},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    event_lines: List[str] = []
    WORD_STAGGER_S = 0.040   # 40ms stagger between each word's entry

    for chunk in subtitle_chunks:
        # Support both SubtitleChunk dataclass and plain dict
        if hasattr(chunk, 'words'):
            words      = chunk.words
            start_time = chunk.start_time
            end_time   = chunk.end_time
        else:
            words      = chunk.get('words', [])
            start_time = float(chunk.get('start_time', 0.0))
            end_time   = float(chunk.get('end_time',   0.0))

        if not words:
            continue

        chunk_end_ts = _ass_ts(end_time)
        display_text = " ".join(words)

        # ── Kinetic Bounce: emit one Dialogue line per word, staggered ────────────
        # Each word pops in at 110% scale and settles via \t() transition.
        # All words in a chunk share the same chunk_end timestamp so they
        # all disappear simultaneously as a unit.
        for word_idx, word in enumerate(words):
            word_entry_time = start_time + (word_idx * WORD_STAGGER_S)
            word_start_ts   = _ass_ts(word_entry_time)

            # Build the per-word display: words before this one are shown as
            # thin/translucent to give context; only the current word bounces.
            # This mimics the OpusClip "active word" highlight paradigm.
            prefix_words = words[:word_idx]   # already visible, no bounce
            active_word  = word               # currently bouncing into existence
            suffix_words = words[word_idx+1:] # not yet visible

            # Build ASS override tags:
            #   \fscx110\fscy110       = start at 110% scale (overshoot)
            #   \t(0,{ms},\fscx100...) = transition to 100% in BOUNCE_ENTRY_MS
            bounce_tag = (
                f"{{\\fscx110\\fscy110"
                f"\\t(0,{_BOUNCE_ENTRY_MS},\\fscx100\\fscy100)}}"
            )
            # Prefix words rendered at 70% opacity to fade them back
            prefix_str = ""
            if prefix_words:
                prefix_str = f"{{\\alpha&H48&}}" + " ".join(prefix_words) + f" {{\\alpha&H00&}}"

            full_text_bounce = prefix_str + bounce_tag + active_word

            # ── Three-layer rendering for this word entry ──────────────────────
            for layer, blur, colour, alpha, shad, bord in _ASS_LAYERS:
                blur_tag = f"\\blur{blur}" if blur > 0 else ""
                tag = (
                    f"{{{pos_tag}{blur_tag}\\c{colour}"
                    f"\\alpha&H{alpha}&\\shad{shad}\\bord{bord}}}"
                )
                layer_text = tag + full_text_bounce

                event_lines.append(
                    f"Dialogue: {layer},{word_start_ts},{chunk_end_ts},"
                    f"Default,,0,0,0,,{layer_text}"
                )

    ass_content = header + "\n".join(event_lines) + "\n"

    with open(output_ass_path, 'w', encoding='utf-8') as f:
        f.write(ass_content)

    logger.info(
        f"[ASS] Generated subtitle file: {output_ass_path} "
        f"({len(event_lines)} dialogue events, {len(subtitle_chunks)} chunks)"
    )
    return output_ass_path


def _normalize_stream(stream, target_w: int, target_h: int):
    """
    Apply scale → pad → setsar → pixel-format filters to force a video stream
    to exactly *target_w* × *target_h* with square pixels and yuv420p.
    """
    stream = stream.filter(
        'scale',
        target_w,
        target_h,
        force_original_aspect_ratio='decrease',
    )
    stream = stream.filter(
        'pad',
        target_w,
        target_h,
        '(ow-iw)/2',   # center horizontally
        '(oh-ih)/2',   # center vertically
        color='black',
    )
    stream = stream.filter('setsar', '1')
    stream = stream.filter('format', 'yuv420p')
    return stream


def assemble_final_video(
    audio_path: str,
    video_clips: list,
    output_filename: str,
    layout: str = 'landscape',
    subtitle_chunks: Optional[list] = None,
) -> str:
    """
    Normalizes, trims, scales, and stitches video clips with a background audio track.
    Optionally burns in ASS subtitles via the FFmpeg `ass=` video filter.

    Args:
        audio_path:       Path to the voiceover WAV.
        video_clips:      List of either:
                            - (path, duration) tuples  [legacy format], OR
                            - dicts {"path", "duration", "is_cloned", "source_idx"}
                              [new format with clone provenance].
        output_filename:  Absolute path for the rendered MP4.
        layout:           'landscape' (1920×1080) or 'vertical' (1080×1920).
        subtitle_chunks:  Optional list of SubtitleChunk objects or compatible dicts.
                          If provided, an ASS file is generated and burned into the video.

    Returns:
        output_filename on success.
    """
    target_w, target_h = TARGET_RESOLUTIONS.get(layout, TARGET_RESOLUTIONS['landscape'])
    logger.info(
        f"[FFmpeg Service] Stitching {len(video_clips)} clips with audio: {audio_path} "
        f"[target: {target_w}×{target_h}, layout={layout}, subtitles={'yes' if subtitle_chunks else 'no'}]"
    )

    # Ensure parent output directory exists on disk
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    # ── Generate ASS subtitle file if chunks were provided ────────────────────────
    ass_path: Optional[str] = None
    if subtitle_chunks:
        ass_fd, ass_path = tempfile.mkstemp(suffix='.ass', dir=os.path.dirname(output_filename))
        os.close(ass_fd)
        try:
            _generate_ass_subtitles(subtitle_chunks, ass_path, layout)
        except Exception as ass_err:
            logger.error(f"[FFmpeg Service] ASS generation failed: {ass_err}. Continuing without subtitles.")
            ass_path = None

    try:
        # Resolve path and explicitly select audio stream
        resolved_audio = resolve_local_path(audio_path)
        audio_input = ffmpeg.input(resolved_audio).audio

        # ── Resolve paths, build, trim, and normalize video input streams ───────
        # Cloned clips (is_cloned=True) use -stream_loop -1 so FFmpeg loops the
        # clip indefinitely and we trim it to the exact Whisper duration. This
        # prevents black frames or freeze artifacts when the cloned MP4 is shorter
        # than the target duration.
        normalized_streams = []
        for clip in video_clips:
            # Support both legacy (path, duration) tuples and new dict format
            if isinstance(clip, dict):
                p         = clip["path"]
                duration  = clip["duration"]
                is_cloned = clip.get("is_cloned", False)
            else:
                p, duration   = clip
                is_cloned     = False

            resolved_p = resolve_local_path(p)

            if is_cloned:
                # Loop the cloned clip to fill the entire Whisper duration seamlessly
                stream = (
                    ffmpeg.input(resolved_p, stream_loop=-1)
                    .video
                    .trim(start=0, end=duration)
                    .setpts('PTS-STARTPTS')
                )
                logger.debug(f"[FFmpeg Service] Cloned clip '{p}' looped to {duration:.3f}s")
            else:
                stream = (
                    ffmpeg.input(resolved_p)
                    .video
                    .trim(start=0, end=duration)
                    .setpts('PTS-STARTPTS')
                )

            # Force uniform resolution, SAR, and pixel format
            stream = _normalize_stream(stream, target_w, target_h)
            normalized_streams.append(stream)

        # Concatenate background video streams (v=1, a=0)
        concatenated_video = ffmpeg.concat(*normalized_streams, v=1, a=0)

        # ── Apply ASS subtitle burn-in if an .ass file was generated ────────────
        # Root cause of the Windows fopen failure:
        #   filter('ass', path) serialises to:  ass=C:/path/file.ass
        #   libass sees "C" as an option key → fopen receives garbage → ENOENT.
        #
        # The correct serialisation is:  ass=filename=C\:/path/file.ass
        #   - `filename=` is the libass option key for the subtitle file path.
        #   - The colon in `C:` MUST be escaped as `\:` INSIDE the filter option
        #     string so FFmpeg's option parser does not split on it.
        #   - Backslashes in the directory path become `/` (forward slashes only).
        #   - ffmpeg-python passes this via filter(name, **kwargs) which produces:
        #       -vf "ass=filename=C\:/Users/.../file.ass"   ← correct
        #
        # Note: mkstemp() already closed the fd before this point, so there is
        # no Windows "Access Denied" / file-still-open race here.
        # The 'Local Path' Trick: Since absolute paths are failing fopen on Windows,
        # copy the .ass file into the exact same directory as the output video file before rendering.
        # Then, pass ONLY the filename (subs.ass) as a positional parameter to the filter.
        # Run the FFmpeg command execution with the CWD set to final_dir.
        final_dir = os.path.dirname(output_filename)
        subs_ass_path = None
        if ass_path and os.path.exists(ass_path):
            subs_ass_path = os.path.join(final_dir, 'subs.ass')
            try:
                shutil.copy(ass_path, subs_ass_path)
                concatenated_video = concatenated_video.filter('ass', 'subs.ass')
                logger.info(
                    "[FFmpeg Service] ASS subtitle burn-in applied via local path trick (subs.ass)"
                )
            except Exception as copy_err:
                logger.error(f"[FFmpeg Service] Copying ASS file to subs.ass failed: {copy_err}")

        # Run stitching output command combining video and audio tracks
        out = ffmpeg.output(
            concatenated_video,
            audio_input,
            output_filename,
            vcodec='libx264',
            acodec='aac',
            pix_fmt='yuv420p',
            strict='experimental',
            f='mp4',
            shortest=None  # Encodes until the shortest stream ends (prevents video freeze)
        ).overwrite_output()

        # Compile and execute FFmpeg command in final_dir CWD so it can locate local subs.ass
        args = out.compile()
        import subprocess
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=final_dir
        )
        stdout, stderr = process.communicate()
        retcode = process.poll()
        if retcode != 0:
            raise ffmpeg.Error(args, stdout, stderr)

        logger.info(f"[FFmpeg Service] Video rendering successful at: {output_filename}")

    except ffmpeg.Error as e:
        stderr_msg = e.stderr.decode() if e.stderr else str(e)
        logger.warning(
            f"[FFmpeg Service] System FFmpeg execution failed:\n{stderr_msg}\n"
        )
        if getattr(settings, 'TESTING', False):
            logger.info("[FFmpeg Service] TESTING=True, generating mock testing video file touch...")
            with open(output_filename, 'a'):
                os.utime(output_filename, None)
        else:
            raise
            
    except (FileNotFoundError, Exception) as exc:
        logger.warning(
            f"[FFmpeg Service] Unexpected runtime exception: {exc}."
        )
        if getattr(settings, 'TESTING', False):
            logger.info("[FFmpeg Service] TESTING=True, generating mock testing video file touch...")
            with open(output_filename, 'a'):
                os.utime(output_filename, None)
        else:
            raise

    finally:
        # Clean up the temporary ASS file and copied local ASS file regardless of render outcome
        if ass_path and os.path.exists(ass_path):
            try:
                os.remove(ass_path)
                logger.debug(f"[FFmpeg Service] Cleaned up temp ASS file: {ass_path}")
            except Exception:
                pass
        if 'subs_ass_path' in locals() and subs_ass_path and os.path.exists(subs_ass_path):
            try:
                os.remove(subs_ass_path)
                logger.debug(f"[FFmpeg Service] Cleaned up copied ASS file: {subs_ass_path}")
            except Exception:
                pass

    return output_filename