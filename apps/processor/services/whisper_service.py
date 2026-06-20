import os
import re
import logging
from dataclasses import dataclass, field
from typing import List
from django.conf import settings

logger = logging.getLogger(__name__)

# ── Subtitle chunking constants (tunable at the architect level) ───────────────
# Punctuation characters that FORCE a chunk boundary — a sentence end is sacred.
_SENTENCE_ENDERS: frozenset = frozenset(".!?;:।॥")

_MAX_WORDS       = 7      # Soft: close the chunk after this many words
_MAX_DURATION_S  = 2.5   # Hard: close the chunk after this many seconds
_MIN_DURATION_S  = 0.4  # Floor: chunks shorter than this are merged forward


@dataclass
class SubtitleChunk:
    """
    Represents one displayable subtitle unit produced by sentence_aware_chunk().

    Attributes:
        words:              Raw word strings that form the display text.
        start_time:         Audio start in seconds (from Whisper word timestamps).
        end_time:           Audio end in seconds (from Whisper word timestamps).
        is_cloned:          True when the backing video clip was duplicated from
                            the previous chunk after a Pexels fetch failure.
        source_chunk_index: Index of the original clip that was cloned (-1 if not cloned).
    """
    words:              List[str]
    start_time:         float
    end_time:           float
    is_cloned:          bool  = False
    source_chunk_index: int   = -1

    @property
    def text(self) -> str:
        """Space-joined display text for ASS rendering."""
        return " ".join(self.words)

    @property
    def duration(self) -> float:
        """Audio duration of this chunk in seconds."""
        return round(self.end_time - self.start_time, 3)

# Whisper model size: 'base' is the best CPU trade-off (fast, ~74MB, accurate enough for alignment)
WHISPER_MODEL_SIZE = "base"


def sentence_aware_chunk(word_timestamps: list) -> List[SubtitleChunk]:
    """
    Deterministic, punctuation-respecting subtitle chunker.

    Converts a flat list of Whisper word-timestamp dicts into SubtitleChunk
    objects where every chunk boundary respects four priority rules:

        PRIORITY 1: Sentence ender (.!?;:)   → ALWAYS close the chunk
        PRIORITY 2: Max duration  (2.2s)     → close (prevents long text on-screen)
        PRIORITY 3: Max words     (4)        → close (visual comfort)
        PRIORITY 4: Last word               → flush any remaining words

    Micro-chunks (< 150ms with no natural Whisper pause) are merged into the
    next chunk to prevent imperceptible screen flashes — unless the micro-chunk
    itself ends a sentence, in which case the boundary is always honoured.

    Args:
        word_timestamps: List of dicts from WhisperAlignmentService._extract_words():
                         [{"word": str, "start": float, "end": float}, ...]

    Returns:
        List[SubtitleChunk] ordered chronologically.
    """
    if not word_timestamps:
        return []

    chunks: List[SubtitleChunk] = []
    current_words: List[str]    = []
    current_start: float        = 0.0
    current_end: float          = 0.0
    total_tokens                = len(word_timestamps)

    for i, token in enumerate(word_timestamps):
        word    = token["word"]
        t_start = float(token["start"])
        t_end   = float(token["end"])

        # ── Open a new chunk on the first word ───────────────────────────────
        if not current_words:
            current_start = t_start

        current_words.append(word)
        current_end = t_end

        # ── Boundary condition evaluation ─────────────────────────────────────
        current_duration = current_end - current_start
        stripped         = word.rstrip()
        last_char        = stripped[-1] if stripped else ""
        hits_sentence    = last_char in _SENTENCE_ENDERS
        hits_word_limit  = len(current_words) >= _MAX_WORDS
        hits_time_limit  = current_duration >= _MAX_DURATION_S
        is_last_word     = (i == total_tokens - 1)

        should_close = hits_sentence or hits_word_limit or hits_time_limit or is_last_word

        if not should_close:
            continue

        # ── Micro-chunk merge-forward guard ───────────────────────────────────
        # A lone word like "Hi." lasting 0.08s would flash imperceptibly.
        # Merge it into the next chunk UNLESS:
        #   a) it ends a sentence (always honour the boundary), OR
        #   b) it hits the strict word limit or time limit (always force close), OR
        #   c) there is a natural Whisper silence gap of >= 0.1s after it, OR
        #   d) it is the very last word (must flush).
        if not is_last_word and not hits_sentence and not hits_word_limit and not hits_time_limit and current_duration < _MIN_DURATION_S:
            next_gap = word_timestamps[i + 1]["start"] - t_end
            if next_gap < 0.1:
                # Carry these words into the next iteration — do not close yet
                continue

        chunks.append(
            SubtitleChunk(
                words      = current_words[:],
                start_time = round(current_start, 3),
                end_time   = round(current_end,   3),
            )
        )

        # Reset accumulator
        current_words = []
        current_start = 0.0
        current_end   = 0.0

    logger.debug(
        f"[sentence_aware_chunk] Produced {len(chunks)} subtitle chunks "
        f"from {total_tokens} word timestamps."
    )
    return chunks


class WhisperAlignmentService:
    """
    Audio-Driven Sync: uses faster-whisper to transcribe the TTS voiceover and extract
    per-word timestamps. Maps those timestamps onto the Gemini-generated scene segments
    to produce exact, audio-true start_time / end_time for every video clip.

    Pipeline:
        voiceover.mp3 → faster-whisper → word-level timestamps
                      → greedy segment alignment
                      → enriched timeline [{"chunk_index", "text", "visual_keyword",
                                            "start_time", "end_time"}, ...]
    """

    @staticmethod
    def _resolve_local_path(path: str) -> str:
        """Resolve a media URL-style path to an absolute filesystem path."""
        if os.path.isabs(path) and os.path.exists(path):
            return path
        media_url = getattr(settings, 'MEDIA_URL', '/media/')
        if path.startswith(media_url):
            rel = path[len(media_url):].replace('/', os.sep)
            return os.path.normpath(os.path.join(settings.MEDIA_ROOT, rel))
        return path

    @staticmethod
    def _uniform_fallback(timeline: list, total_duration: float = None) -> list:
        """
        Fallback: distribute time uniformly across segments when Whisper is unavailable.
        Uses a fixed 3.5-second-per-chunk estimate if total_duration is unknown.
        """
        chunk_duration = (total_duration / len(timeline)) if (total_duration and len(timeline)) else 3.5
        enriched = []
        for i, chunk in enumerate(timeline):
            enriched.append({
                **chunk,
                "start_time": round(i * chunk_duration, 3),
                "end_time":   round((i + 1) * chunk_duration, 3),
            })
        logger.warning(
            f"[WhisperService] Using uniform fallback: {len(enriched)} segments "
            f"at {chunk_duration:.2f}s each."
        )
        return enriched

    @staticmethod
    def _extract_words(segments) -> list:
        """
        Flatten faster-whisper segment/word objects into a simple list of dicts:
        [{"word": str, "start": float, "end": float}, ...]
        """
        words = []
        for seg in segments:
            if hasattr(seg, 'words') and seg.words:
                for w in seg.words:
                    words.append({
                        "word":  w.word.strip().lower(),
                        "start": w.start,
                        "end":   w.end,
                    })
        return words

    @staticmethod
    def _align_segments(timeline: list, words: list) -> list:
        """
        Greedy sequential alignment:
        For each scene segment (which contains N words of text), consume the
        matching N words from the flat word list. The segment's start_time is the
        first consumed word's start, and end_time is the last consumed word's end.

        Falls back to the next available word boundary if text doesn't align perfectly
        (handles minor TTS pronunciation differences).
        """
        import re

        def tokenize(text: str) -> list:
            return re.findall(r"[\w']+", text.lower())

        enriched = []
        word_cursor = 0
        total_words = len(words)

        for chunk in timeline:
            chunk_tokens = tokenize(chunk.get('text', ''))
            n_tokens = len(chunk_tokens)

            if word_cursor >= total_words or n_tokens == 0:
                # No more words to consume — use boundary of last known word
                last_end = words[-1]["end"] if words else 0.0
                enriched.append({
                    **chunk,
                    "start_time": round(last_end, 3),
                    "end_time":   round(last_end + 3.5, 3),
                })
                continue

            seg_start = words[word_cursor]["start"]
            # Consume up to n_tokens words (or remaining words, whichever is fewer)
            end_cursor = min(word_cursor + n_tokens, total_words) - 1
            seg_end = words[end_cursor]["end"]

            enriched.append({
                **chunk,
                "start_time": round(seg_start, 3),
                "end_time":   round(seg_end, 3),
            })

            word_cursor = end_cursor + 1

        return enriched

    @classmethod
    def align(cls, audio_path: str, timeline: list) -> list:
        """
        Main entry point. Resolves the audio file, runs Whisper transcription with
        word-level timestamps, and returns the timeline enriched with exact audio timestamps.

        Args:
            audio_path: Absolute path or media-URL path to the voiceover MP3.
            timeline:   List of Gemini scene dicts with 'text' and 'visual_keyword'.

        Returns:
            Enriched timeline list — each item has 'start_time' and 'end_time' added.
        """
        if not timeline:
            logger.warning("[WhisperService] Empty timeline — nothing to align.")
            return timeline

        # In test mode, skip real Whisper inference entirely
        if getattr(settings, 'TESTING', False):
            logger.info("[WhisperService] TESTING=True — using uniform fallback (no model loaded).")
            return cls._uniform_fallback(timeline)

        local_path = cls._resolve_local_path(audio_path)
        if not os.path.exists(local_path):
            logger.warning(
                f"[WhisperService] Audio file not found at '{local_path}'. "
                f"Using uniform fallback."
            )
            return cls._uniform_fallback(timeline)

        try:
            from faster_whisper import WhisperModel  # lazy import — only loaded at render time

            logger.info(
                f"[WhisperService] Loading faster-whisper model '{WHISPER_MODEL_SIZE}' "
                f"and transcribing: {local_path}"
            )
            model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
            segments, info = model.transcribe(local_path, word_timestamps=True)
            segments = list(segments)  # exhaust the generator

            total_duration = info.duration if hasattr(info, 'duration') else None
            logger.info(
                f"[WhisperService] Transcription complete. "
                f"Audio duration: {total_duration}s. Segments: {len(segments)}"
            )

            words = cls._extract_words(segments)
            if not words:
                logger.warning("[WhisperService] No word timestamps extracted — using uniform fallback.")
                return cls._uniform_fallback(timeline, total_duration)

            enriched = cls._align_segments(timeline, words)
            logger.info(
                f"[WhisperService] Alignment complete: {len(enriched)} scenes mapped to audio timestamps."
            )
            return enriched

        except ImportError:
            logger.error(
                "[WhisperService] faster-whisper is not installed. "
                "Run: pip install faster-whisper. Using uniform fallback."
            )
            return cls._uniform_fallback(timeline)

        except Exception as exc:
            logger.error(
                f"[WhisperService] Whisper transcription failed: {exc}. "
                f"Using uniform fallback."
            )
            return cls._uniform_fallback(timeline)
