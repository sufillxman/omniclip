from abc import ABC
from typing import List, Tuple


class LanguageProfile(ABC):
    """Base class for language-specific configuration.
    
    Every language is completely isolated in its own subclass.
    Zero cross-contamination — consumers call ProfileManager.detect()
    and use the returned profile polymorphically.
    """

    # ── TTS ──────────────────────────────────────────────────────────────────
    tts_voice_id: str
    tts_rate: str
    tts_pitch: str

    # ── Whisper subtitle chunking ─────────────────────────────────────────────
    whisper_max_words: int
    whisper_max_duration: float
    whisper_min_duration: float

    # ── FFmpeg / ASS subtitle rendering ───────────────────────────────────────
    ffmpeg_font_name: str
    ffmpeg_bounce_enabled: bool
    ass_margin_v: int
    ass_font_size: int

    # ── Unicode validation ───────────────────────────────────────────────────
    allowed_unicode_blocks: List[Tuple[int, int]]
    rejected_unicode_blocks: List[Tuple[int, int]]

    # ── Detection heuristics ─────────────────────────────────────────────────
    detection_indicators: List[str]

    def validate_unicode(self, text: str) -> Tuple[bool, str]:
        """Check text only contains characters from allowed blocks.
        Returns (is_valid, error_message).
        """
        for char in text:
            code = ord(char)
            if code < 32:  # control characters
                continue
            in_allowed = any(lo <= code <= hi for lo, hi in self.allowed_unicode_blocks)
            in_rejected = any(lo <= code <= hi for lo, hi in self.rejected_unicode_blocks)
            if in_rejected:
                ranges = ", ".join(f"0x{lo:04X}-0x{hi:04X}" for lo, hi in self.rejected_unicode_blocks)
                return False, f"Contains rejected Unicode block ({ranges})"
            if not in_allowed:
                return False, f"Character U+{code:04X} not in allowed blocks"
        return True, ""


class EnglishProfile(LanguageProfile):

    tts_voice_id = "en-US-ChristopherNeural"
    tts_rate = "+0%"
    tts_pitch = "+0Hz"

    whisper_max_words = 9
    whisper_max_duration = 2.5
    whisper_min_duration = 0.3

    ffmpeg_font_name = "Arial Black"
    ffmpeg_bounce_enabled = True
    ass_margin_v = 100
    ass_font_size = 90

    allowed_unicode_blocks = [
        (0x0020, 0x007F),   # Basic Latin
        (0x00A0, 0x00FF),   # Latin-1 Supplement
        (0x0100, 0x017F),   # Latin Extended-A
        (0x2018, 0x201F),   # quotes
        (0x2026, 0x2026),   # ellipsis
    ]
    rejected_unicode_blocks: List[Tuple[int, int]] = []

    detection_indicators: List[str] = []

class HindiProfile(LanguageProfile):

    tts_voice_id = "hi-IN-MadhurNeural"
    tts_rate = "-5%"
    tts_pitch = "+0Hz"

    whisper_max_words = 6
    whisper_max_duration = 2.2
    whisper_min_duration = 0.5

    ffmpeg_font_name = "Nirmala UI"
    ffmpeg_bounce_enabled = False
    ass_margin_v = 120
    ass_font_size = 85

    allowed_unicode_blocks = [
        (0x0020, 0x007F),   # Basic Latin (punctuation, digits)
        (0x0900, 0x097F),   # Devanagari
        (0x2000, 0x206F),   # General Punctuation
        (0x2018, 0x201F),   # quotes
        (0x2026, 0x2026),   # ellipsis
    ]
    rejected_unicode_blocks = [
        (0x0600, 0x06FF),   # Arabic block (includes Urdu/Perso-Arabic)
    ]

    detection_indicators: List[str] = ["hindi", "हिंदी", "हिन्दी"]

class GujaratiProfile(LanguageProfile):

    tts_voice_id = "gu-IN-NiranjanNeural"
    tts_rate = "-5%"
    tts_pitch = "+0Hz"

    whisper_max_words = 6
    whisper_max_duration = 2.2
    whisper_min_duration = 0.5

    ffmpeg_font_name = "Nirmala UI"
    ffmpeg_bounce_enabled = False
    ass_margin_v = 120
    ass_font_size = 85

    allowed_unicode_blocks = [
        (0x0020, 0x007F),   # Basic Latin (punctuation, digits)
        (0x0A80, 0x0AFF),   # Gujarati
        (0x0964, 0x0965),   # Devanagari Danda (।॥) — common across Indian scripts
        (0x2000, 0x206F),   # General Punctuation
        (0x2018, 0x201F),   # quotes
        (0x2026, 0x2026),   # ellipsis
    ]
    rejected_unicode_blocks: List[Tuple[int, int]] = []

    detection_indicators: List[str] = ["gujarati", "ગુજરાતી"]

class ProfileManager:
    """Factory that returns the correct LanguageProfile for a given text or prompt."""

    @staticmethod
    def detect(text: str) -> LanguageProfile:
        """Detect language from text content using Unicode heuristics."""
        has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in text)
        has_gujarati = any(0x0A80 <= ord(c) <= 0x0AFF for c in text)
        if has_devanagari:
            return HindiProfile()
        if has_gujarati:
            return GujaratiProfile()
        return EnglishProfile()

    @staticmethod
    def detect_from_prompt(prompt: str) -> LanguageProfile:
        """Detect language from a user-facing prompt string (not the generated text)."""
        prompt_lower = prompt.lower()
        for indicator in HindiProfile.detection_indicators:
            if indicator in prompt_lower or indicator in prompt:
                return HindiProfile()
        for indicator in GujaratiProfile.detection_indicators:
            if indicator in prompt_lower or indicator in prompt:
                return GujaratiProfile()
        return EnglishProfile()
