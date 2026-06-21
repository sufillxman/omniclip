"""
tests_e2e_multilingual.py — Autonomous E2E test suite for multilingual pipeline.

Tests:
  1. LanguageProfile isolation (no cross-contamination between EN/HI/GU)
  2. ProfileManager.detect() Unicode-based language detection
  3. ProfileManager.detect_from_prompt() prompt-based detection
  4. SSML wrapping via profile.wrap_ssml() — language tags, prosody, breaks
  5. Unicode validation via profile.validate_unicode() — reject/allow blocks
  6. Poll: EnglishProfile correctly rejects Devanagari/Gujarati
  7. Sentence-aware chunking with profile-driven constants
  8. HindiProfile correctly rejects Arabic/Urdu (U+0600–U+06FF)
  9. GeminiService VideoScript Pydantic validator triggers on bad Unicode

Run:  python -m pytest tests_e2e_multilingual.py -v
      or
      python manage.py test tests_e2e_multilingual -v2  (if Django settings allow)
"""

import re
import pytest
from typing import List, Tuple


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LanguageProfile Isolation — verify each profile carries its OWN rules
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("profile_class, expected", [
    ("EnglishProfile", {
        "tts_voice_id": "en-US-ChristopherNeural",
        "tts_rate": "+0%",
        "tts_pitch": "+0Hz",
        "whisper_max_words": 9,
        "whisper_max_duration": 2.5,
        "whisper_min_duration": 0.3,
        "ffmpeg_font_name": "Arial Black",
        "ffmpeg_bounce_enabled": True,
        "ass_margin_v": 100,
        "ass_font_size": 90,
    }),
    ("HindiProfile", {
        "tts_voice_id": "hi-IN-MadhurNeural",
        "tts_rate": "-5%",
        "tts_pitch": "+0Hz",
        "whisper_max_words": 6,
        "whisper_max_duration": 2.2,
        "whisper_min_duration": 0.5,
        "ffmpeg_font_name": "Nirmala UI",
        "ffmpeg_bounce_enabled": False,
        "ass_margin_v": 120,
        "ass_font_size": 85,
    }),
    ("GujaratiProfile", {
        "tts_voice_id": "gu-IN-NiranjanNeural",
        "tts_rate": "-5%",
        "tts_pitch": "+0Hz",
        "whisper_max_words": 6,
        "whisper_max_duration": 2.2,
        "whisper_min_duration": 0.5,
        "ffmpeg_font_name": "Nirmala UI",
        "ffmpeg_bounce_enabled": False,
        "ass_margin_v": 120,
        "ass_font_size": 85,
    }),
])
def test_language_profile_isolation(profile_class, expected):
    """Each LanguageProfile subclass carries its OWN constants — no cross-contamination."""
    from apps.processor.services.language_profiles import (
        EnglishProfile, HindiProfile, GujaratiProfile
    )
    cls_map = {
        "EnglishProfile": EnglishProfile,
        "HindiProfile": HindiProfile,
        "GujaratiProfile": GujaratiProfile,
    }
    profile = cls_map[profile_class]()
    for attr, val in expected.items():
        assert getattr(profile, attr) == val, (
            f"{profile_class}.{attr} expected {val!r}, got {getattr(profile, attr)!r}"
        )


def test_english_profile_has_no_rejected_blocks():
    from apps.processor.services.language_profiles import EnglishProfile
    profile = EnglishProfile()
    assert profile.rejected_unicode_blocks == []


def test_hindi_profile_rejects_arabic_block():
    from apps.processor.services.language_profiles import HindiProfile
    profile = HindiProfile()
    assert (0x0600, 0x06FF) in profile.rejected_unicode_blocks


def test_gujarati_profile_no_rejected_blocks():
    from apps.processor.services.language_profiles import GujaratiProfile
    profile = GujaratiProfile()
    assert profile.rejected_unicode_blocks == []


# ═══════════════════════════════════════════════════════════════════════════════
# 2. ProfileManager.detect() — Unicode-heuristic text detection
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("text, expected_class_name", [
    ("Hello world, this is English text.", "EnglishProfile"),
    ("नमस्ते दुनिया यह हिंदी पाठ है।", "HindiProfile"),
    ("નમસ્તે દુનિયા આ ગુજરાતી લખાણ છે.", "GujaratiProfile"),
    ("Mix of English and नमस्ते", "HindiProfile"),
    ("Mix of English and નમસ્તે", "GujaratiProfile"),
    ("12345 !@#$%", "EnglishProfile"),
    ("", "EnglishProfile"),
])
def test_profile_manager_detect(text, expected_class_name):
    from apps.processor.services.language_profiles import ProfileManager
    from apps.processor.services.language_profiles import (
        EnglishProfile, HindiProfile, GujaratiProfile
    )
    cls_map = {
        "EnglishProfile": EnglishProfile,
        "HindiProfile": HindiProfile,
        "GujaratiProfile": GujaratiProfile,
    }
    profile = ProfileManager.detect(text)
    assert isinstance(profile, cls_map[expected_class_name]), (
        f"detect({text!r}) returned {type(profile).__name__}, expected {expected_class_name}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ProfileManager.detect_from_prompt() — prompt-based detection
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("prompt, expected_class_name", [
    ("Create a video about technology", "EnglishProfile"),
    ("Create a Hindi video about nature", "HindiProfile"),
    ("Create a Gujarati video about culture", "GujaratiProfile"),
    ("हिंदी में एक वीडियो बनाओ", "HindiProfile"),
    ("ગુજરાતીમાં એક વિડિયો બનાવો", "GujaratiProfile"),
    ("Hindi and Gujarati mix", "HindiProfile"),  # hindi detected first
])
def test_profile_manager_detect_from_prompt(prompt, expected_class_name):
    from apps.processor.services.language_profiles import ProfileManager
    from apps.processor.services.language_profiles import (
        EnglishProfile, HindiProfile, GujaratiProfile
    )
    cls_map = {
        "EnglishProfile": EnglishProfile,
        "HindiProfile": HindiProfile,
        "GujaratiProfile": GujaratiProfile,
    }
    profile = ProfileManager.detect_from_prompt(prompt)
    assert isinstance(profile, cls_map[expected_class_name]), (
        f"detect_from_prompt({prompt!r}) returned {type(profile).__name__}, expected {expected_class_name}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Unicode Validation — profile.validate_unicode()
# ═══════════════════════════════════════════════════════════════════════════════

def test_hindi_validate_pure_devanagari():
    from apps.processor.services.language_profiles import HindiProfile
    profile = HindiProfile()
    text = "नमस्ते दुनिया यह पूरी तरह से देवनागरी है।"
    valid, msg = profile.validate_unicode(text)
    assert valid, f"Pure Devanagari should pass: {msg}"


def test_hindi_rejects_arabic_urdu():
    from apps.processor.services.language_profiles import HindiProfile
    profile = HindiProfile()
    text = "نستے دuniya यह मिला हुआ है"  # Arabic chars mixed in
    valid, msg = profile.validate_unicode(text)
    assert not valid, "HindiProfile must reject Arabic/Urdu characters"


def test_hindi_rejects_pure_arabic():
    from apps.processor.services.language_profiles import HindiProfile
    profile = HindiProfile()
    text = "مرحبا بالعالم"  # Pure Arabic
    valid, msg = profile.validate_unicode(text)
    assert not valid, "HindiProfile must reject pure Arabic text"


def test_gujarati_validate_pure_gujarati():
    from apps.processor.services.language_profiles import GujaratiProfile
    profile = GujaratiProfile()
    text = "નમસ્તે દુનિયા આ સંપૂર્ણ ગુજરાતી છે।"
    valid, msg = profile.validate_unicode(text)
    assert valid, f"Pure Gujarati should pass: {msg}"


def test_english_validate_pure_english():
    from apps.processor.services.language_profiles import EnglishProfile
    profile = EnglishProfile()
    text = "Hello world, this is pure English text."
    valid, msg = profile.validate_unicode(text)
    assert valid, f"Pure English should pass: {msg}"


def test_english_rejects_devanagari():
    from apps.processor.services.language_profiles import EnglishProfile
    profile = EnglishProfile()
    text = "Hello नमस्ते world"
    valid, msg = profile.validate_unicode(text)
    assert not valid, "EnglishProfile must reject Devanagari"


def test_english_rejects_gujarati():
    from apps.processor.services.language_profiles import EnglishProfile
    profile = EnglishProfile()
    text = "Hello નમસ્તે world"
    valid, msg = profile.validate_unicode(text)
    assert not valid, "EnglishProfile must reject Gujarati"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Profile-driven sentence-aware chunking
# ═══════════════════════════════════════════════════════════════════════════════

def test_sentence_aware_chunk_respects_english_profile():
    """
    EnglishProfile has whisper_max_words=9. 11 words with no punctuation:
    the lookahead merges the trailing 2 words (≤2) into the first chunk
    to prevent orphans → 1 chunk of 11 words.
    """
    from apps.processor.services.whisper_service import sentence_aware_chunk
    from apps.processor.services.language_profiles import EnglishProfile

    profile = EnglishProfile()
    words = [
        {"word": f"word{i}", "start": i * 0.2, "end": (i + 1) * 0.2}
        for i in range(11)
    ]
    chunks = sentence_aware_chunk(words, profile=profile)
    assert len(chunks) == 1, f"Expected 1 chunk (lookahead merged orphans), got {len(chunks)}"
    assert len(chunks[0].words) == 11, f"All 11 words should merge, got {len(chunks[0].words)}"


def test_sentence_aware_chunk_respects_hindi_profile():
    """
    HindiProfile has whisper_max_words=6. 8 words with no punctuation:
    the lookahead merges the trailing 2 words (≤2) into the first chunk
    to prevent orphans → 1 chunk of 8 words.
    """
    from apps.processor.services.whisper_service import sentence_aware_chunk
    from apps.processor.services.language_profiles import HindiProfile

    profile = HindiProfile()
    words = [
        {"word": f"word{i}", "start": i * 0.2, "end": (i + 1) * 0.2}
        for i in range(8)
    ]
    chunks = sentence_aware_chunk(words, profile=profile)
    assert len(chunks) == 1, f"Expected 1 chunk (lookahead merged orphans), got {len(chunks)}"
    assert len(chunks[0].words) == 8, f"All 8 words should merge, got {len(chunks[0].words)}"


def test_sentence_aware_chunk_auto_detects_profile():
    """
    When no profile is provided, sentence_aware_chunk should auto-detect
    from the text content.
    """
    from apps.processor.services.whisper_service import sentence_aware_chunk

    # Pure English → EnglishProfile → 9 word max, 10 words → lookahead merges 1 orphan
    eng_words = [
        {"word": f"word{i}", "start": i * 0.2, "end": (i + 1) * 0.2}
        for i in range(10)
    ]
    eng_chunks = sentence_aware_chunk(eng_words)
    assert len(eng_chunks[0].words) == 10, "Lookahead should merge the 1 trailing word"

    # Hindi text → HindiProfile → 6 word max, 8 words → lookahead merges 2 orphans
    hin_words = [
        {"word": "नमस्ते", "start": 0.0, "end": 0.1},
        {"word": "दुनिया", "start": 0.1, "end": 0.2},
        {"word": "यह", "start": 0.2, "end": 0.3},
        {"word": "हिंदी", "start": 0.3, "end": 0.4},
        {"word": "पाठ", "start": 0.4, "end": 0.5},
        {"word": "है", "start": 0.5, "end": 0.6},
        {"word": "बहुत", "start": 0.6, "end": 0.7},
        {"word": "अच्छा", "start": 0.7, "end": 0.8},
    ]
    hin_chunks = sentence_aware_chunk(hin_words)
    assert len(hin_chunks[0].words) == 8, "Lookahead should merge the 2 trailing words"


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Gemini VideoScript Pydantic validator
# ═══════════════════════════════════════════════════════════════════════════════

def test_video_script_rejects_arabic_in_voiceover():
    """
    The Pydantic field_validator on VideoScript.voiceover_script must
    reject Arabic/Urdu characters in Hindi context.
    """
    from apps.engine.services.gemini_service import VideoScript

    # voiceover_script contains Arabic (U+0600-06FF) → should fail validation
    with pytest.raises(Exception) as exc_info:
        VideoScript(
            title="Test",
            seo_tags=["test"],
            voiceover_script="مرحبا नमस्ते",  # Arabic + Devanagari mix
            json_timeline=[
                {"chunk_index": 0, "text": "مرحبا नमस्ते", "visual_keyword": "hello"}
            ],
        )
    assert "Unicode" in str(exc_info.value)


def test_video_script_passes_pure_hindi():
    """Pure Devanagari voiceover_script must pass the validator."""
    from apps.engine.services.gemini_service import VideoScript

    script = VideoScript(
        title="Hindi Test",
        seo_tags=["test"],
        voiceover_script="नमस्ते दुनिया यह हिंदी में एक परीक्षण है",
        json_timeline=[
            {"chunk_index": 0, "text": "नमस्ते दुनिया", "visual_keyword": "world"}
        ],
    )
    assert script.voiceover_script is not None


def test_video_script_passes_pure_english():
    """Pure English voiceover_script must pass the validator."""
    from apps.engine.services.gemini_service import VideoScript

    script = VideoScript(
        title="English Test",
        seo_tags=["test"],
        voiceover_script="Hello world this is a pure English test",
        json_timeline=[
            {"chunk_index": 0, "text": "Hello world", "visual_keyword": "world"}
        ],
    )
    assert script.voiceover_script is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 8. FFmpeg ASS generation uses profile-driven fonts
# ═══════════════════════════════════════════════════════════════════════════════

def test_ass_subtitle_uses_english_font_for_english():
    from apps.processor.services.ffmpeg_service import _generate_ass_subtitles
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.ass', delete=False, mode='w') as f:
        ass_path = f.name
    try:
        chunks = [
            type('Chunk', (), {'words': ['Hello', 'world'], 'start_time': 0.0, 'end_time': 1.0})(),
        ]
        _generate_ass_subtitles(chunks, ass_path, layout='landscape')
        with open(ass_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "Arial Black" in content, "English subtitles should use Arial Black"
        assert "\\fscx110" in content, "English subtitles should have bounce animation"
    finally:
        import os
        os.remove(ass_path)


def test_ass_subtitle_uses_hindi_font_for_hindi():
    from apps.processor.services.ffmpeg_service import _generate_ass_subtitles
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.ass', delete=False, mode='w') as f:
        ass_path = f.name
    try:
        chunks = [
            type('Chunk', (), {'words': ['नमस्ते', 'दुनिया'], 'start_time': 0.0, 'end_time': 1.0})(),
        ]
        _generate_ass_subtitles(chunks, ass_path, layout='landscape')
        with open(ass_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "Nirmala UI" in content, "Hindi subtitles should use Nirmala UI"
        assert "\\fscx110" not in content, "Hindi subtitles should NOT have bounce animation"
    finally:
        import os
        os.remove(ass_path)


def test_ass_subtitle_uses_gujarati_font_for_gujarati():
    from apps.processor.services.ffmpeg_service import _generate_ass_subtitles
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.ass', delete=False, mode='w') as f:
        ass_path = f.name
    try:
        chunks = [
            type('Chunk', (), {'words': ['નમસ્તે', 'દુનિયા'], 'start_time': 0.0, 'end_time': 1.0})(),
        ]
        _generate_ass_subtitles(chunks, ass_path, layout='landscape')
        with open(ass_path, 'r', encoding='utf-8') as f:
            content = f.read()
        assert "Nirmala UI" in content, "Gujarati subtitles should use Nirmala UI"
        assert "\\fscx110" not in content, "Gujarati subtitles should NOT have bounce animation"
    finally:
        import os
        os.remove(ass_path)


def test_ass_subtitle_duration_adjustment_for_indic():
    """
    For Hindi/Gujarati, chunks shorter than whisper_min_duration (0.5s)
    should be extended to at least 0.5s.
    """
    from apps.processor.services.ffmpeg_service import _generate_ass_subtitles, _ass_ts
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.ass', delete=False, mode='w') as f:
        ass_path = f.name
    try:
        chunks = [
            type('Chunk', (), {'words': ['नमस्ते'], 'start_time': 0.0, 'end_time': 0.3})(),
        ]
        _generate_ass_subtitles(chunks, ass_path, layout='landscape')
        with open(ass_path, 'r', encoding='utf-8') as f:
            content = f.read()
        # 0.3s < 0.5s → should be extended: end_time becomes 0.5
        # ASS format: 0:00:00.00 -> H:MM:SS.cc
        from apps.processor.services.ffmpeg_service import _ass_ts
        assert _ass_ts(0.5) in content, "Duration should be extended to at least 0.5s"
    finally:
        import os
        os.remove(ass_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. E2E Pipeline Simulation — no cross-contamination
# ═══════════════════════════════════════════════════════════════════════════════

def test_full_pipeline_english_profile_has_correct_cadence():
    """
    Simulate the pipeline for English: profile-driven chunking should use
    9 max words and 2.5s max duration.
    """
    from apps.processor.services.language_profiles import EnglishProfile
    profile = EnglishProfile()
    assert profile.whisper_max_words == 9
    assert profile.whisper_max_duration == 2.5
    assert profile.ffmpeg_bounce_enabled is True
    assert profile.tts_voice_id == "en-US-ChristopherNeural"
    assert profile.ffmpeg_font_name == "Arial Black"


def test_full_pipeline_hindi_profile_has_correct_cadence():
    """
    Simulate the pipeline for Hindi: profile-driven chunking should use
    6 max words and 2.2s max duration, NO bounce, Nirmala UI font.
    """
    from apps.processor.services.language_profiles import HindiProfile
    profile = HindiProfile()
    assert profile.whisper_max_words == 6
    assert profile.whisper_max_duration == 2.2
    assert profile.ffmpeg_bounce_enabled is False
    assert profile.tts_voice_id == "hi-IN-MadhurNeural"
    assert profile.ffmpeg_font_name == "Nirmala UI"


def test_full_pipeline_gujarati_profile_has_correct_cadence():
    """
    Simulate the pipeline for Gujarati: profile-driven chunking should use
    6 max words and 2.2s max duration, NO bounce, Nirmala UI font.
    """
    from apps.processor.services.language_profiles import GujaratiProfile
    profile = GujaratiProfile()
    assert profile.whisper_max_words == 6
    assert profile.whisper_max_duration == 2.2
    assert profile.ffmpeg_bounce_enabled is False
    assert profile.tts_voice_id == "gu-IN-NiranjanNeural"
    assert profile.ffmpeg_font_name == "Nirmala UI"


def test_no_english_logic_bleeds_into_hindi():
    """Verify HindiProfile does NOT inherit English values."""
    from apps.processor.services.language_profiles import EnglishProfile, HindiProfile
    en = EnglishProfile()
    hi = HindiProfile()
    # These MUST differ
    assert hi.whisper_max_words != en.whisper_max_words
    assert hi.whisper_max_duration != en.whisper_max_duration
    assert hi.whisper_min_duration != en.whisper_min_duration
    assert hi.ffmpeg_bounce_enabled != en.ffmpeg_bounce_enabled
    assert hi.ffmpeg_font_name != en.ffmpeg_font_name
    assert hi.tts_voice_id != en.tts_voice_id


def test_no_english_logic_bleeds_into_gujarati():
    """Verify GujaratiProfile does NOT inherit English values."""
    from apps.processor.services.language_profiles import EnglishProfile, GujaratiProfile
    en = EnglishProfile()
    gu = GujaratiProfile()
    assert gu.whisper_max_words != en.whisper_max_words
    assert gu.whisper_max_duration != en.whisper_max_duration
    assert gu.whisper_min_duration != en.whisper_min_duration
    assert gu.ffmpeg_bounce_enabled != en.ffmpeg_bounce_enabled
    assert gu.ffmpeg_font_name != en.ffmpeg_font_name
    assert gu.tts_voice_id != en.tts_voice_id


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

def test_empty_string_detection():
    from apps.processor.services.language_profiles import ProfileManager, EnglishProfile
    profile = ProfileManager.detect("")
    assert isinstance(profile, EnglishProfile)


def test_mixed_scripts_detects_hindi_first():
    """When both Devanagari and Gujarati present, ProfileManager returns Hindi (checked first)."""
    from apps.processor.services.language_profiles import ProfileManager, HindiProfile
    text = "नमस्ते નમસ્તે"  # Hindi + Gujarati
    profile = ProfileManager.detect(text)
    assert isinstance(profile, HindiProfile)


def test_voice_id_fallbacks_mapping_present():
    """Verify VOICE_MAP still has all required entries."""
    from apps.processor.services.audio_service import VOICE_MAP
    assert "hi-IN-MadhurNeural" in VOICE_MAP
    assert "gu-IN-NiranjanNeural" in VOICE_MAP
    assert "en-US-ChristopherNeural" in VOICE_MAP
