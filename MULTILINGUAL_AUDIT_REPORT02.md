# Multilingual Video Generation Pipeline Audit Report (Version 02)

Deep-dive technical analysis of the OmniClip video processing pipeline for non-Latin script handling (Devanagari Hindi, Gujarati).

---

## 1. Pexels Stock Footage Irrelevance (Keyword Translation Failure)

### The Exact Bug in the System

When generating videos for Hindi or Gujarati topics, `visual_keyword` values are generated in the target language script and passed directly to the Pexels API without English translation. For example, if Gemini generates `"text": "तकनीक हमारे जीवन को बेहतर बनाती है"` (technology improves our lives), the corresponding `visual_keyword` becomes `"तकनीक"` (technology in Devanagari) instead of `"technology"`.

**Specific code locations:**
- `gemini_service.py:15` - The `visual_keyword` Field description does NOT explicitly mandate English output
- `gemini_service.py:73-74` - The rule states keywords must be "simple words" but does not specify language
- `gemini_service.py:66` - System instruction describes keywords as English examples but doesn't enforce the constraint

### Technical Reason ("The Why")

1. **Inference Without Explicit Constraint**: The Gemini model receives instructions like "You MUST provide only 1 or 2 simple words for the visual keyword (e.g., 'robot', 'city')" but the "e.g." examples can be interpreted as suggestions, not requirements. When the voiceover_script is in Hindi/Gujarati, the model infers that visual_keyword should match the script language.

2. **Pexels API Localization Deficiency**: Pexels video search index has minimal support for Devanagari (U+0900-U+097F) and Gujarati (U+0A80-U+0AFF) Unicode ranges. Searching for `"राम"` (Ram) or `"મહેશ્વર"` (Maheshwar) returns culturally-tagged content (temples, festivals) rather than conceptual matches (technology, science, business).

3. **No Language Boundary in Prompt**: The `system_instruction` at `gemini_service.py:53-79` does not contain any directive distinguishing between text (regional script) and keywords (English required). The model treats both fields uniformly.

### Proposed Step-by-Step Code Solution

**Step 1**: Modify the `TimelineSegment` Pydantic schema to mandate English keywords:

```python
# apps/engine/services/gemini_service.py
class TimelineSegment(BaseModel):
    chunk_index: int = Field(..., description="Sequential index of the segment starting at 0")
    text: str = Field(..., description="The exact words spoken in this segment (7 to 9 words)")
    visual_keyword: str = Field(
        ..., 
        description="A unique, context-specific keyword for video search in ENGLISH ONLY. "
                    "You MUST provide 1 or 2 simple English words (e.g., 'robot', 'city', 'technology'). "
                    "DO NOT write Hindi, Gujarati, or any non-Latin script. DO NOT write long sentences or descriptive phrases, "
                    "as this breaks the Pexels API search which requires English keywords."
    )
```

**Step 2**: Update the `system_instruction` in `generate_script()` to explicitly separate text language from keyword language:

```python
system_instruction = (
    "... "
    "RULES — you must follow all of these precisely:\n"
    "1. Split voiceover_script into sequential segments of 7 to 9 words each. "
    "   The 'text' field of each segment must be the exact words from that portion of voiceover_script.\n"
    "2. CRITICAL - Every visual_keyword MUST be in ENGLISH regardless of the language of 'text'. "
    "   The Pexels API cannot search for Hindi or Gujarati words. Generate conceptual English keywords "
    "   (e.g., 'technology', 'robot', 'business') that represent the meaning of the Hindi/Gujarati text.\n"
    "   Each visual_keyword must be 100% unique across all segments.\n"
    ...
)
```

---

## 2. Choppy and De-synchronized Subtitles

### The Exact Bug in the System

Subtitles in Hindi/GUJARATI videos appear jerky, freeze on screen, or lag behind the spoken voice. The root cause is in `tasks.py:168-185` where word timestamps are artificially reconstructed by uniform division instead of using actual Whisper-generated timestamps.

**Specific code locations:**
- `tasks.py:170-177` - Text is split on whitespace and timestamps are uniformly distributed
- `tasks.py:179` - `word_dur = (seg_end - seg_start) / len(seg_words)` formula ignores actual speech rhythm
- `whisper_service.py:198` - Words are lowercased and stripped in `_extract_words()`, but this is not the primary issue
- `whisper_service.py:217-218` - The `tokenize()` function uses `text.lower()` which works for Devanagari/GUJARATI but doesn't account for the visual display needs

### Technical Reason ("The Why")

1. **Discarded Whisper Word Timestamps**: In `whisper_service.py:305`, `WhisperAlignmentService.align()` calls `_align_segments()` which returns enriched timeline with segment-level timestamps. However, the actual per-word timestamps extracted at line 300 are discarded and never returned to the caller.

2. **Uniform Distribution Invalidates Rhythm**: Hindi and Gujarati syllables have variable lengths. A single Devanagari syllable like `"संगीत"` (music) may be spoken in ~0.3s, while a conjunct consonant like `"क्रमशः"` (sequentially) takes ~0.6s. Uniform division assumes equal durations, causing:
   - Short words to appear frozen (lingering beyond their actual speech)
   - Long words to be cut off (subtitle ends before speech completes)

3. **Whitespace Splitting Issues**: At `tasks.py:175`, Devanagari text is split on space:
   ```python
   seg_words = seg_text.split()  # "नमस्ते गुजरात" → ["नमस्ते", "गुजरात"]
   ```
   This works for modern Hindi/GUJARATI which uses spaces, but the lowercasing at `whisper_service.py:198` (`w.word.strip().lower()`) can cause issues with certain Unicode characters that don't have case mappings.

4. **No Micro-Chunk Floor Enforcement**: The `_MIN_DURATION_S = 0.4` at `whisper_service.py:16` applies to Whisper-aligned words, but the artificial division in `tasks.py` can still produce chunks shorter than this floor because it doesn't check for minimum duration after uniform assignment.

### Proposed Step-by-Step Code Solution

**Step 1**: Modify `WhisperAlignmentService.align()` to return both aligned timeline AND raw word timestamps:

```python
# whisper_service.py - modify return signature
@classmethod
def align(cls, audio_path: str, timeline: list) -> tuple[list, list]:  # Return tuple
    """
    Returns:
        tuple: (enriched_timeline, raw_words_timestamps)
    """
    ...
    return enriched, words  # Instead of just enriched
```

**Step 2**: Update `_uniform_fallback()` to generate synthetic word timestamps:

```python
# whisper_service.py
@staticmethod
def _uniform_fallback(timeline: list, total_duration: float = None) -> tuple[list, list]:
    """Return (enriched_timeline, synthetic_words)."""
    chunk_duration = (total_duration / len(timeline)) if total_duration and len(timeline) else 3.5
    enriched = []
    all_words = []
    for i, chunk in enumerate(timeline):
        seg_text = chunk.get('text', '')
        seg_words = seg_text.split()
        word_dur = chunk_duration / max(len(seg_words), 1) if seg_words else chunk_duration
        for wi, w in enumerate(seg_words):
            all_words.append({
                "word": w,
                "start": round(i * chunk_duration + wi * word_dur, 3),
                "end": round(i * chunk_duration + (wi + 1) * word_dur, 3),
            })
        enriched.append({
            **chunk,
            "start_time": round(i * chunk_duration, 3),
            "end_time": round((i + 1) * chunk_duration, 3),
        })
    return enriched, all_words
```

**Step 3**: Modify `tasks.py` to use real Whisper timestamps instead of artificial division:

```python
# tasks.py - replace lines 168-196
try:
    # NEW: Get real word timestamps from WhisperAlignmentService
    aligned_timeline, real_word_timestamps = WhisperAlignmentService.align(
        audio_path, raw_timeline
    )
    
    # Use actual Whisper timestamps directly instead of uniform division
    subtitle_chunks = sentence_aware_chunk(real_word_timestamps)
    ...
```

---

## 3. FFmpeg libass Subtitle Rendering Glitches (Matra/Ligature Overlaps)

### The Exact Bug in the System

Complex Indic scripts suffer from:
1. Vowel signs (matras) rendering detached from base consonants
2. Conjunct consonants ( ligatures) overlapping or clipping incorrectly
3. Assamese/Gujarati character clusters causing subtitle layer overlaps on short durations

**Specific code locations:**
- `ffmpeg_service.py:42-56` - Three-layer rendering with fixed font "Arial Black"
- `ffmpeg_service.py:54-56` - `_ASS_LAYERS` applies blur and transitions to all text uniformly
- `ffmpeg_service.py:81-82` - Bounce animation tag `\fscx110\fscy110\t(0,{_BOUNCE_ENTRY_MS},\fscx100\fscy100)` applied regardless of script complexity
- `ffmpeg_service.py:147-172` - All layers rendered with same timing; no discrimination for complex scripts

### Technical Reason ("The Why")

1. **Font Substitution Chain**: "Arial Black" lacks Devanagari (U+0900 block) and Gujarati (U+0A80 block) glyph tables. FFmpeg libass must substitute fonts at render time, selecting from system fonts like "Nirmala UI" or "Mangal". The substitution occurs AFTER the ASS file is parsed, meaning:
   - Layer 0 (glow) renders with font A's metrics
   - Layer 1 (shadow) renders with font B's metrics (slightly different)
   - Layer 2 (text) renders with font C's metrics
   This causes misalignment between layers, manifesting as double-images or ghosting.

2. **HarfBuzz Shaping Under Animation Stress**: The transition tag `\t(0,80,\fscx100\fscy100)` instructs libass to interpolate font scale from 110% to 100% over 80ms. For Devanagari, HarfBuzz must:
   - Resolve the Halant + Consonant sequence to form conjuncts
   - Position matras (U+0900-U+094F) relative to base glyphs
   This shaping operation is computationally expensive. At 80ms duration, the shaper may not complete before frame deadline, causing:
   - Unpositioned matras appearing at origin (0,0), overlapping text
   - Incomplete shaping causing glyph collisions
   - Short-duration words (0.1-0.3s) experiencing repeated shape recalculations

3. **Minimum Duration Race Condition**: The `_MIN_DURATION_S = 0.4` constant at `whisper_service.py:16` was chosen to prevent imperceptible flashes in Latin scripts. For Indic languages, shorter durations (7+ words per segment) can still produce subtitle chunks under 0.4s when combined with uniform division. These micro-durations with bounce animations create a perfect storm for libass rendering failures.

4. **Collision Handling Ignored**: At `ffmpeg_service.py:115`, `"Collisions": "Ignore"` is set, meaning overlapping subtitle events don't resolve conflicts. Combined with layer misalignment from font substitution, this causes subtitle text to visually overwrite itself.

### Proposed Step-by-Step Code Solution

**Step 1**: Add script detection helper in `ffmpeg_service.py`:

```python
# ffmpeg_service.py - add after imports
def _contains_indic_script(text: str) -> bool:
    """Detect Devanagari or Gujarati Unicode characters."""
    return any(
        0x0900 <= ord(c) <= 0x097F  # Devanagari block
        or 0x0A80 <= ord(c) <= 0x0AFF  # Gujarati block
        for c in text
    )
```

**Step 2**: Modify `_generate_ass_subtitles()` to use appropriate fonts and disable animations:

```python
# ffmpeg_service.py - inside _generate_ass_subtitles()
# Detect if any chunk contains Indic script
has_indic = any(_contains_indic_script((" ".join(c.words) if hasattr(c, 'words') else " ".join(c.get('words', []))) for c in subtitle_chunks)

# Choose font based on script
font_name = "Nirmala UI" if has_indic else _ASS_FONT_NAME

# Disable bounce animation for Indic scripts
bounce_tag = "" if has_indic else f"{{\\fscx110\\fscy110\\t(0,{_BOUNCE_ENTRY_MS},\\fscx100\\fscy100)}}"

# Rebuild header with dynamic font
header = header.replace(_ASS_FONT_NAME, font_name)
```

**Step 3**: Add minimum duration floor for short subtitle chunks:

```python
# ffmpeg_service.py - after chunk creation
# Ensure minimum 0.5s duration for Indic script chunks to allow shaping
if has_indic and (end_time - start_time) < 0.5:
    # Extend end time to minimum floor - but only for display, keep original for FFmpeg
    chunk_end_ts = _ass_ts(max(end_time, start_time + 0.5))
```

---

## 4. TTS EdgeTTS Fallback Failures for Regional Languages

### The Exact Bug in the System

When ElevenLabs fails or `voice_id` is unrecognized, the system falls back to EdgeTTS with the default English voice `"en-US-ChristopherNeural"`, even when the script text contains Hindi or Gujarati characters. This causes EdgeTTS to either:
1. Speak English phonetics for Devanagari characters
2. Produce partial/no audio output
3. Drop entire words that cannot be mapped

**Specific code locations:**
- `audio_service.py:112-113` - Voice lookup falls back to `_DEFAULT_EDGE_VOICE` without text inspection
- `audio_service.py:57-58` - Default voices are English-only
- `audio_service.py:47-54` - Voice map includes Hindi voices but only for known keys; unknown keys fall through to English

### Technical Reason ("The Why")

1. **Missing Text-Language Correlation**: At `audio_service.py:112`, the lookup `VOICE_MAP.get(voice_id, (_DEFAULT_ELEVENLABS_VOICE, _DEFAULT_EDGE_VOICE))` uses the voice_id as the primary key. If an unknown voice_id is provided, or if the ElevenLabs-specific voice ID (like `JBFqnCBsd6RMkjVDRZzb`) is passed:
   - Line 112 returns the fallback tuple `(_DEFAULT_ELEVENLABS_VOICE, _DEFAULT_EDGE_VOICE)` = `("pNInz6obpgDQGcFmaJgB", "en-US-ChristopherNeural")`
   - The EdgeTTS voice becomes English regardless of script content

2. **EdgeTTS Voice Compatibility**: The `hi-IN-MadhurNeural` and `gu-IN-NiranjanNeural` voices are capable of proper Devanagari/Gujarati speech. However, sending them to `en-US-ChristopherNeural` causes:
   - The Azure Neural engine attempts IPA (International Phonetic Alphabet) fallback
   - Devanagari characters fail IPA conversion → silent tokens
   - Gujarati script contains no recognizable Latin graphemes → entire word dropped

3. **No Graceful Degradation Path**: Even when Hindi voices are correctly mapped (`VOICE_MAP["hindi_male"]` = `("JBFqnCBsd6RMkjVDRZzb", "hi-IN-MadhurNeural")`), if the EdgeTTS subprocess call fails (network, timeout), the exception handler at `audio_service.py:147-151` doesn't distinguish between:
   - English text with English voice failure (retryable)
   - Hindi text with English voice fallback (unrecoverable without text-aware rerouting)

### Proposed Step-by-Step Code Solution

**Step 1**: Add language detection and default voices in `audio_service.py`:

```python
# audio_service.py - add after line 58
def _detect_script_language(text: str) -> str:
    """Detect if text contains Hindi, Gujarati, or English script."""
    has_devanagari = any(0x0900 <= ord(c) <= 0x097F for c in text)
    has_gujarati = any(0x0A80 <= ord(c) <= 0x0AFF for c in text)
    if has_devanagari:
        return "hindi"
    if has_gujarati:
        return "gujarati"
    return "english"

_DEFAULT_HINDI_EDGE_VOICE = "hi-IN-MadhurNeural"
_DEFAULT_GUJARATI_EDGE_VOICE = "gu-IN-NiranjanNeural"
```

**Step 2**: Modify `EdgeTTSProvider.generate()` to audit text language:

```python
# audio_service.py - modify EdgeTTSProvider.generate()
@staticmethod
def generate(text: str, voice_id: str, output_wav_path: str) -> None:
    # Detect script language BEFORE voice lookup
    detected_lang = _detect_script_language(text)
    
    # Get voice from map or determine appropriate fallback
    mapping = VOICE_MAP.get(voice_id)
    if mapping:
        _, edge_voice = mapping
    else:
        # Intelligent fallback based on detected script
        if detected_lang == "hindi":
            edge_voice = _DEFAULT_HINDI_EDGE_VOICE
        elif detected_lang == "gujarati":
            edge_voice = _DEFAULT_GUJARATI_EDGE_VOICE
        else:
            edge_voice = _DEFAULT_EDGE_VOICE
    
    # ... rest of method unchanged ...
```

**Step 3**: Add retry logic for wrong voice/script combination:

```python
# audio_service.py - in generate_tts_audio() exception handler
except RuntimeError as exc:
    logger.warning(f"[HybridTTS] ElevenLabs provider failed: {exc}. Falling back to EdgeTTS...")
    try:
        EdgeTTSProvider.generate(script_text, voice_id, output_path)
        return url_path
    except RuntimeError as edge_exc:
        # If text is Indic and error suggests voice mismatch, retry with correct voice
        if _detect_script_language(script_text) != "english" and "voice" in str(edge_exc).lower():
            detected_lang = _detect_script_language(script_text)
            correct_voice = _DEFAULT_HINDI_EDGE_VOICE if detected_lang == "hindi" else _DEFAULT_GUJARATI_EDGE_VOICE
            logger.info(f"[HybridTTS] Retrying with {correct_voice} for {detected_lang} text")
            EdgeTTSProvider.generate(script_text, correct_voice, output_path)
            return url_path
        raise edge_exc
```

---

## 5. Additional Critical Bug: Word Timestamp Lowercasing Corruption

### The Exact Bug in the System

At `whisper_service.py:198`, all words are lowercased:
```python
"word": w.word.strip().lower(),
```
While Devanagari and Gujarati characters typically don't have case mappings, certain Unicode characters in extended Devanagari (Sanskrit extensions) or mixed-script scenarios can cause `.lower()` to produce unexpected output or even raise exceptions.

### Technical Reason ("The Why")

1. **Python `.lower()` Behavior on Indic Scripts**: For codepoints like U+0900-U+097F (Devanagari) and U+0A80-U+0AFF (Gujarati), `.lower()` is typically a no-op. However, for rare extended characters or when combining Latin and Indic text, the lowercasing can cause:
   - Unicode normalization side-effects
   - Different character forms that don't match the original TTS output
   - The alignment algorithm failing to find matching tokens

2. **Alignment Token Mismatch**: The `_align_segments()` tokenize function at `whisper_service.py:217-218` uses `text.lower()`. Combined with the lowercasing in `_extract_words()`, this creates a situation where:
   - TTS speaks "नमस्ते" (namaste)
   - Whisper returns "नमस्ते" with lowercase no-op
   - But downstream comparisons (if any) may behave unexpectedly

### Proposed Step-by-Step Code Solution

Remove unnecessary lowercasing for Indic scripts:

```python
# whisper_service.py - modify _extract_words()
@staticmethod
def _extract_words(segments) -> list:
    def _is_indic(word: str) -> bool:
        return any(
            0x0900 <= ord(c) <= 0x097F or 0x0A80 <= ord(c) <= 0x0AFF
            for c in word
        )
    
    words = []
    for seg in segments:
        if hasattr(seg, 'words') and seg.words:
            for w in seg.words:
                raw_word = w.word.strip()
                # Preserve Indic script as-is to avoid Unicode normalization issues
                processed_word = raw_word if _is_indic(raw_word) else raw_word.lower()
                words.append({
                    "word": processed_word,
                    "start": w.start,
                    "end": w.end,
                })
    return words
```

---

## Summary of Critical Fixes Required

| Bug | File | Critical Line | Fix Priority |
|-----|------|---------------|--------------|
| Pexels keyword language mismatch | gemini_service.py | Line 15, 73-74 | High |
| Subtitle timestamp artificial division | tasks.py | Lines 170-196 | High |
| libass font/ligature glitches | ffmpeg_service.py | Lines 42, 81, 147 | High |
| TTS voice fallback without script detection | audio_service.py | Lines 112-113 | Medium |
| Word lowercasing for Indic scripts | whisper_service.py | Line 198 | Medium |

---

*Report compiled by the Lead Backend Architect & Code Auditor.  
Review required before applying any fixes.*