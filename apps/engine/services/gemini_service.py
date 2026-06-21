import json
import logging
from typing import List

from django.conf import settings
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator

from apps.processor.services.language_profiles import ProfileManager

logger = logging.getLogger(__name__)


class TimelineSegment(BaseModel):
    chunk_index: int = Field(
        ...,
        description="Sequential index of the segment starting at 0"
    )
    text: str = Field(
        ...,
        description="The exact words spoken in this segment (7 to 9 words)"
    )
    visual_keyword: str = Field(
        ...,
        description=(
            "A unique, context-specific keyword for video search in ENGLISH ONLY. "
            "You MUST provide only 1 or 2 simple words for the visual keyword (e.g., \"robot\", \"city\", \"technology\"). "
            "DO NOT write Hindi, Gujarati, or any non-Latin script. DO NOT write long sentences or descriptive phrases, "
            "as this breaks the Pexels API search which requires English keywords."
        )
    )


class VideoScript(BaseModel):
    title: str = Field(
        ...,
        description="Catchy title of the video in ENGLISH ONLY regardless of the input prompt language."
    )
    seo_tags: List[str] = Field(
        ...,
        description="SEO keywords in ENGLISH ONLY regardless of the input prompt language."
    )
    voiceover_script: str = Field(
        ...,
        description=(
            "Full continuous text for TTS narration in the target language. "
            "CRITICAL UNICODE CONSTRAINT — You MUST output characters from the following Unicode ranges:\n"
            "  - For Hindi requests: ONLY Devanagari (U+0900–U+097F). NEVER output Arabic/Urdu (U+0600–U+06FF). "
            "Arabic/Urdu characters will cause TTS errors and are FORBIDDEN.\n"
            "  - For Gujarati requests: ONLY Gujarati script (U+0A80–U+0AFF).\n"
            "  - For English requests: ONLY Basic Latin (U+0020–U+007F).\n"
            "VIOLATING THESE CONSTRAINTS WILL BREAK THE ENTIRE PIPELINE."
        )
    )
    json_timeline: List[TimelineSegment] = Field(
        ...,
        description="List of visual timeline segments sync'd with the voiceover"
    )

    @field_validator('voiceover_script')
    @classmethod
    def validate_unicode_blocks(cls, v: str) -> str:
        """Reject any voiceover_script that violates the active profile's
        Unicode block constraints at the schema level."""
        profile = ProfileManager.detect(v)
        valid, msg = profile.validate_unicode(v)
        if not valid:
            raise ValueError(
                f"voiceover_script failed Unicode validation: {msg}. "
                f"Detected profile: {type(profile).__name__}. "
                f"The Gemini model hallucinated forbidden Unicode blocks. "
                f"Regenerate with strict block adherence."
            )
        return v


GEMINI_MODEL_FALLBACK_CHAIN = [
    'gemini-2.5-flash',
    'gemini-1.5-flash',
    'gemini-1.5-pro-latest',
]

_HINDI_INDICATORS = ["hindi", "हिंदी", "हिन्दी"]
_GUJARATI_INDICATORS = ["gujarati", "ગુજરાતી"]


def _build_unicode_enforced_prompt(base_prompt: str, format_type: str) -> str:
    """Build a system prompt with explicit, aggressive Unicode block constraints."""

    prompt_lower = base_prompt.lower()
    is_hindi = any(ind in prompt_lower or ind in base_prompt for ind in _HINDI_INDICATORS)
    is_gujarati = any(ind in prompt_lower or ind in base_prompt for ind in _GUJARATI_INDICATORS)

    language_rules = []
    if is_hindi:
        language_rules = [
            "LANGUAGE RULE — You are generating Hindi content.",
            "voiceover_script MUST use ONLY Devanagari script (Unicode U+0900–U+097F).",
            "ABSOLUTELY FORBIDDEN: Arabic script (U+0600–U+06FF), which is used for Urdu and Persian.",
            "Every single character in voiceover_script must be a valid Devanagari character.",
            "If you use any Arabic/Urdu character, the TTS engine will produce robotic, unintelligible audio.",
        ]
    elif is_gujarati:
        language_rules = [
            "LANGUAGE RULE — You are generating Gujarati content.",
            "voiceover_script MUST use ONLY Gujarati script (Unicode U+0A80–U+0AFF).",
            "Every single character in voiceover_script must be a valid Gujarati character.",
        ]
    else:
        language_rules = [
            "LANGUAGE RULE — You are generating English content.",
            "voiceover_script MUST use ONLY Basic Latin characters (Unicode U+0020–U+007F).",
        ]

    return (
        "You are an expert copywriter and video producer for a SaaS platform called OmniClip. "
        "Your goal is to take a base prompt/topic and generate a highly engaging structured video script "
        f"optimized for the target format: '{format_type}'. "
        "You must return a valid JSON object matching the following structure exactly:\n"
        "{\n"
        "  \"title\": \"Short catchy title of the video in English\",\n"
        "  \"seo_tags\": [\"english_keyword1\", \"english_keyword2\", \"english_keyword3\"],\n"
        "  \"voiceover_script\": \"Full, continuous, fluid text for Text-to-Speech narration in the target language\",\n"
        "  \"json_timeline\": [\n"
        "     {\n"
        "       \"chunk_index\": 0,\n"
        "       \"text\": \"The exact words spoken in this segment in the target language (7 to 9 words).\",\n"
        "       \"visual_keyword\": \"A unique, context-specific keyword for video search in ENGLISH ONLY.\"\n"
        "     }\n"
        "  ]\n"
        "}\n"
        "RULES — you must follow all of these precisely:\n"
        "1. Split voiceover_script into sequential segments of 7 to 9 words each.\n"
        + "\n".join(f"{r}" for r in language_rules) + "\n"
        "2. CRITICAL - The fields 'title', 'seo_tags', and 'visual_keyword' MUST ALWAYS be in English, "
        "regardless of the language of 'voiceover_script' or the input prompt. "
        "The Pexels API cannot search for Hindi or Gujarati words. Generate conceptual English keywords "
        "that represent the meaning of the Hindi/Gujarati text.\n"
        "3. Every visual_keyword must be 100% unique across all segments — no keyword may be reused.\n"
        "4. Generate exactly enough segments to cover the full voiceover_script.\n"
        "5. Do NOT include start_time or end_time fields — timestamps are computed from real audio.\n"
        "6. Do not return markdown syntax or backticks.\n"
        "7. Ensure all numbers are integers and the structure conforms to JSON specification.\n"
        f"\nUser Input Prompt: {base_prompt}"
    )


class GeminiService:
    @staticmethod
    def generate_script(base_prompt: str, format_type: str) -> dict:
        api_key = getattr(settings, 'GEMINI_API_KEY', None)
        if not api_key:
            logger.error("GEMINI_API_KEY is not configured in Django settings.")
            raise ValueError("GEMINI_API_KEY settings configuration is missing.")

        client = genai.Client(api_key=api_key)

        full_prompt = _build_unicode_enforced_prompt(base_prompt, format_type)
        last_exception = None

        for model_name in GEMINI_MODEL_FALLBACK_CHAIN:
            try:
                logger.info(
                    f"[GeminiService] Attempting script generation with model: {model_name} "
                    f"for topic: '{base_prompt[:50]}...'"
                )

                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=VideoScript,
                    )
                )

                parsed_data = json.loads(response.text)

                try:
                    validated = VideoScript.model_validate(parsed_data)
                    parsed_data = validated.model_dump()
                except Exception as val_err:
                    logger.error(
                        f"[GeminiService] Pydantic validation failed for model '{model_name}': {val_err}. "
                        f"Retrying with next fallback model..."
                    )
                    raise ValueError(
                        f"Script output failed Pydantic validation: {val_err}. "
                        f"This is retryable — requesting regeneration."
                    ) from val_err

                logger.info(
                    f"[GeminiService] Script generation successful with model: {model_name}. "
                    f"Timeline segments: {len(parsed_data.get('json_timeline', []))}"
                )
                return parsed_data

            except json.JSONDecodeError as exc:
                logger.error(
                    f"[GeminiService] Model '{model_name}' returned invalid JSON: {response.text[:200]}"
                )
                raise ValueError(
                    f"Gemini response parsing failed on model '{model_name}'. "
                    f"Raw response: {response.text}"
                ) from exc

            except Exception as exc:
                last_exception = exc
                logger.error(
                    f"[GeminiService] Model '{model_name}' failed with exception: {exc}",
                    exc_info=True
                )
                logger.warning(f"[GeminiService] Falling back to next model in chain...")
                continue

        logger.error(
            f"[GeminiService] All models in fallback chain exhausted. "
            f"Final error: {last_exception}"
        )
        raise ValueError(
            f"Gemini script generation failed across all fallback models "
            f"({', '.join(GEMINI_MODEL_FALLBACK_CHAIN)}). Last error: {last_exception}"
        )
