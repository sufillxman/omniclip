import json
import logging
from django.conf import settings
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List

logger = logging.getLogger(__name__)

# Define Pydantic models for structured output schema
class TimelineSegment(BaseModel):
    chunk_index: int = Field(..., description="Sequential index of the segment starting at 0")
    text: str = Field(..., description="The exact words spoken in this segment (7 to 9 words)")
    visual_keyword: str = Field(..., description="A unique, context-specific keyword for video search")

class VideoScript(BaseModel):
    title: str = Field(..., description="Catchy title of the video")
    seo_tags: List[str] = Field(..., description="SEO keywords")
    voiceover_script: str = Field(..., description="Full continuous text for TTS narration")
    json_timeline: List[TimelineSegment] = Field(..., description="List of visual timeline segments sync'd with the voiceover")

# Sequential fallback chain — primary model first, then progressively more stable fallbacks
GEMINI_MODEL_FALLBACK_CHAIN = [
    'gemini-2.5-flash',
    'gemini-1.5-flash',
    'gemini-1.5-pro-latest',
]


class GeminiService:
    @staticmethod
    def generate_script(base_prompt: str, format_type: str) -> dict:
        """
        Calls the Google Gemini API to generate a structured video script.
        Iterates through GEMINI_MODEL_FALLBACK_CHAIN sequentially — if the primary model
        throws an API error or quota exception, the next fallback model is tried immediately.
        Raises ValueError only after all models in the chain are exhausted.

        NOTE: Gemini only outputs scene descriptions and keywords. It does NOT output
        timestamps. Real start/end times are computed post-TTS by WhisperAlignmentService
        using actual audio word-level alignment.
        """
        api_key = getattr(settings, 'GEMINI_API_KEY', None)
        if not api_key:
            logger.error("GEMINI_API_KEY is not configured in Django settings.")
            raise ValueError("GEMINI_API_KEY settings configuration is missing.")

        # Initialize the GenAI Client
        client = genai.Client(api_key=api_key)

        # ── Audio-Driven Sync system prompt ──────────────────────────────────
        system_instruction = (
            "You are an expert copywriter and video producer for a SaaS platform called OmniClip. "
            "Your goal is to take a base prompt/topic and generate a highly engaging structured video script "
            f"optimized for the target format: '{format_type}'. "
            "You must return a valid JSON object matching the following structure exactly:\n"
            "{\n"
            "  \"title\": \"Short catchy title of the video\",\n"
            "  \"seo_tags\": [\"keyword1\", \"keyword2\", \"keyword3\"],\n"
            "  \"voiceover_script\": \"Full, continuous, fluid text for Text-to-Speech narration without labels or actions.\",\n"
            "  \"json_timeline\": [\n"
            "     {\n"
            "       \"chunk_index\": 0,\n"
            "       \"text\": \"The exact words spoken in this segment (7 to 9 words).\",\n"
            "       \"visual_keyword\": \"A unique, highly context-specific search keyword for this exact clip.\"\n"
            "     }\n"
            "  ]\n"
            "}\n"
            "RULES — you must follow all of these precisely:\n"
            "1. Split voiceover_script into sequential segments of 7 to 9 words each. "
            "   The 'text' field of each segment must be the exact words from that portion of voiceover_script.\n"
            "2. Every visual_keyword must be 100% unique across all segments — no keyword may be "
            "   reused or duplicated. Each keyword must be specific to the content of that exact "
            "   text segment (e.g., 'astronaut floating in space', 'stock market chart rising', "
            "   'scientist in laboratory').\n"
            "3. Generate exactly enough segments to cover the full voiceover_script with no shortages.\n"
            "4. Do NOT include start_time or end_time fields — timestamps are computed from real audio.\n"
            "5. Do not return markdown syntax, wrapping blocks, or backticks like ```json.\n"
            "6. Ensure all numbers are integers and the structure conforms strictly to JSON specification."
        )

        full_prompt = f"System Instruction: {system_instruction}\n\nUser Input Prompt: {base_prompt}"
        last_exception = None

        # ── Sequential fallback loop ──────────────────────────────────────────
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
                logger.info(
                    f"[GeminiService] Script generation successful with model: {model_name}. "
                    f"Timeline segments: {len(parsed_data.get('json_timeline', []))}"
                )
                return parsed_data

            except json.JSONDecodeError as exc:
                # JSON parse failure is a hard error — raw response logged, no retry benefit
                logger.error(
                    f"[GeminiService] Model '{model_name}' returned invalid JSON: {response.text[:200]}"
                )
                raise ValueError(
                    f"Gemini response parsing failed on model '{model_name}'. "
                    f"Raw response: {response.text}"
                ) from exc

            except Exception as exc:
                # API error, quota exceeded, model unavailable — try next fallback
                last_exception = exc
                logger.error(
                    f"[GeminiService] Model '{model_name}' failed with EXACT exception details: {exc}",
                    exc_info=True
                )
                logger.warning(
                    f"[GeminiService] Falling back to next model in chain..."
                )
                continue

        # All models exhausted — raise the last recorded exception
        logger.error(
            f"[GeminiService] All models in fallback chain exhausted. "
            f"Final error: {last_exception}"
        )
        raise ValueError(
            f"Gemini script generation failed across all fallback models "
            f"({', '.join(GEMINI_MODEL_FALLBACK_CHAIN)}). Last error: {last_exception}"
        )
