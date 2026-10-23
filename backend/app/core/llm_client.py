"""
Gemini LLM Client wrapper with Pydantic Schema enforcement and retry mechanism.
Supports:
- Gemini Flash (gemini-2.5-flash / gemini-1.5-flash) for routine tutor/evaluation
- Gemini Pro (gemini-1.5-pro) for complex multi-concept diagnosis
- Offline high-fidelity fallback when API key is not configured
- Max 2 retries upon schema validation errors
"""

import os
import json
import logging
from typing import Type, TypeVar, Optional, Any, Dict
from pydantic import BaseModel, ValidationError

logger = logging.getLogger("VidyaSaathi.LLM")

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """Manages Gemini calls with structured JSON output and schema validation."""

    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.has_live_key = bool(self.api_key and len(self.api_key.strip()) > 5)
        self.client = None

        if self.has_live_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info("Initialized Google GenAI live client.")
            except Exception as e:
                logger.warning(f"Failed to initialize google-genai client: {e}. Falling back to internal engine.")
                self.has_live_key = False

    def generate_structured(
        self,
        prompt: str,
        system_instruction: str,
        response_schema: Type[T],
        model: str = "gemini-2.5-flash",
        max_retries: int = 2,
        fallback_factory: Optional[callable] = None
    ) -> T:
        """
        Invokes LLM and parses result into `response_schema`. Retries up to `max_retries` times.
        """
        if not self.has_live_key:
            if fallback_factory:
                return fallback_factory()
            raise RuntimeError("Live LLM API key not configured and no fallback factory provided.")

        last_error = None
        current_prompt = prompt

        for attempt in range(max_retries + 1):
            try:
                from google.genai import types

                response = self.client.models.generate_content(
                    model=model,
                    contents=current_prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=response_schema,
                        temperature=0.3
                    )
                )

                raw_text = response.text.strip()
                # Validate against Pydantic schema
                validated_obj = response_schema.model_validate_json(raw_text)
                return validated_obj

            except (ValidationError, Exception) as err:
                last_error = err
                logger.warning(f"Attempt {attempt + 1} failed schema validation or generation: {err}")
                current_prompt = (
                    f"{prompt}\n\n[ATTENTION: Previous attempt failed validation with error: {str(err)}. "
                    f"Please output strictly valid JSON conforming to the requested schema.]"
                )

        if fallback_factory:
            logger.info("Retries exhausted; invoking fallback factory.")
            return fallback_factory()

        raise RuntimeError(f"Failed to generate valid output after {max_retries + 1} attempts: {last_error}")
