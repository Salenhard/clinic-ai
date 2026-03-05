"""Base class for all pipeline stages."""
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Optional

from google.genai import types as genai_types

from .rate_limiter import get_limiter

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    pass


class RateLimitError(PipelineError):
    pass


class BasePipelineStage(ABC):
    stage_name: str = "base"
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    RATE_LIMIT_BACKOFF = 65  # seconds — slightly over 1-minute window

    def __init__(
        self,
        client,            # google.genai.Client instance
        model: str = "gemini-2.0-flash",
        requests_per_minute: int = 15,
    ):
        self.client = client
        self.model = model
        self.tokens_used = 0
        self._limiter = get_limiter(requests_per_minute)

    # ── JSON helpers ──────────────────────────────────────────────────────────

    def _clean_json(self, text: str) -> str:
        """Remove markdown fences and extract JSON."""
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        start = text.find("{")
        if start == -1:
            start = text.find("[")
        end = text.rfind("}")
        end_arr = text.rfind("]")

        if start == -1:
            return text
        if end == -1 and end_arr == -1:
            return text[start:]
        if end == -1:
            return text[start:end_arr + 1]
        if end_arr == -1:
            return text[start:end + 1]
        return text[start:max(end, end_arr) + 1]

    # ── LLM call ──────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, system: Optional[str] = None) -> str:
        """
        Make a single call via google-genai SDK (google.genai.Client).
        - Acquires a rate-limit slot BEFORE the request.
        - On 429 / ResourceExhausted backs off RATE_LIMIT_BACKOFF seconds.
        """
        self._limiter.acquire()

        config = genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=8192,
            system_instruction=system if system else None,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except Exception as exc:
            exc_str = str(exc).lower()
            if "429" in exc_str or "resource_exhausted" in exc_str or "quota" in exc_str:
                logger.warning(
                    f"{self.stage_name}: quota exceeded — "
                    f"backing off {self.RATE_LIMIT_BACKOFF}s ..."
                )
                time.sleep(self.RATE_LIMIT_BACKOFF)
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
            else:
                raise

        # Track token usage
        meta = getattr(response, "usage_metadata", None)
        if meta:
            self.tokens_used += getattr(meta, "prompt_token_count", 0)
            self.tokens_used += getattr(meta, "candidates_token_count", 0)

        return response.text

    # ── JSON repair ───────────────────────────────────────────────────────────

    def _repair_json(self, broken_text: str) -> dict:
        """Ask LLM to fix broken JSON (counts as one additional request)."""
        logger.warning(f"{self.stage_name}: attempting JSON repair")
        repair_prompt = (
            "The following text should be valid JSON but has syntax errors. "
            "Fix it and return ONLY valid JSON, no explanations, no markdown:\n\n"
            + broken_text[:4000]
        )
        fixed = self._call_llm(repair_prompt)
        cleaned = self._clean_json(fixed)
        return json.loads(cleaned)

    # ── Execute with retry ────────────────────────────────────────────────────

    def _execute_with_retry(self, prompt: str) -> dict:
        """Execute LLM call with retry logic.

        Retries on JSON parse errors (up to MAX_RETRIES).
        Rate-limit 429s are handled inside _call_llm transparently.
        """
        last_error = None
        raw = ""

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw = self._call_llm(prompt)
                return self.parse_response(raw)

            except json.JSONDecodeError as e:
                logger.warning(
                    f"{self.stage_name} attempt {attempt}/{self.MAX_RETRIES}: "
                    f"JSON parse error -- {e}"
                )
                last_error = e
                if attempt == self.MAX_RETRIES:
                    try:
                        return self._repair_json(raw)
                    except Exception as repair_err:
                        logger.error(f"{self.stage_name}: JSON repair failed: {repair_err}")
                        raise PipelineError(
                            f"{self.stage_name} failed after {self.MAX_RETRIES} attempts: {e}"
                        ) from repair_err

            except PipelineError as e:
                logger.warning(f"{self.stage_name} attempt {attempt}/{self.MAX_RETRIES}: {e}")
                last_error = e

            except Exception as e:
                logger.error(
                    f"{self.stage_name} attempt {attempt}/{self.MAX_RETRIES} unexpected: {e}"
                )
                last_error = e

            if attempt < self.MAX_RETRIES:
                delay = self.RETRY_DELAY * attempt
                logger.info(f"{self.stage_name}: retrying in {delay}s ...")
                time.sleep(delay)

        raise PipelineError(
            f"{self.stage_name} failed after {self.MAX_RETRIES} attempts. Last: {last_error}"
        )

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def parse_response(self, response_text: str) -> dict:
        pass

    @abstractmethod
    def run(self, *args, **kwargs) -> dict:
        pass
