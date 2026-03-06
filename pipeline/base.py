"""Base class for all pipeline stages."""
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import List, Optional

from google.genai import types as genai_types

from .rate_limiter import get_limiter
from .chunker import Chunk

logger = logging.getLogger(__name__)

class RateLimiterError(Exception):
    pass

class PipelineError(Exception):
    pass


class BasePipelineStage(ABC):
    stage_name: str = "base"
    MAX_RETRIES = 10
    RETRY_DELAY = 2

    RATE_LIMIT_BACKOFF = 65  # seconds

    def __init__(
        self,
        client,
        model: str = "gemini-3.1-flash-lite-preview",
        requests_per_minute: int = 15,
    ):
        self.client = client
        self.model = model
        self.tokens_used = 0
        self._limiter = get_limiter(requests_per_minute)

    # ── JSON helpers ──────────────────────────────────────────────────────────

    def _clean_json(self, text: str) -> str:
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

    # ── Single LLM call ───────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, system: Optional[str] = None) -> str:
        self._limiter.acquire()

        config = genai_types.GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=8192,
            system_instruction=system if system else None,
        )
        try:
            response = self.client.models.generate_content_stream(
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
                response = self.client.models.generate_content_stream(
                    model=self.model,
                    contents=prompt,
                    config=config,
                )
            else:
                raise

        meta = getattr(response, "usage_metadata", None)
        if meta:
            self.tokens_used += getattr(meta, "prompt_token_count", 0)
            self.tokens_used += getattr(meta, "candidates_token_count", 0)
        result = ""
        for chunk in response:
            result += chunk.text
        return result

    # ── JSON repair ───────────────────────────────────────────────────────────

    def _repair_json(self, broken_text: str) -> dict:
        logger.warning(f"{self.stage_name}: attempting JSON repair")
        fixed = self._call_llm(
            "The following text should be valid JSON but has syntax errors. "
            "Fix it and return ONLY valid JSON, no explanations, no markdown:\n\n"
            + broken_text[:4000]
        )
        return json.loads(self._clean_json(fixed))

    # ── Retry wrapper ─────────────────────────────────────────────────────────

    def _execute_with_retry(self, prompt: str) -> dict:
        last_error = None
        raw = ""
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw = self._call_llm(prompt)
                return self.parse_response(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"{self.stage_name} attempt {attempt}/{self.MAX_RETRIES}: JSON error — {e}")
                last_error = e
                if attempt == self.MAX_RETRIES:
                    try:
                        return self._repair_json(raw)
                    except Exception as re_err:
                        raise PipelineError(f"{self.stage_name}: JSON repair failed: {re_err}") from re_err
            except PipelineError as e:
                logger.warning(f"{self.stage_name} attempt {attempt}/{self.MAX_RETRIES}: {e}")
                last_error = e
            except Exception as e:
                logger.error(f"{self.stage_name} attempt {attempt}/{self.MAX_RETRIES} unexpected: {e}")
                last_error = e
            if attempt < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY * attempt)

        raise PipelineError(f"{self.stage_name} failed after {self.MAX_RETRIES} attempts. Last: {last_error}")

    # ── Chunked execution ─────────────────────────────────────────────────────

    def _execute_over_chunks(
        self,
        chunks: List[Chunk],
        build_prompt_fn,          # (chunk_text, chunk_index, total_chunks) -> str
        merge_fn,                 # (list[dict]) -> dict
    ) -> dict:
        """
        Run the stage prompt over each chunk independently, then merge results.

        Parameters
        ----------
        chunks          : list of Chunk objects
        build_prompt_fn : callable(chunk_text, chunk_index, total_chunks) → prompt str
        merge_fn        : callable(list[dict]) → merged dict
        """
        if len(chunks) == 1:
            # Fast path — no merging needed
            return self._execute_with_retry(
                build_prompt_fn(chunks[0].text, 0, 1)
            )

        results = []
        for chunk in chunks:
            logger.info(
                f"{self.stage_name}: processing chunk {chunk.index + 1}/{len(chunks)} "
                f"({chunk.char_count:,} chars)"
            )
            prompt = build_prompt_fn(chunk.text, chunk.index, len(chunks))
            try:
                result = self._execute_with_retry(prompt)
                results.append(result)
            except PipelineError as e:
                logger.error(f"{self.stage_name} chunk {chunk.index} failed: {e}")
                # Continue with remaining chunks — don't abort entire stage

        if not results:
            raise PipelineError(f"{self.stage_name}: all chunks failed")

        merged = merge_fn(results)
        logger.info(f"{self.stage_name}: merged {len(results)} chunk results")
        return merged

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def parse_response(self, response_text: str) -> dict:
        pass

    @abstractmethod
    def run(self, *args, **kwargs) -> dict:
        pass
