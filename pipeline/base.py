"""Base class for all pipeline stages."""
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    pass


class BasePipelineStage(ABC):
    stage_name: str = "base"
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    def __init__(self, client, model: str = "claude-sonnet-4-20250514"):
        self.client = client
        self.model = model
        self.tokens_used = 0

    def _clean_json(self, text: str) -> str:
        """Remove markdown fences and extract JSON."""
        # Strip ```json ... ``` or ``` ... ```
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*", "", text)
        text = text.strip()

        # Try to find JSON object boundaries
        start = text.find("{")
        if start == -1:
            start = text.find("[")
        end = text.rfind("}")
        end_arr = text.rfind("]")

        if start == -1:
            return text

        # Pick the right end
        if end == -1 and end_arr == -1:
            return text[start:]
        if end == -1:
            return text[start:end_arr + 1]
        if end_arr == -1:
            return text[start:end + 1]
        return text[start:max(end, end_arr) + 1]

    def _call_llm(self, prompt: str, system: Optional[str] = None) -> str:
        """Make a single LLM call and track tokens."""
        messages = [{"role": "user", "content": prompt}]
        kwargs = {
            "model": self.model,
            "max_tokens": 8000,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        usage = response.usage
        self.tokens_used += usage.input_tokens + usage.output_tokens
        return response.content[0].text

    def _repair_json(self, broken_text: str) -> dict:
        """Ask LLM to fix broken JSON."""
        logger.warning(f"{self.stage_name}: attempting JSON repair")
        repair_prompt = (
            "The following text should be valid JSON but has syntax errors. "
            "Fix it and return ONLY valid JSON, no explanations, no markdown:\n\n"
            + broken_text[:4000]
        )
        fixed = self._call_llm(repair_prompt)
        cleaned = self._clean_json(fixed)
        return json.loads(cleaned)

    def _execute_with_retry(self, prompt: str) -> dict:
        """Execute LLM call with retry logic."""
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                raw = self._call_llm(prompt)
                return self.parse_response(raw)
            except json.JSONDecodeError as e:
                logger.warning(f"{self.stage_name} attempt {attempt}: JSON parse error: {e}")
                last_error = e
                # Try to repair on last attempt
                if attempt == self.MAX_RETRIES:
                    try:
                        return self._repair_json(raw)
                    except Exception as repair_err:
                        logger.error(f"{self.stage_name}: JSON repair failed: {repair_err}")
                        raise PipelineError(
                            f"{self.stage_name} failed after {self.MAX_RETRIES} attempts: {e}"
                        ) from repair_err
            except PipelineError as e:
                logger.warning(f"{self.stage_name} attempt {attempt}: {e}")
                last_error = e
            except Exception as e:
                logger.error(f"{self.stage_name} attempt {attempt} unexpected error: {e}")
                last_error = e

            if attempt < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY * attempt)

        raise PipelineError(
            f"{self.stage_name} failed after {self.MAX_RETRIES} attempts. Last: {last_error}"
        )

    @abstractmethod
    def parse_response(self, response_text: str) -> dict:
        pass

    @abstractmethod
    def run(self, *args, **kwargs) -> dict:
        pass
