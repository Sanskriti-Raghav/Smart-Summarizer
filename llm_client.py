import json
import os
import re
from typing import Dict, Any

import google.generativeai as genai
from dotenv import load_dotenv

from prompts import build_prompt


class GeminiClient:
    def __init__(self):
        # Load from .env once per process
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not found. Create a .env file and set GEMINI_API_KEY=your_key_here"
            )
        genai.configure(api_key=api_key)

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        """
        Robustly extract the first JSON object from a possibly noisy LLM response.
        """
        if text is None:
            raise ValueError("Empty response from model.")

        # Common fences cleanup
        text = text.strip()
        text = re.sub(r"^```json\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        text = re.sub(r"^```\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except Exception:
            pass

        # Fallback: find first {...} block
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidate = match.group(0)
            try:
                return json.loads(candidate)
            except Exception:
                # Soft repair: replace single quotes with double quotes when safe
                candidate2 = re.sub(r"(?<!\\)'", '"', candidate)
                return json.loads(candidate2)

        raise ValueError("Could not parse JSON from model output.")

    def generate_sections(
        self, text: str, model: str, summary_words: int, takeaways_count: int
    ) -> Dict[str, Any]:
        """
        Calls Gemini with a strict JSON-only instruction and returns:
        {
          "summary": "...",
          "key_takeaways": ["...", "..."]
        }
        Will retry once if takeaways count mismatches.
        """
        prompt = build_prompt(
            text=text,
            summary_words=summary_words,
            takeaways_count=takeaways_count,
        )

        def _call() -> Dict[str, Any]:
            llm = genai.GenerativeModel(model)
            resp = llm.generate_content(prompt)
            content = resp.text if hasattr(resp, "text") else None
            data = self._extract_json(content)
            # Normalize
            if "key_takeaways" in data and data["key_takeaways"] is None:
                data["key_takeaways"] = []
            return data

        data = _call()

        # Validate count of key takeaways; retry once if mismatch
        if takeaways_count is not None and takeaways_count >= 0:
            kt = data.get("key_takeaways", [])
            if not isinstance(kt, list):
                kt = []
            if len(kt) != takeaways_count:
                # Retry once
                data = _call()
                kt = data.get("key_takeaways", [])
                if not isinstance(kt, list):
                    kt = []
                # If still mismatched, truncate/pad locally to be safe
                if len(kt) < takeaways_count:
                    kt = kt + [""] * (takeaways_count - len(kt))
                elif len(kt) > takeaways_count:
                    kt = kt[:takeaways_count]
                data["key_takeaways"] = kt

        # Ensure required keys exist
        data.setdefault("summary", "")
        data.setdefault("key_takeaways", [])
        return data


# Convenience function required by spec
_client_singleton = None

def generate_sections(text: str, model: str, summary_words: int, takeaways_count: int) -> dict:
    global _client_singleton
    if _client_singleton is None:
        _client_singleton = GeminiClient()
    return _client_singleton.generate_sections(
        text=text,
        model=model,
        summary_words=summary_words,
        takeaways_count=takeaways_count,
    )
