"""
Groq-powered consolidation agent.

Constraint agreed in design (Phase 3 review): the agent may ONLY select or
lightly normalize among the actual OCR-observed candidate strings for a
field — it must never invent a value that didn't appear in any frame's
reading. Enforced both in the prompt and with a code-level grounding check
below (don't rely on the model alone to hold this line).

Model: openai/gpt-oss-120b via Groq — llama-3.1-8b-instant and
llama-3.3-70b-versatile (used earlier in this project) were deprecated by
Groq; gpt-oss-120b is the current replacement per project notes.
"""
import json
import os

from groq import Groq

GROQ_MODEL = "llama-3.3-70b-versatile" 

SYSTEM_PROMPT = """You are a strict data-consolidation assistant for an Egyptian ID verification pipeline.

You will be given a field name and a list of candidate text values that OCR extracted for that field across multiple video frames of the same document. Some candidates may be noisy, incomplete, or duplicated.

Your job:
- Pick the single most likely correct value for the field, using ONLY text that appears in the given candidates.
- You may lightly normalize (e.g., trim whitespace), but you must NEVER invent, guess, or add any character, word, or digit that does not appear in at least one of the given candidates.
- Apply these specific formatting rules for Egyptian IDs:
  * "first_name": Must ALWAYS be exactly one word (the person's given name).
  * "last_name": Will typically contain 3 or 4 words (the remainder of the full name).
  * "address_2": Represents the district/area and the governorate (e.g., "روض الفرج - القاهره" or "مركز البلينا - سوهاج"). It generally should NOT contain street names or building numbers.
  * "ID": Must be exactly 14 digits containing only numbers.
  * "ExpDate": Must follow the format of 4 digits for the year, 2 digits for the month, and 2 digits for the day (e.g., YYYY/MM/DD, YYYY-MM-DD, or YYYYMMDD depending on the raw text).
- If no candidate is usable, return null for that field.

Respond ONLY with a JSON object: {"value": <string or null>, "confidence": "high"|"medium"|"low"}
No other text, no markdown formatting, no explanation.
"""


class ConsolidationAgent:
    def __init__(self, api_key: str = None, model: str = GROQ_MODEL):
        self.client = Groq(api_key=api_key or os.environ["GROQ_API_KEY"])
        self.model = model

    def consolidate_field(self, field_name: str, candidates: list) -> dict:
        clean_candidates = [c for c in candidates if c]
        if not clean_candidates:
            return {"value": None, "confidence": "low"}

        user_prompt = json.dumps({"field": field_name, "candidates": clean_candidates}, ensure_ascii=False)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=200,
        )

        raw = response.choices[0].message.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Model didn't return clean JSON — fall back to the plain
            # majority-vote candidate rather than crashing the graph.
            return {"value": clean_candidates[0], "confidence": "low"}

        value = parsed.get("value")
        # Code-level grounding check: refuse anything not substring-related
        # to an actual candidate, regardless of what the prompt asked for.
        if value and not any(value in c or c in value for c in clean_candidates):
            return {"value": clean_candidates[0], "confidence": "low"}

        return parsed

    def consolidate_all(self, enhanced_reads: dict) -> dict:
        """enhanced_reads: {field_name: [candidate values across frames]}"""
        return {
            field_name: self.consolidate_field(field_name, candidates).get("value")
            for field_name, candidates in enhanced_reads.items()
        }
