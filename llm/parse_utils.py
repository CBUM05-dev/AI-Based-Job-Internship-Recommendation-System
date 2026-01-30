# Right now, your LLM returns raw text JSON.
# You should never trust LLM output blindly.

import json
import logging

logger = logging.getLogger(__name__)

def safe_parse_llm_output(llm_output: str) -> dict:
    try:
        parsed = json.loads(llm_output)
        return {
            "skills": parsed.get("skills", []),
            "level": parsed.get("level", "beginner"),
            "mode": parsed.get("mode", "remote"),
            "domain": parsed.get("domain", "other"),
        }
    except Exception as e:
        logger.error(f"Failed to parse LLM output: {e}")
        return {
            "skills": [],
            "level": "beginner",
            "mode": "remote",
            "domain": "other",
        }


    """
    This protects from:

Broken JSON

Hallucinated text

API glitches
    
    """
