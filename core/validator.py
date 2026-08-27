"""
Validator - the second AI layer. Deterministic asserts can't judge 'are these
results relevant to the query?' — the LLM can. It receives the use-case
objective + validation rules + captured data and returns a verdict.
"""
import json
from core.nim_client import NimClient

VALIDATOR_SYSTEM = """You are a QA result validator. You receive:
1. The test objective
2. Validation rules in plain English
3. Data actually captured from the live site

Judge each rule strictly against the captured data. If data is missing or empty,
the rule fails. Prices arrive as strings like '₹1,299' - parse them numerically
when a rule involves price comparison or ordering.

Return JSON:
{
  "verdict": "PASS" | "FAIL",
  "rules": [
    {"rule": "<rule text>", "result": "PASS|FAIL", "evidence": "<one line citing the data>"}
  ],
  "summary": "<2 sentence human-readable summary>"
}"""


def validate(nim: NimClient, use_case: dict, execution_result: dict) -> dict:
    if execution_result["execution_status"] == "failed":
        return {
            "verdict": "FAIL",
            "rules": [],
            "summary": "Execution aborted before validation - see step log / failure.png",
        }
    payload = (
        f"Objective: {use_case['objective']}\n"
        f"Validation rules:\n" + "\n".join(f"- {r}" for r in use_case["validations"]) + "\n"
        f"Captured data:\n{json.dumps(execution_result['captured'], indent=2, ensure_ascii=False)[:6000]}"
    )
    return nim.chat_json(VALIDATOR_SYSTEM, payload, max_tokens=1500)
