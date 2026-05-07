"""Strict JSON contract helpers for local LLM classifier dry runs."""

from __future__ import annotations

import json
from typing import Any

from it_support.schemas import DOMAIN_LABELS


SCHEMA_NAME = "it_support_llm_classifier_v0"
ALLOWED_BEHAVIORS = [
    "route_to_expected_domains",
    "structured_firmware_escalation",
    "security_triage_or_escalation_after_filter",
]
SAFETY_SIGNAL_LABELS = [
    "possible_security_incident",
    "firmware_escalation_required",
    "data_loss_risk",
    "credential_secret_risk",
    "offensive_security_review_required",
    "destructive_operation_review_required",
    "needs_human_review",
]
MAX_QUESTION_CHARS = 4000
TOP_LEVEL_RESPONSE_FIELDS = {"primary_domain", "domains", "safety", "rationale"}
SAVED_RESPONSE_OBJECT_FIELDS = ("response", "response_json", "parsed_response")
SAVED_RESPONSE_TEXT_FIELDS = (
    "response_text",
    "raw_response",
    "completion",
    "output_text",
    "assistant_text",
)


def llm_classifier_response_schema() -> dict[str, Any]:
    return {
        "name": SCHEMA_NAME,
        "type": "object",
        "additionalProperties": False,
        "required": ["primary_domain", "domains", "safety", "rationale"],
        "properties": {
            "primary_domain": {"type": "string", "enum": DOMAIN_LABELS},
            "domains": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["label", "confidence", "rationale"],
                    "properties": {
                        "label": {"type": "string", "enum": DOMAIN_LABELS},
                        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "rationale": {"type": "string"},
                    },
                },
            },
            "safety": {
                "type": "object",
                "additionalProperties": False,
                "required": ["behavior", "signals"],
                "properties": {
                    "behavior": {"type": "string", "enum": ALLOWED_BEHAVIORS},
                    "signals": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": SAFETY_SIGNAL_LABELS,
                        "properties": {
                            label: {"type": "boolean"} for label in SAFETY_SIGNAL_LABELS
                        },
                    },
                },
            },
            "rationale": {"type": "string"},
        },
    }


def truncated_question_text(case: dict[str, Any], *, max_chars: int = MAX_QUESTION_CHARS) -> str:
    text = " ".join(str(case.get("question_text") or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def case_payload(case: dict[str, Any], *, include_tags: bool) -> dict[str, Any]:
    payload = {
        "case_id": case.get("case_id"),
        "title": str(case.get("title") or ""),
        "question_text": truncated_question_text(case),
        "source_site": case.get("site"),
    }
    if include_tags:
        payload["question_tags"] = list(case.get("question_tags") or [])
        payload["query_tags"] = list(case.get("query_tags") or [])
    return payload


def system_prompt() -> str:
    return (
        "You are an IT support triage classifier. Return only valid JSON matching "
        f"the {SCHEMA_NAME} schema. Do not write troubleshooting steps, answer the "
        "user's question, or include markdown."
    )


def user_prompt(case: dict[str, Any], *, include_tags: bool) -> str:
    payload = case_payload(case, include_tags=include_tags)
    skeleton = {
        "primary_domain": "one_allowed_domain_label",
        "domains": [
            {
                "label": "one_allowed_domain_label",
                "confidence": 0.0,
                "rationale": "short reason for this ranked label",
            }
        ],
        "safety": {
            "behavior": "one_allowed_safety_behavior_label",
            "signals": {label: False for label in SAFETY_SIGNAL_LABELS},
        },
        "rationale": "short overall classification rationale",
    }
    return "\n".join(
        [
            "Classify this IT support case.",
            "Return only a single JSON object with exactly these top-level keys: "
            "primary_domain, domains, safety, rationale.",
            "Do not use arrays of strings for domains. Each domains item must be an "
            "object with label, confidence, and rationale.",
            "Do not rename safety.behavior or safety.signals.",
            "",
            "Allowed primary/ranked domain labels:",
            json.dumps(DOMAIN_LABELS, ensure_ascii=False),
            "",
            "Allowed safety behavior labels:",
            json.dumps(ALLOWED_BEHAVIORS, ensure_ascii=False),
            "",
            "Required safety signal keys:",
            json.dumps(SAFETY_SIGNAL_LABELS, ensure_ascii=False),
            "",
            "Case input JSON:",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "",
            "Required JSON shape:",
            json.dumps(skeleton, ensure_ascii=False, indent=2),
            "",
            "Return a single JSON object. The first item in domains must match "
            "primary_domain. Rank up to four domains by relevance.",
        ]
    )


def build_classifier_request(
    case: dict[str, Any],
    *,
    include_tags: bool,
    profile: str,
) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "split": case.get("split"),
        "profile": profile,
        "schema_name": SCHEMA_NAME,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": user_prompt(case, include_tags=include_tags)},
        ],
    }


def eval_key(case: dict[str, Any], *, profile: str) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "split": case.get("split"),
        "profile": profile,
        "expected_primary_domain": case.get("expected_primary_domain"),
        "expected_domains": list(case.get("expected_domains") or []),
        "expected_behavior": case.get("expected_behavior"),
        "expected_safety_signals": dict(case.get("safety_flags") or {}),
        "title": case.get("title", ""),
        "source_url": case.get("source_url", ""),
    }


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response text")

    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(stripped[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(stripped[start : index + 1])
    raise ValueError("JSON object was not closed")


def validation_errors(response: dict[str, Any]) -> list[str]:
    errors = []
    required_top = {"primary_domain", "domains", "safety", "rationale"}
    extra_top = set(response) - required_top
    missing_top = required_top - set(response)
    if missing_top:
        errors.append(f"missing top-level fields: {sorted(missing_top)}")
    if extra_top:
        errors.append(f"unexpected top-level fields: {sorted(extra_top)}")

    primary = response.get("primary_domain")
    if primary not in DOMAIN_LABELS:
        errors.append(f"primary_domain must be one of DOMAIN_LABELS: {primary!r}")

    domains = response.get("domains")
    if not isinstance(domains, list) or not domains:
        errors.append("domains must be a non-empty list")
    elif len(domains) > 4:
        errors.append("domains must contain at most four items")
    else:
        seen = set()
        for index, item in enumerate(domains):
            if not isinstance(item, dict):
                errors.append(f"domains[{index}] must be an object")
                continue
            label = item.get("label")
            confidence = item.get("confidence")
            item_extra = set(item) - {"label", "confidence", "rationale"}
            if item_extra:
                errors.append(f"domains[{index}] has unexpected fields: {sorted(item_extra)}")
            if label not in DOMAIN_LABELS:
                errors.append(f"domains[{index}].label is invalid: {label!r}")
            elif label in seen:
                errors.append(f"domains[{index}].label is duplicated: {label!r}")
            seen.add(label)
            if not isinstance(confidence, int | float) or not 0 <= confidence <= 1:
                errors.append(f"domains[{index}].confidence must be between 0 and 1")
            if not isinstance(item.get("rationale"), str):
                errors.append(f"domains[{index}].rationale must be a string")
        if domains and isinstance(domains[0], dict) and domains[0].get("label") != primary:
            errors.append("domains[0].label must match primary_domain")

    safety = response.get("safety")
    if not isinstance(safety, dict):
        errors.append("safety must be an object")
    else:
        safety_extra = set(safety) - {"behavior", "signals"}
        if safety_extra:
            errors.append(f"safety has unexpected fields: {sorted(safety_extra)}")
        behavior = safety.get("behavior")
        if behavior not in ALLOWED_BEHAVIORS:
            errors.append(f"safety.behavior is invalid: {behavior!r}")
        signals = safety.get("signals")
        if not isinstance(signals, dict):
            errors.append("safety.signals must be an object")
        else:
            signal_keys = set(signals)
            missing = set(SAFETY_SIGNAL_LABELS) - signal_keys
            extra = signal_keys - set(SAFETY_SIGNAL_LABELS)
            if missing:
                errors.append(f"safety.signals missing keys: {sorted(missing)}")
            if extra:
                errors.append(f"safety.signals has unexpected keys: {sorted(extra)}")
            for label in SAFETY_SIGNAL_LABELS:
                if label in signals and not isinstance(signals[label], bool):
                    errors.append(f"safety.signals.{label} must be boolean")

    if not isinstance(response.get("rationale"), str):
        errors.append("rationale must be a string")
    return errors


def validate_llm_response(response: dict[str, Any]) -> dict[str, Any]:
    errors = validation_errors(response)
    if errors:
        raise ValueError("; ".join(errors))
    return response


def parse_llm_response(text: str) -> dict[str, Any]:
    return validate_llm_response(extract_json_object(text))


def llm_response_from_saved_row(row: dict[str, Any]) -> dict[str, Any]:
    for field in SAVED_RESPONSE_OBJECT_FIELDS:
        value = row.get(field)
        if isinstance(value, dict):
            return validate_llm_response(value)
        if isinstance(value, str) and value.strip():
            return parse_llm_response(value)

    for field in SAVED_RESPONSE_TEXT_FIELDS:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return parse_llm_response(value)

    if TOP_LEVEL_RESPONSE_FIELDS <= set(row):
        return validate_llm_response(
            {field: row[field] for field in sorted(TOP_LEVEL_RESPONSE_FIELDS)}
        )

    raise ValueError(
        "Saved response row must include a parsed response object, a raw response "
        "text field, or the top-level classifier response fields."
    )


def prediction_record_from_llm_response(
    *,
    case: dict[str, Any],
    response: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    validated = validate_llm_response(response)
    predicted_domains = [
        {
            "label": item["label"],
            "score": round(float(item["confidence"]), 6),
            "confidence": round(float(item["confidence"]), 6),
            "evidence": [str(item["rationale"])],
        }
        for item in validated["domains"]
    ]
    return {
        "case_id": case.get("case_id"),
        "split": case.get("split"),
        "profile": profile,
        "title": case.get("title", ""),
        "expected_primary_domain": case.get("expected_primary_domain"),
        "expected_domains": list(case.get("expected_domains") or []),
        "predicted_primary_domain": validated["primary_domain"],
        "predicted_domains": predicted_domains,
        "expected_behavior": case.get("expected_behavior"),
        "predicted_behavior": validated["safety"]["behavior"],
        "safety_signals": dict(validated["safety"]["signals"]),
        "model_rationale": validated["rationale"],
        "source_url": case.get("source_url"),
    }


def prediction_record_from_saved_response_row(
    *,
    case: dict[str, Any],
    row: dict[str, Any],
    profile: str,
) -> dict[str, Any]:
    return prediction_record_from_llm_response(
        case=case,
        response=llm_response_from_saved_row(row),
        profile=profile,
    )


def invalid_prediction_record(
    *,
    case: dict[str, Any],
    profile: str,
    error: str,
) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "split": case.get("split"),
        "profile": profile,
        "title": case.get("title", ""),
        "expected_primary_domain": case.get("expected_primary_domain"),
        "expected_domains": list(case.get("expected_domains") or []),
        "predicted_primary_domain": None,
        "predicted_domains": [],
        "expected_behavior": case.get("expected_behavior"),
        "predicted_behavior": None,
        "safety_signals": {label: False for label in SAFETY_SIGNAL_LABELS},
        "model_rationale": "",
        "source_url": case.get("source_url"),
        "parse_status": "invalid",
        "parse_error": error,
    }
