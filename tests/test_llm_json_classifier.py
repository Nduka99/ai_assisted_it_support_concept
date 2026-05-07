from __future__ import annotations

import pytest

from it_support.classification.llm_json_classifier import (
    SAFETY_SIGNAL_LABELS,
    build_classifier_request,
    llm_response_from_saved_row,
    parse_llm_response,
    prediction_record_from_saved_response_row,
    prediction_record_from_llm_response,
    validate_llm_response,
)


def _case() -> dict:
    return {
        "case_id": "c1",
        "split": "routing_dev_eval",
        "title": "WiFi drops after driver update",
        "question_text": "The adapter disconnects every few minutes.",
        "question_tags": ["wireless", "drivers"],
        "query_tags": ["wireless"],
        "site": "askubuntu",
        "expected_primary_domain": "network_connectivity",
        "expected_domains": ["network_connectivity", "os_kernel_drivers"],
        "expected_behavior": "route_to_expected_domains",
        "source_url": "https://example.test/q/1",
    }


def _response() -> dict:
    return {
        "primary_domain": "network_connectivity",
        "domains": [
            {
                "label": "network_connectivity",
                "confidence": 0.82,
                "rationale": "Wireless connectivity is the main failure.",
            },
            {
                "label": "os_kernel_drivers",
                "confidence": 0.51,
                "rationale": "The driver update is relevant.",
            },
        ],
        "safety": {
            "behavior": "route_to_expected_domains",
            "signals": {label: False for label in SAFETY_SIGNAL_LABELS},
        },
        "rationale": "Route to networking first and OS/drivers second.",
    }


def test_build_classifier_request_keeps_expected_labels_out_of_prompt_keys() -> None:
    request = build_classifier_request(
        _case(),
        include_tags=True,
        profile="llm_json_classifier_metadata_dry_run",
    )

    prompt_text = "\n".join(message["content"] for message in request["messages"])

    assert '"expected_primary_domain"' not in prompt_text
    assert '"expected_domains"' not in prompt_text
    assert '"expected_behavior"' not in prompt_text
    assert "question_tags" in prompt_text


def test_validate_llm_response_accepts_strict_contract() -> None:
    assert validate_llm_response(_response())["primary_domain"] == "network_connectivity"


def test_validate_llm_response_rejects_unknown_or_misordered_domain() -> None:
    response = _response()
    response["primary_domain"] = "hardware"

    with pytest.raises(ValueError, match="domains\\[0\\].label"):
        validate_llm_response(response)

    response = _response()
    response["domains"][0]["label"] = "made_up_domain"

    with pytest.raises(ValueError, match="invalid"):
        validate_llm_response(response)


def test_parse_llm_response_extracts_fenced_json() -> None:
    text = """
```json
{
  "primary_domain": "network_connectivity",
  "domains": [
    {
      "label": "network_connectivity",
      "confidence": 0.82,
      "rationale": "Wireless connectivity is the main failure."
    }
  ],
  "safety": {
    "behavior": "route_to_expected_domains",
    "signals": {
      "possible_security_incident": false,
      "firmware_escalation_required": false,
      "data_loss_risk": false,
      "credential_secret_risk": false,
      "offensive_security_review_required": false,
      "destructive_operation_review_required": false,
      "needs_human_review": false
    }
  },
  "rationale": "Route to networking."
}
```
"""

    assert parse_llm_response(text)["primary_domain"] == "network_connectivity"


def test_prediction_record_from_llm_response_matches_ladder_fields() -> None:
    record = prediction_record_from_llm_response(
        case=_case(),
        response=_response(),
        profile="llm_json_classifier_metadata",
    )

    assert record["predicted_primary_domain"] == "network_connectivity"
    assert [row["label"] for row in record["predicted_domains"]] == [
        "network_connectivity",
        "os_kernel_drivers",
    ]
    assert record["predicted_behavior"] == "route_to_expected_domains"
    assert record["safety_signals"]["needs_human_review"] is False


def test_saved_response_row_accepts_response_text_and_top_level_fields() -> None:
    response = _response()
    row_with_text = {
        "case_id": "c1",
        "response_text": (
            "extra preface "
            '{"primary_domain": "network_connectivity", "domains": ['
            '{"label": "network_connectivity", "confidence": 0.82, '
            '"rationale": "Wireless connectivity."}], "safety": '
            '{"behavior": "route_to_expected_domains", "signals": {'
            '"possible_security_incident": false, '
            '"firmware_escalation_required": false, '
            '"data_loss_risk": false, '
            '"credential_secret_risk": false, '
            '"offensive_security_review_required": false, '
            '"destructive_operation_review_required": false, '
            '"needs_human_review": false}}, "rationale": "Route to networking."}'
        ),
    }
    row_with_top_level = {"case_id": "c1", **response}

    assert llm_response_from_saved_row(row_with_text)["primary_domain"] == (
        "network_connectivity"
    )
    assert llm_response_from_saved_row(row_with_top_level)["primary_domain"] == (
        "network_connectivity"
    )


def test_prediction_record_from_saved_response_row_uses_saved_row_parser() -> None:
    record = prediction_record_from_saved_response_row(
        case=_case(),
        row={"case_id": "c1", "response": _response()},
        profile="llm_json_classifier_metadata",
    )

    assert record["predicted_primary_domain"] == "network_connectivity"
