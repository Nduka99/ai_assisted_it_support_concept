from __future__ import annotations

from it_support.classification.ladder import (
    multilabel_domain_report,
    primary_multiclass_report,
    ranked_domain_report,
    ranked_prediction_labels,
    safety_behavior_report,
    safety_signal_report,
)


def _routing_row(
    case_id: str,
    *,
    expected_primary: str,
    expected_domains: list[str],
    predicted_domains: list[str],
) -> dict:
    return {
        "case_id": case_id,
        "expected_primary_domain": expected_primary,
        "expected_domains": expected_domains,
        "predicted_primary_domain": predicted_domains[0] if predicted_domains else None,
        "predicted_domains": [
            {"label": label, "score": len(predicted_domains) - index}
            for index, label in enumerate(predicted_domains)
        ],
    }


def test_ranked_prediction_labels_deduplicates_in_order() -> None:
    row = {
        "predicted_domains": [
            {"label": "network_connectivity"},
            {"label": "hardware"},
            {"label": "network_connectivity"},
        ]
    }

    assert ranked_prediction_labels(row) == ["network_connectivity", "hardware"]


def test_routing_ladder_metrics_cover_primary_multilabel_and_ranked_views() -> None:
    rows = [
        _routing_row(
            "q1",
            expected_primary="network_connectivity",
            expected_domains=["network_connectivity", "hardware"],
            predicted_domains=["network_connectivity", "hardware"],
        ),
        _routing_row(
            "q2",
            expected_primary="hardware",
            expected_domains=["hardware"],
            predicted_domains=["network_connectivity"],
        ),
    ]
    labels = ["network_connectivity", "hardware"]

    primary, _, _ = primary_multiclass_report(rows, labels=labels, profile="p")
    multilabel, _ = multilabel_domain_report(rows, labels=labels, profile="p")
    ranked = ranked_domain_report(rows, profile="p", cutoffs=(1, 2))

    assert primary["accuracy"] == 0.5
    assert multilabel["exact_match_ratio"] == 0.5
    assert multilabel["micro_f1"] == 0.6667
    assert ranked["primary_hit_at_1"] == 0.5
    assert ranked["primary_hit_at_2"] == 0.5
    assert ranked["any_expected_hit_at_1"] == 0.5


def test_safety_behavior_report_uses_expected_behavior_labels() -> None:
    rows = [
        {"expected_behavior": "structured_firmware_escalation", "predicted_behavior": "structured_firmware_escalation"},
        {"expected_behavior": "security_triage_or_escalation_after_filter", "predicted_behavior": "route_to_expected_domains"},
    ]
    labels = [
        "security_triage_or_escalation_after_filter",
        "structured_firmware_escalation",
    ]

    summary, per_label, confusion = safety_behavior_report(rows, labels=labels, profile="p")

    assert summary["accuracy"] == 0.5
    assert summary["macro_f1"] == 0.5
    assert {row["expected_behavior"] for row in per_label} == set(labels)
    assert any(row["predicted_behavior"] == "route_to_expected_domains" for row in confusion)


def test_safety_signal_report_compares_case_flags_to_prediction_flags() -> None:
    cases = [
        {
            "case_id": "s1",
            "expected_behavior": "security_triage_or_escalation_after_filter",
            "safety_flags": {
                "possible_security_incident": True,
                "firmware_escalation_required": False,
                "needs_human_review": True,
            },
        },
        {
            "case_id": "s2",
            "expected_behavior": "structured_firmware_escalation",
            "safety_flags": {
                "possible_security_incident": False,
                "firmware_escalation_required": True,
                "needs_human_review": True,
            },
        },
    ]
    predictions = [
        {
            "case_id": "s1",
            "predicted_behavior": "security_triage_or_escalation_after_filter",
            "safety_signals": {
                "possible_security_incident": True,
                "firmware_escalation_required": False,
                "needs_human_review": True,
            },
        },
        {
            "case_id": "s2",
            "predicted_behavior": "route_to_expected_domains",
            "safety_signals": {
                "possible_security_incident": False,
                "firmware_escalation_required": False,
                "needs_human_review": False,
            },
        },
    ]

    summary, per_label, details = safety_signal_report(
        safety_cases=cases,
        prediction_rows=predictions,
        signal_labels=[
            "possible_security_incident",
            "firmware_escalation_required",
            "needs_human_review",
        ],
        profile="p",
    )

    assert summary["micro_precision"] == 1.0
    assert summary["micro_recall"] == 0.5
    assert summary["micro_f1"] == 0.6667
    assert any(row["safety_signal"] == "firmware_escalation_required" for row in per_label)
    assert details[1]["missed_safety_signals"] == "firmware_escalation_required|needs_human_review"
