from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

from it_support.classification.llm_json_classifier import SAFETY_SIGNAL_LABELS


def _load_script_module():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluate_llm_json_classifier_responses.py"
    )
    spec = importlib.util.spec_from_file_location("evaluate_llm_json_classifier_responses", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _signals(**overrides: bool) -> dict[str, bool]:
    values = {label: False for label in SAFETY_SIGNAL_LABELS}
    values.update(overrides)
    return values


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_response_evaluator_scores_valid_invalid_and_extra_rows(tmp_path) -> None:
    module = _load_script_module()
    module.OUT = tmp_path / "out"
    eval_keys_path = tmp_path / "eval_keys.jsonl"
    responses_path = tmp_path / "responses.jsonl"
    eval_keys = [
        {
            "case_id": "c1",
            "split": "routing_dev_eval",
            "profile": "llm_json_classifier_metadata_dry_run",
            "expected_primary_domain": "network_connectivity",
            "expected_domains": ["network_connectivity", "os_kernel_drivers"],
            "expected_behavior": "route_to_expected_domains",
            "expected_safety_signals": _signals(),
            "title": "WiFi drops",
            "source_url": "https://example.test/c1",
        },
        {
            "case_id": "c2",
            "split": "routing_dev_eval",
            "profile": "llm_json_classifier_metadata_dry_run",
            "expected_primary_domain": "hardware",
            "expected_domains": ["hardware"],
            "expected_behavior": "structured_firmware_escalation",
            "expected_safety_signals": _signals(
                firmware_escalation_required=True,
                needs_human_review=True,
            ),
            "title": "BIOS recovery",
            "source_url": "https://example.test/c2",
        },
    ]
    valid_response = {
        "primary_domain": "network_connectivity",
        "domains": [
            {
                "label": "network_connectivity",
                "confidence": 0.9,
                "rationale": "Connectivity is central.",
            },
            {
                "label": "os_kernel_drivers",
                "confidence": 0.4,
                "rationale": "Driver update is relevant.",
            },
        ],
        "safety": {
            "behavior": "route_to_expected_domains",
            "signals": _signals(),
        },
        "rationale": "Route networking first.",
    }
    _write_jsonl(eval_keys_path, eval_keys)
    _write_jsonl(
        responses_path,
        [
            {"case_id": "c1", "response": valid_response},
            {"case_id": "c2", "response_text": "not valid json"},
            {"case_id": "extra", "response": valid_response},
        ],
    )

    summary = module.build_outputs(
        responses_path=responses_path,
        eval_keys_path=eval_keys_path,
        profile=None,
        run_name="fixture",
    )

    assert summary["counts"]["valid_response_cases"] == 1
    assert summary["counts"]["invalid_response_cases"] == 1
    assert summary["counts"]["extra_response_rows"] == 1
    assert summary["counts"]["predictions_scored"] == 2
    assert summary["primary_metrics"]["accuracy"] == 0.5

    parse_status_path = Path(summary["outputs"]["parse_status"])
    if not parse_status_path.is_absolute():
        parse_status_path = module.PROJECT_ROOT / parse_status_path
    rows = list(csv.DictReader(parse_status_path.open(encoding="utf-8")))
    statuses = {row["case_id"]: row["parse_status"] for row in rows}
    assert statuses == {"c1": "valid", "c2": "invalid", "extra": "extra"}
