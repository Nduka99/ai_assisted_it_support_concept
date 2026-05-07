"""Evaluate saved local-LLM JSON classifier responses on dev eval keys.

This script consumes saved response JSONL rows and the dry-run eval-key JSONL.
It does not load a model, call an LLM, train, read holdout, or generate answers.
Invalid or missing responses are retained as failed predictions so validity
problems are visible in the metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from it_support.classification.ladder import (  # noqa: E402
    multilabel_domain_report,
    primary_multiclass_report,
    ranked_domain_report,
    routing_case_detail_rows,
    safety_behavior_report,
    safety_signal_report,
)
from it_support.classification.llm_json_classifier import (  # noqa: E402
    ALLOWED_BEHAVIORS,
    SAFETY_SIGNAL_LABELS,
    invalid_prediction_record,
    llm_response_from_saved_row,
    prediction_record_from_llm_response,
)
from it_support.config import DATA_DIR, PROJECT_ROOT  # noqa: E402
from it_support.schemas import DOMAIN_LABELS  # noqa: E402


DEFAULT_EVAL_KEYS = (
    DATA_DIR
    / "eval"
    / "llm_json_classifier_dry_run"
    / "llm_json_classifier_eval_keys_dev_sample_metadata.jsonl"
)
OUT = DATA_DIR / "eval" / "llm_json_classifier_dev"


def safe_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._-") or "llm_json_classifier"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = [
            str(row.get(field, "")).replace("|", "\\|").replace("\n", " ")
            for field in fields
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def assert_dev_eval_keys(rows: list[dict[str, Any]], *, path: Path) -> None:
    splits = sorted({str(row.get("split")) for row in rows})
    if splits != ["routing_dev_eval"]:
        raise ValueError(f"{path} must contain only routing_dev_eval rows; got {splits}")


def profile_from_eval_keys(eval_keys: list[dict[str, Any]], requested: str | None) -> str:
    if requested:
        return requested
    profiles = sorted({str(row.get("profile") or "") for row in eval_keys if row.get("profile")})
    if len(profiles) == 1:
        return profiles[0].replace("_dry_run", "_saved_response_eval")
    return "llm_json_classifier_saved_response_eval"


def response_index(response_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_id = {}
    duplicates = []
    for row in response_rows:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise ValueError("Every saved response row must include case_id")
        if case_id in by_id:
            duplicates.append(case_id)
            continue
        by_id[case_id] = row
    if duplicates:
        raise ValueError(f"Duplicate response case_id rows: {sorted(set(duplicates))[:10]}")
    return by_id


def eval_key_to_safety_case(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": row.get("case_id"),
        "expected_behavior": row.get("expected_behavior"),
        "safety_flags": dict(row.get("expected_safety_signals") or {}),
        "title": row.get("title", ""),
        "source_url": row.get("source_url", ""),
    }


def prediction_rows_from_saved_responses(
    *,
    eval_keys: list[dict[str, Any]],
    response_rows: list[dict[str, Any]],
    profile: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    responses_by_id = response_index(response_rows)
    eval_ids = {str(row["case_id"]) for row in eval_keys}
    predictions = []
    parsed_responses = []
    parse_status = []

    for case in sorted(eval_keys, key=lambda row: str(row["case_id"])):
        case_id = str(case["case_id"])
        response_row = responses_by_id.get(case_id)
        if response_row is None:
            error = "missing saved response row"
            predictions.append(invalid_prediction_record(case=case, profile=profile, error=error))
            parse_status.append(
                {
                    "case_id": case_id,
                    "profile": profile,
                    "parse_status": "missing",
                    "parse_error": error,
                }
            )
            continue

        try:
            parsed = llm_response_from_saved_row(response_row)
            prediction = prediction_record_from_llm_response(
                case=case,
                response=parsed,
                profile=profile,
            )
            prediction["parse_status"] = "valid"
            prediction["parse_error"] = ""
            predictions.append(prediction)
            parsed_responses.append(
                {
                    "case_id": case_id,
                    "profile": profile,
                    "response": parsed,
                }
            )
            parse_status.append(
                {
                    "case_id": case_id,
                    "profile": profile,
                    "parse_status": "valid",
                    "parse_error": "",
                }
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            predictions.append(invalid_prediction_record(case=case, profile=profile, error=error))
            parse_status.append(
                {
                    "case_id": case_id,
                    "profile": profile,
                    "parse_status": "invalid",
                    "parse_error": error,
                }
            )

    for case_id in sorted(set(responses_by_id) - eval_ids):
        parse_status.append(
            {
                "case_id": case_id,
                "profile": profile,
                "parse_status": "extra",
                "parse_error": "response case_id is not present in eval keys",
            }
        )

    return predictions, parsed_responses, parse_status


def build_outputs(
    *,
    responses_path: Path,
    eval_keys_path: Path,
    profile: str | None,
    run_name: str | None,
) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    eval_keys = read_jsonl(eval_keys_path)
    response_rows = read_jsonl(responses_path)
    assert_dev_eval_keys(eval_keys, path=eval_keys_path)

    resolved_profile = profile_from_eval_keys(eval_keys, profile)
    suffix = safe_name(run_name or resolved_profile)
    predictions, parsed_responses, parse_status = prediction_rows_from_saved_responses(
        eval_keys=eval_keys,
        response_rows=response_rows,
        profile=resolved_profile,
    )

    primary_summary, primary_rows, confusion_rows = primary_multiclass_report(
        predictions,
        labels=DOMAIN_LABELS,
        profile=resolved_profile,
    )
    multilabel_summary, multilabel_rows = multilabel_domain_report(
        predictions,
        labels=DOMAIN_LABELS,
        profile=resolved_profile,
    )
    ranked_summary = ranked_domain_report(predictions, profile=resolved_profile)
    routing_details = routing_case_detail_rows(predictions, profile=resolved_profile)
    safety_cases = [eval_key_to_safety_case(row) for row in eval_keys]
    safety_behavior_summary, safety_behavior_rows, safety_behavior_confusion = (
        safety_behavior_report(
            predictions,
            labels=ALLOWED_BEHAVIORS,
            profile=resolved_profile,
        )
    )
    safety_signal_summary, safety_signal_rows, safety_signal_details = safety_signal_report(
        safety_cases=safety_cases,
        prediction_rows=predictions,
        signal_labels=SAFETY_SIGNAL_LABELS,
        profile=resolved_profile,
    )

    outputs = {
        "predictions": OUT / f"llm_json_classifier_predictions_{suffix}.jsonl",
        "parsed_responses": OUT / f"llm_json_classifier_parsed_responses_{suffix}.jsonl",
        "parse_status": OUT / f"llm_json_classifier_parse_status_{suffix}.csv",
        "primary_metrics": OUT / f"llm_json_classifier_primary_metrics_{suffix}.csv",
        "primary_per_domain": OUT / f"llm_json_classifier_primary_per_domain_{suffix}.csv",
        "primary_confusion": OUT / f"llm_json_classifier_primary_confusion_{suffix}.csv",
        "multilabel_metrics": OUT / f"llm_json_classifier_multilabel_metrics_{suffix}.csv",
        "multilabel_per_domain": OUT / f"llm_json_classifier_multilabel_per_domain_{suffix}.csv",
        "ranked_metrics": OUT / f"llm_json_classifier_ranked_metrics_{suffix}.csv",
        "routing_details": OUT / f"llm_json_classifier_routing_details_{suffix}.csv",
        "safety_behavior_metrics": OUT
        / f"llm_json_classifier_safety_behavior_metrics_{suffix}.csv",
        "safety_behavior_per_label": OUT
        / f"llm_json_classifier_safety_behavior_per_label_{suffix}.csv",
        "safety_behavior_confusion": OUT
        / f"llm_json_classifier_safety_behavior_confusion_{suffix}.csv",
        "safety_signal_metrics": OUT
        / f"llm_json_classifier_safety_signal_metrics_{suffix}.csv",
        "safety_signal_per_label": OUT
        / f"llm_json_classifier_safety_signal_per_label_{suffix}.csv",
        "safety_signal_details": OUT
        / f"llm_json_classifier_safety_signal_details_{suffix}.csv",
        "summary_json": OUT / f"llm_json_classifier_summary_{suffix}.json",
        "summary_md": OUT / f"llm_json_classifier_summary_{suffix}.md",
    }

    write_jsonl(outputs["predictions"], predictions)
    write_jsonl(outputs["parsed_responses"], parsed_responses)
    write_csv(outputs["parse_status"], parse_status, ["profile", "case_id", "parse_status", "parse_error"])
    write_csv(outputs["primary_metrics"], [primary_summary], ["profile", "cases", "accuracy", "macro_f1", "weighted_f1"])
    write_csv(outputs["primary_per_domain"], primary_rows, ["profile", "domain", "support", "tp", "fp", "fn", "precision", "recall", "f1"])
    write_csv(outputs["primary_confusion"], confusion_rows, ["profile", "expected_primary_domain", "predicted_primary_domain", "records"])
    write_csv(outputs["multilabel_metrics"], [multilabel_summary], ["profile", "cases", "exact_match_ratio", "micro_precision", "micro_recall", "micro_f1", "macro_f1", "hamming_loss", "avg_expected_labels", "avg_predicted_labels"])
    write_csv(outputs["multilabel_per_domain"], multilabel_rows, ["profile", "domain", "support", "tp", "fp", "fn", "tn", "precision", "recall", "f1"])
    write_csv(outputs["ranked_metrics"], [ranked_summary], list(ranked_summary))
    write_csv(outputs["routing_details"], routing_details, ["profile", "case_id", "expected_primary_domain", "predicted_primary_domain", "primary_correct", "expected_domains", "predicted_domains", "exact_match", "first_primary_rank", "first_any_expected_rank", "title", "source_url"])
    write_csv(outputs["safety_behavior_metrics"], [safety_behavior_summary], ["profile", "cases", "accuracy", "macro_f1", "weighted_f1"])
    write_csv(outputs["safety_behavior_per_label"], safety_behavior_rows, ["profile", "expected_behavior", "support", "tp", "fp", "fn", "precision", "recall", "f1"])
    write_csv(outputs["safety_behavior_confusion"], safety_behavior_confusion, ["profile", "expected_behavior", "predicted_behavior", "records"])
    write_csv(outputs["safety_signal_metrics"], [safety_signal_summary], ["profile", "cases", "micro_precision", "micro_recall", "micro_f1", "macro_f1"])
    write_csv(outputs["safety_signal_per_label"], safety_signal_rows, ["profile", "safety_signal", "support", "tp", "fp", "fn", "tn", "precision", "recall", "f1"])
    write_csv(outputs["safety_signal_details"], safety_signal_details, ["profile", "case_id", "expected_behavior", "predicted_behavior", "behavior_correct", "expected_safety_signals", "predicted_safety_signals", "missed_safety_signals", "extra_safety_signals", "title", "source_url"])

    valid_count = sum(row["parse_status"] == "valid" for row in parse_status)
    missing_count = sum(row["parse_status"] == "missing" for row in parse_status)
    invalid_count = sum(row["parse_status"] == "invalid" for row in parse_status)
    extra_count = sum(row["parse_status"] == "extra" for row in parse_status)
    counts = {
        "eval_key_cases": len(eval_keys),
        "response_rows": len(response_rows),
        "valid_response_cases": valid_count,
        "invalid_response_cases": invalid_count,
        "missing_response_cases": missing_count,
        "extra_response_rows": extra_count,
        "predictions_scored": len(predictions),
    }
    summary = {
        "stage": "llm_json_classifier_response_eval",
        "scope": "dev_saved_responses",
        "profile": resolved_profile,
        "policy": [
            "Consumes saved local-LLM JSON classifier responses and dev eval keys only.",
            "Does not read holdout files.",
            "Does not load a model, call an LLM, train, or generate answers.",
            "Invalid or missing responses are retained as failed predictions.",
        ],
        "inputs": {
            "responses": safe_rel(responses_path),
            "eval_keys": safe_rel(eval_keys_path),
        },
        "counts": counts,
        "primary_metrics": primary_summary,
        "multilabel_metrics": multilabel_summary,
        "ranked_metrics": ranked_summary,
        "safety_behavior_metrics": safety_behavior_summary,
        "safety_signal_metrics": safety_signal_summary,
        "outputs": {key: safe_rel(path) for key, path in outputs.items()},
    }
    write_json(outputs["summary_json"], summary)

    overview = {
        "profile": resolved_profile,
        **counts,
        "primary_accuracy": primary_summary["accuracy"],
        "multilabel_micro_f1": multilabel_summary["micro_f1"],
        "primary_hit_at_3": ranked_summary["primary_hit_at_3"],
        "safety_signal_micro_f1": safety_signal_summary["micro_f1"],
    }
    lines = [
        "# LLM JSON Classifier Response Evaluation",
        "",
        "This stage scores saved JSON classifier responses against dev eval keys. "
        "It does not load or call a model.",
        "",
        "## Policy",
        "",
        *[f"- {item}" for item in summary["policy"]],
        "",
        "## Overview",
        "",
        md_table([overview], list(overview)),
        "",
        "## Outputs",
        "",
        *[f"- `{path}`" for path in summary["outputs"].values()],
    ]
    outputs["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--eval-keys", type=Path, default=DEFAULT_EVAL_KEYS)
    parser.add_argument("--profile")
    parser.add_argument("--run-name")
    parser.add_argument("--summary", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_outputs(
        responses_path=args.responses,
        eval_keys_path=args.eval_keys,
        profile=args.profile,
        run_name=args.run_name,
    )
    print(f"wrote {summary['outputs']['summary_md']}")
    print(
        f"profile={summary['profile']} valid={summary['counts']['valid_response_cases']} "
        f"invalid={summary['counts']['invalid_response_cases']} "
        f"missing={summary['counts']['missing_response_cases']} "
        f"primary_acc={summary['primary_metrics']['accuracy']}"
    )
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
