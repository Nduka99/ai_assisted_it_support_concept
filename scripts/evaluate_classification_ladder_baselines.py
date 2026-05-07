"""Evaluate the dev-only transparent classification ladder baselines.

This script consumes existing question-only dev fixtures and saved rule-based
prediction artifacts. It does not read holdout, train, embed, call an LLM, or
generate troubleshooting answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from it_support.classification.ladder import (
    multilabel_domain_report,
    primary_multiclass_report,
    ranked_domain_report,
    routing_case_detail_rows,
    safety_behavior_report,
    safety_signal_report,
)
from it_support.config import DATA_DIR, PROJECT_ROOT
from it_support.schemas import DOMAIN_LABELS


SPLITS_DIR = DATA_DIR / "eval" / "candidate_loader_and_eval_splits"
RULE_BASELINE_DIR = DATA_DIR / "eval" / "rule_based_triage_safety_baseline"
OUT = DATA_DIR / "eval" / "classification_ladder_dev"

PROFILES = ("text_only", "metadata_assisted")
ROUTING_DEV_SPLIT = "routing_dev_eval"
SAFETY_DEV_SPLIT = "safety_dev_eval"
SAFETY_SIGNAL_LABELS = [
    "possible_security_incident",
    "firmware_escalation_required",
    "data_loss_risk",
    "credential_secret_risk",
    "needs_human_review",
]


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


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


def assert_dev_split(rows: list[dict[str, Any]], *, expected_split: str, path: Path) -> None:
    splits = sorted({str(row.get("split")) for row in rows})
    if splits != [expected_split]:
        raise ValueError(f"{path} must contain only {expected_split!r}; got {splits}")


def assert_case_alignment(
    *,
    cases: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    path: Path,
) -> None:
    case_ids = {str(row["case_id"]) for row in cases}
    prediction_ids = {str(row["case_id"]) for row in predictions}
    if case_ids != prediction_ids:
        missing_predictions = sorted(case_ids - prediction_ids)
        extra_predictions = sorted(prediction_ids - case_ids)
        raise ValueError(
            f"Prediction case IDs in {path} do not match dev fixture: "
            f"missing_predictions={missing_predictions[:5]} "
            f"extra_predictions={extra_predictions[:5]}"
        )


def profile_prediction_paths(profile: str) -> tuple[Path, Path]:
    return (
        RULE_BASELINE_DIR / f"routing_predictions_dev_only_{profile}.jsonl",
        RULE_BASELINE_DIR / f"safety_predictions_dev_only_{profile}.jsonl",
    )


def safety_behavior_labels(
    *,
    safety_cases: list[dict[str, Any]],
) -> list[str]:
    return sorted({str(row.get("expected_behavior") or "") for row in safety_cases if row})


def build_outputs() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)

    routing_cases_path = SPLITS_DIR / "routing_eval_dev.jsonl"
    safety_cases_path = SPLITS_DIR / "safety_eval_dev.jsonl"
    routing_cases = read_jsonl(routing_cases_path)
    safety_cases = read_jsonl(safety_cases_path)
    assert_dev_split(routing_cases, expected_split=ROUTING_DEV_SPLIT, path=routing_cases_path)
    assert_dev_split(safety_cases, expected_split=SAFETY_DEV_SPLIT, path=safety_cases_path)

    routing_predictions_by_profile = {}
    safety_predictions_by_profile = {}
    input_prediction_paths = {}
    for profile in PROFILES:
        routing_path, safety_path = profile_prediction_paths(profile)
        routing_predictions = read_jsonl(routing_path)
        safety_predictions = read_jsonl(safety_path)
        assert_dev_split(routing_predictions, expected_split=ROUTING_DEV_SPLIT, path=routing_path)
        assert_dev_split(safety_predictions, expected_split=SAFETY_DEV_SPLIT, path=safety_path)
        assert_case_alignment(cases=routing_cases, predictions=routing_predictions, path=routing_path)
        assert_case_alignment(cases=safety_cases, predictions=safety_predictions, path=safety_path)
        routing_predictions_by_profile[profile] = sorted(
            routing_predictions, key=lambda row: str(row["case_id"])
        )
        safety_predictions_by_profile[profile] = sorted(
            safety_predictions, key=lambda row: str(row["case_id"])
        )
        input_prediction_paths[f"routing_predictions_{profile}"] = rel(routing_path)
        input_prediction_paths[f"safety_predictions_{profile}"] = rel(safety_path)

    behavior_labels = safety_behavior_labels(safety_cases=safety_cases)

    primary_metrics = []
    primary_per_domain = []
    primary_confusion = []
    multilabel_metrics = []
    multilabel_per_domain = []
    ranked_metrics = []
    routing_details = []
    safety_behavior_metrics = []
    safety_behavior_per_label = []
    safety_behavior_confusion = []
    safety_signal_metrics = []
    safety_signal_per_label = []
    safety_details = []

    for profile in PROFILES:
        routing_predictions = routing_predictions_by_profile[profile]
        safety_predictions = safety_predictions_by_profile[profile]

        primary_summary, primary_rows, confusion_rows = primary_multiclass_report(
            routing_predictions,
            labels=DOMAIN_LABELS,
            profile=profile,
        )
        primary_metrics.append(primary_summary)
        primary_per_domain.extend(primary_rows)
        primary_confusion.extend(confusion_rows)

        multilabel_summary, multilabel_rows = multilabel_domain_report(
            routing_predictions,
            labels=DOMAIN_LABELS,
            profile=profile,
        )
        multilabel_metrics.append(multilabel_summary)
        multilabel_per_domain.extend(multilabel_rows)
        ranked_metrics.append(ranked_domain_report(routing_predictions, profile=profile))
        routing_details.extend(routing_case_detail_rows(routing_predictions, profile=profile))

        behavior_summary, behavior_rows, behavior_confusion = safety_behavior_report(
            safety_predictions,
            labels=behavior_labels,
            profile=profile,
        )
        safety_behavior_metrics.append(behavior_summary)
        safety_behavior_per_label.extend(behavior_rows)
        safety_behavior_confusion.extend(behavior_confusion)

        signal_summary, signal_rows, signal_details = safety_signal_report(
            safety_cases=safety_cases,
            prediction_rows=safety_predictions,
            signal_labels=SAFETY_SIGNAL_LABELS,
            profile=profile,
        )
        safety_signal_metrics.append(signal_summary)
        safety_signal_per_label.extend(signal_rows)
        safety_details.extend(signal_details)

    outputs = {
        "primary_metrics": OUT / "classification_primary_multiclass_metrics_dev.csv",
        "primary_per_domain": OUT / "classification_primary_per_domain_dev.csv",
        "primary_confusion": OUT / "classification_primary_confusion_dev.csv",
        "multilabel_metrics": OUT / "classification_multilabel_domain_metrics_dev.csv",
        "multilabel_per_domain": OUT / "classification_multilabel_per_domain_dev.csv",
        "ranked_metrics": OUT / "classification_ranked_domain_metrics_dev.csv",
        "routing_details": OUT / "classification_routing_case_details_dev.csv",
        "safety_behavior_metrics": OUT / "classification_safety_behavior_metrics_dev.csv",
        "safety_behavior_per_label": OUT / "classification_safety_behavior_per_label_dev.csv",
        "safety_behavior_confusion": OUT / "classification_safety_behavior_confusion_dev.csv",
        "safety_signal_metrics": OUT / "classification_safety_signal_metrics_dev.csv",
        "safety_signal_per_label": OUT / "classification_safety_signal_per_label_dev.csv",
        "safety_details": OUT / "classification_safety_case_details_dev.csv",
        "summary_json": OUT / "classification_ladder_summary_dev.json",
        "summary_md": OUT / "classification_ladder_summary_dev.md",
    }

    write_csv(
        outputs["primary_metrics"],
        primary_metrics,
        ["profile", "cases", "accuracy", "macro_f1", "weighted_f1"],
    )
    write_csv(
        outputs["primary_per_domain"],
        primary_per_domain,
        ["profile", "domain", "support", "tp", "fp", "fn", "precision", "recall", "f1"],
    )
    write_csv(
        outputs["primary_confusion"],
        primary_confusion,
        ["profile", "expected_primary_domain", "predicted_primary_domain", "records"],
    )
    write_csv(
        outputs["multilabel_metrics"],
        multilabel_metrics,
        [
            "profile",
            "cases",
            "exact_match_ratio",
            "micro_precision",
            "micro_recall",
            "micro_f1",
            "macro_f1",
            "hamming_loss",
            "avg_expected_labels",
            "avg_predicted_labels",
        ],
    )
    write_csv(
        outputs["multilabel_per_domain"],
        multilabel_per_domain,
        [
            "profile",
            "domain",
            "support",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "f1",
        ],
    )
    write_csv(outputs["ranked_metrics"], ranked_metrics)
    write_csv(outputs["routing_details"], routing_details)
    write_csv(
        outputs["safety_behavior_metrics"],
        safety_behavior_metrics,
        ["profile", "cases", "accuracy", "macro_f1", "weighted_f1"],
    )
    write_csv(
        outputs["safety_behavior_per_label"],
        safety_behavior_per_label,
        [
            "profile",
            "expected_behavior",
            "support",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
        ],
    )
    write_csv(
        outputs["safety_behavior_confusion"],
        safety_behavior_confusion,
        ["profile", "expected_behavior", "predicted_behavior", "records"],
    )
    write_csv(
        outputs["safety_signal_metrics"],
        safety_signal_metrics,
        ["profile", "cases", "micro_precision", "micro_recall", "micro_f1", "macro_f1"],
    )
    write_csv(
        outputs["safety_signal_per_label"],
        safety_signal_per_label,
        [
            "profile",
            "safety_signal",
            "support",
            "tp",
            "fp",
            "fn",
            "tn",
            "precision",
            "recall",
            "f1",
        ],
    )
    write_csv(outputs["safety_details"], safety_details)

    profile_summary = []
    for profile in PROFILES:
        primary = next(row for row in primary_metrics if row["profile"] == profile)
        multilabel = next(row for row in multilabel_metrics if row["profile"] == profile)
        ranked = next(row for row in ranked_metrics if row["profile"] == profile)
        behavior = next(row for row in safety_behavior_metrics if row["profile"] == profile)
        signals = next(row for row in safety_signal_metrics if row["profile"] == profile)
        profile_summary.append(
            {
                "profile": profile,
                "primary_accuracy": primary["accuracy"],
                "primary_macro_f1": primary["macro_f1"],
                "multilabel_micro_f1": multilabel["micro_f1"],
                "multilabel_exact_match": multilabel["exact_match_ratio"],
                "ranked_primary_hit_at_1": ranked["primary_hit_at_1"],
                "ranked_primary_hit_at_3": ranked["primary_hit_at_3"],
                "ranked_graded_ndcg_at_3": ranked["graded_ndcg_at_3"],
                "safety_behavior_accuracy": behavior["accuracy"],
                "safety_signal_micro_f1": signals["micro_f1"],
            }
        )

    summary = {
        "stage": "classification_ladder_transparent_baselines",
        "scope": "dev_only",
        "policy": [
            "Consumes existing question-only dev fixtures and saved rule-based predictions.",
            "Does not read holdout files.",
            "Does not train, embed, call an LLM, or generate answers.",
            "Primary multi-class uses expected_primary_domain.",
            "Domain multi-label uses expected_domains.",
            "Ranked-domain metrics use predicted domain order; grade 2 is primary, grade 1 is secondary expected domain.",
            "Safety signal metrics compare question-visible safety_flags with predicted safety_signals.",
        ],
        "inputs": {
            "routing_cases": rel(routing_cases_path),
            "safety_cases": rel(safety_cases_path),
            **input_prediction_paths,
        },
        "counts": {
            "routing_dev_cases": len(routing_cases),
            "safety_dev_cases": len(safety_cases),
            "profiles": len(PROFILES),
        },
        "profile_summary": profile_summary,
        "primary_metrics": primary_metrics,
        "multilabel_metrics": multilabel_metrics,
        "ranked_metrics": ranked_metrics,
        "safety_behavior_metrics": safety_behavior_metrics,
        "safety_signal_metrics": safety_signal_metrics,
        "outputs": {key: rel(path) for key, path in outputs.items()},
    }
    write_json(outputs["summary_json"], summary)

    lines = [
        "# Classification Ladder Transparent Baselines",
        "",
        "This stage starts the classification ladder using existing dev-only fixtures "
        "and saved rule-based predictions. It does not train, embed, call an LLM, "
        "generate answers, or read holdout.",
        "",
        "## Policy",
        "",
        *[f"- {item}" for item in summary["policy"]],
        "",
        "## Counts",
        "",
        md_table([summary["counts"]], list(summary["counts"])),
        "",
        "## Profile Summary",
        "",
        md_table(
            profile_summary,
            [
                "profile",
                "primary_accuracy",
                "primary_macro_f1",
                "multilabel_micro_f1",
                "multilabel_exact_match",
                "ranked_primary_hit_at_1",
                "ranked_primary_hit_at_3",
                "ranked_graded_ndcg_at_3",
                "safety_behavior_accuracy",
                "safety_signal_micro_f1",
            ],
        ),
        "",
        "## Outputs",
        "",
        *[f"- `{rel(path)}`" for path in outputs.values()],
    ]
    outputs["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_outputs()
    print(f"wrote {summary['outputs']['summary_md']}")
    for row in summary["profile_summary"]:
        print(
            f"{row['profile']}: primary_acc={row['primary_accuracy']} "
            f"multilabel_micro_f1={row['multilabel_micro_f1']} "
            f"safety_signal_micro_f1={row['safety_signal_micro_f1']}"
        )
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
