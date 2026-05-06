"""Evaluate the deterministic triage/safety baseline on dev fixtures only.

No models, embeddings, FAISS indexes, or long-running libraries are used here.
The holdout files are intentionally not read unless ``--include-holdout`` is
passed for a later, explicit evaluation run.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from it_support.config import DATA_DIR, PROJECT_ROOT
from it_support.schemas import DOMAIN_LABELS
from it_support.triage import predict_case, prediction_to_record


EVAL_IN = DATA_DIR / "eval" / "candidate_loader_and_eval_splits"
OUT = DATA_DIR / "eval" / "rule_based_triage_safety_baseline"
PROFILES = {
    "text_only": False,
    "metadata_assisted": True,
}


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
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


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def round_metric(value: float) -> float:
    return round(value, 4)


def domain_metrics(predictions: list[dict[str, Any]], *, profile: str) -> dict[str, Any]:
    rows = []
    totals = Counter()
    exact_match = 0
    primary_hits = 0

    for item in predictions:
        expected = set(item.get("expected_domains") or [])
        predicted = {row["label"] for row in item.get("predicted_domains") or []}
        if expected == predicted:
            exact_match += 1
        if item.get("expected_primary_domain") == item.get("predicted_primary_domain"):
            primary_hits += 1
        for domain in DOMAIN_LABELS:
            expected_has = domain in expected
            predicted_has = domain in predicted
            if expected_has and predicted_has:
                totals[(domain, "tp")] += 1
                totals[("micro", "tp")] += 1
            elif not expected_has and predicted_has:
                totals[(domain, "fp")] += 1
                totals[("micro", "fp")] += 1
            elif expected_has and not predicted_has:
                totals[(domain, "fn")] += 1
                totals[("micro", "fn")] += 1
            else:
                totals[(domain, "tn")] += 1

    for domain in DOMAIN_LABELS:
        tp = totals[(domain, "tp")]
        fp = totals[(domain, "fp")]
        fn = totals[(domain, "fn")]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "profile": profile,
                "domain": domain,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round_metric(precision),
                "recall": round_metric(recall),
                "f1": round_metric(f1(precision, recall)),
            }
        )

    micro_tp = totals[("micro", "tp")]
    micro_fp = totals[("micro", "fp")]
    micro_fn = totals[("micro", "fn")]
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    macro_f1 = sum(row["f1"] for row in rows) / len(rows)
    return {
        "profile": profile,
        "cases": len(predictions),
        "primary_accuracy": round_metric(primary_hits / len(predictions)),
        "exact_match_ratio": round_metric(exact_match / len(predictions)),
        "micro_precision": round_metric(micro_precision),
        "micro_recall": round_metric(micro_recall),
        "micro_f1": round_metric(f1(micro_precision, micro_recall)),
        "macro_f1": round_metric(macro_f1),
        "per_domain": rows,
    }


def safety_metrics(predictions: list[dict[str, Any]], *, profile: str) -> dict[str, Any]:
    expected_labels = [
        "structured_firmware_escalation",
        "security_triage_or_escalation_after_filter",
    ]
    rows = []
    confusion = Counter()
    misses = defaultdict(list)
    for item in predictions:
        expected = item.get("expected_behavior")
        predicted = item.get("predicted_behavior")
        confusion[(expected, predicted)] += 1
        if expected in expected_labels and predicted != expected:
            misses[expected].append(
                {
                    "case_id": item.get("case_id"),
                    "title": item.get("title"),
                    "predicted_behavior": predicted,
                    "predicted_primary_domain": item.get("predicted_primary_domain"),
                    "source_url": item.get("source_url"),
                    "question_tags": item.get("question_tags", []),
                }
            )

    for label in expected_labels:
        tp = confusion[(label, label)]
        fp = sum(
            count
            for (expected, predicted), count in confusion.items()
            if predicted == label and expected != label
        )
        fn = sum(
            count
            for (expected, predicted), count in confusion.items()
            if expected == label and predicted != label
        )
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "profile": profile,
                "expected_behavior": label,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round_metric(precision),
                "recall": round_metric(recall),
                "f1": round_metric(f1(precision, recall)),
            }
        )

    return {
        "profile": profile,
        "cases": len(predictions),
        "confusion": [
            {
                "expected_behavior": expected,
                "predicted_behavior": predicted,
                "records": count,
            }
            for (expected, predicted), count in sorted(confusion.items())
        ],
        "per_behavior": rows,
        "firmware_false_negative_count": len(misses["structured_firmware_escalation"]),
        "security_false_negative_count": len(misses["security_triage_or_escalation_after_filter"]),
        "false_negative_samples": {
            key: value[:20]
            for key, value in misses.items()
        },
    }


def evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    profile: str,
    include_source_tags: bool,
) -> list[dict[str, Any]]:
    records = []
    for case in cases:
        prediction = predict_case(case, include_source_tags=include_source_tags)
        records.append(prediction_to_record(case, prediction))
    return records


def build_outputs(*, include_holdout: bool = False) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    routing_cases = read_jsonl(EVAL_IN / "routing_eval_dev.jsonl")
    safety_cases = read_jsonl(EVAL_IN / "safety_eval_dev.jsonl")
    split_scope = "dev_only"
    if include_holdout:
        routing_cases.extend(read_jsonl(EVAL_IN / "routing_eval_holdout.jsonl"))
        safety_cases.extend(read_jsonl(EVAL_IN / "safety_eval_holdout.jsonl"))
        split_scope = "dev_plus_holdout"

    outputs: dict[str, Path] = {}
    profile_summaries = []
    domain_rows = []
    safety_rows = []

    for profile, include_tags in PROFILES.items():
        routing_predictions = evaluate_cases(
            routing_cases,
            profile=profile,
            include_source_tags=include_tags,
        )
        safety_predictions = evaluate_cases(
            safety_cases,
            profile=profile,
            include_source_tags=include_tags,
        )

        routing_path = OUT / f"routing_predictions_{split_scope}_{profile}.jsonl"
        safety_path = OUT / f"safety_predictions_{split_scope}_{profile}.jsonl"
        write_jsonl(routing_path, routing_predictions)
        write_jsonl(safety_path, safety_predictions)
        outputs[f"routing_predictions_{profile}"] = routing_path
        outputs[f"safety_predictions_{profile}"] = safety_path

        routing_metrics = domain_metrics(routing_predictions, profile=profile)
        behavior_metrics = safety_metrics(safety_predictions, profile=profile)
        domain_rows.extend(routing_metrics["per_domain"])
        safety_rows.extend(behavior_metrics["per_behavior"])
        profile_summaries.append(
            {
                "profile": profile,
                "routing": {
                    key: value
                    for key, value in routing_metrics.items()
                    if key != "per_domain"
                },
                "safety": {
                    key: value
                    for key, value in behavior_metrics.items()
                    if key not in {"per_behavior", "false_negative_samples"}
                },
                "safety_false_negative_samples": behavior_metrics[
                    "false_negative_samples"
                ],
            }
        )

    domain_csv = OUT / f"routing_metrics_{split_scope}.csv"
    safety_csv = OUT / f"safety_metrics_{split_scope}.csv"
    summary_json = OUT / f"baseline_summary_{split_scope}.json"
    summary_md = OUT / f"baseline_summary_{split_scope}.md"
    write_csv(domain_csv, domain_rows)
    write_csv(safety_csv, safety_rows)
    outputs["routing_metrics"] = domain_csv
    outputs["safety_metrics"] = safety_csv
    outputs["summary_json"] = summary_json
    outputs["summary_md"] = summary_md

    summary = {
        "stage": "rule_based_triage_safety_baseline",
        "scope": split_scope,
        "policy": [
            "No model, embedding, FAISS, training, or generation path is used.",
            "Holdout files are not read unless --include-holdout is passed explicitly.",
            "text_only uses title and question_text only.",
            "metadata_assisted also uses Stack Exchange question/query tags for audit comparison.",
        ],
        "input_counts": {
            "routing_cases": len(routing_cases),
            "safety_cases": len(safety_cases),
        },
        "profiles": profile_summaries,
        "outputs": {key: rel(path) for key, path in outputs.items()},
    }
    write_json(summary_json, summary)

    overview_rows = []
    for item in profile_summaries:
        overview_rows.append(
            {
                "profile": item["profile"],
                "routing_micro_f1": item["routing"]["micro_f1"],
                "routing_macro_f1": item["routing"]["macro_f1"],
                "routing_primary_accuracy": item["routing"]["primary_accuracy"],
                "firmware_fn": item["safety"]["firmware_false_negative_count"],
                "security_fn": item["safety"]["security_false_negative_count"],
            }
        )

    lines = [
        "# Rule-Based Triage Safety Baseline",
        "",
        "This is a deterministic dev-set baseline. It does not load models, build indexes, train, "
        "or generate troubleshooting answers.",
        "",
        f"Scope: `{split_scope}`",
        "",
        "## Profile Summary",
        "",
        md_table(
            overview_rows,
            [
                "profile",
                "routing_micro_f1",
                "routing_macro_f1",
                "routing_primary_accuracy",
                "firmware_fn",
                "security_fn",
            ],
        ),
        "",
        "## Notes",
        "",
        "- `text_only` uses only title and question text.",
        "- `metadata_assisted` also uses Stack Exchange tags, so treat it as a "
        "source-metadata audit profile.",
        "- Holdout evaluation remains untouched unless explicitly requested later.",
        "- Safety false negatives are measured against conservative safety fixtures; "
        "review samples before treating them as true model failures.",
        "",
        "## Outputs",
        "",
        *[f"- `{rel(path)}`" for path in outputs.values()],
    ]
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="Explicitly include holdout files. Do not use during dev tuning.",
    )
    parser.add_argument("--summary", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_outputs(include_holdout=args.include_holdout)
    scope = summary["scope"]
    print(f"wrote {rel(OUT / f'baseline_summary_{scope}.md')}")
    print(f"scope={scope}")
    for profile in summary["profiles"]:
        print(
            f"{profile['profile']}: routing_micro_f1="
            f"{profile['routing']['micro_f1']} firmware_fn="
            f"{profile['safety']['firmware_false_negative_count']} security_fn="
            f"{profile['safety']['security_false_negative_count']}"
        )
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
