"""Build a dev-only BGE vs Qwen3 retrieval complementarity report.

This script consumes existing non-self retrieval prediction artifacts. It does
not embed, train, generate answers, or score holdout.
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

from it_support.config import DATA_DIR, PROJECT_ROOT
from it_support.retrieval.complementarity import (
    compare_prediction_rows,
    diagnostic_upper_bound_rows,
    domain_complementarity_rows,
    example_rows,
    outcome_count_rows,
    rank_advantage_rows,
)
from it_support.retrieval.eval_scoring import group_qrels


FIXTURE = DATA_DIR / "eval" / "nonself_retrieval_eval_fixture"
BGE_DEV = DATA_DIR / "eval" / "nonself_retrieval_bge_dev"
QWEN3_DEV = DATA_DIR / "eval" / "nonself_retrieval_qwen3_dev"
OUTPUT_DIR = DATA_DIR / "eval" / "nonself_retrieval_complementarity_dev"

DEFAULT_BGE_PREDICTIONS = BGE_DEV / "retrieval_predictions_dev_bge_small_en_v15.jsonl"
DEFAULT_QWEN3_PREDICTIONS = (
    QWEN3_DEV / "retrieval_predictions_dev_qwen3_embedding_06b.jsonl"
)
DEFAULT_QRELS = FIXTURE / "retrieval_qrels.jsonl"
DEFAULT_BASELINE_PROFILE = "rule_triage_filtered_bge_small_en_v15"
DEFAULT_CHALLENGER_PROFILE = "rule_triage_filtered_qwen3_embedding_06b"
EXPECTED_SPLIT = "retrieval_dev_eval"


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
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


def select_profile(
    rows: list[dict[str, Any]],
    *,
    profile: str,
    expected_split: str,
) -> list[dict[str, Any]]:
    selected = [row for row in rows if row.get("profile") == profile]
    if not selected:
        profiles = sorted({str(row.get("profile")) for row in rows})
        raise ValueError(f"Profile {profile!r} not found. Available profiles: {profiles}")

    splits = sorted({str(row.get("split")) for row in selected})
    if splits != [expected_split]:
        raise ValueError(
            f"Complementarity analysis is dev-only; expected split {expected_split!r}, "
            f"got {splits}"
        )
    return sorted(selected, key=lambda row: str(row["query_id"]))


def interpretation_rows(
    *,
    outcome_rows: list[dict[str, Any]],
    upper_bound_rows: list[dict[str, Any]],
) -> list[str]:
    top1 = {row["outcome"]: row for row in outcome_rows if int(row["cutoff"]) == 1}
    top5 = {row["outcome"]: row for row in outcome_rows if int(row["cutoff"]) == 5}
    upper5 = next(row for row in upper_bound_rows if int(row["cutoff"]) == 5)
    qwen_rank1_only = int(top1.get("qwen3_only", {}).get("queries", 0))
    bge_rank1_only = int(top1.get("bge_only", {}).get("queries", 0))
    qwen_top5_only = int(top5.get("qwen3_only", {}).get("queries", 0))
    bge_top5_only = int(top5.get("bge_only", {}).get("queries", 0))

    lines = []
    if qwen_rank1_only > bge_rank1_only:
        lines.append("Qwen3 contributes more unique grade-2 rank-1 hits than BGE.")
    elif bge_rank1_only > qwen_rank1_only:
        lines.append("BGE contributes more unique grade-2 rank-1 hits than Qwen3.")
    else:
        lines.append("BGE and Qwen3 contribute the same number of unique grade-2 rank-1 hits.")

    if bge_top5_only > qwen_top5_only:
        lines.append("BGE contributes more unique grade-2 top-5 recall than Qwen3.")
    elif qwen_top5_only > bge_top5_only:
        lines.append("Qwen3 contributes more unique grade-2 top-5 recall than BGE.")
    else:
        lines.append("BGE and Qwen3 contribute the same number of unique grade-2 top-5 hits.")

    lines.append(
        "The either-model top-5 line is a diagnostic upper bound, not a deployable "
        f"oracle: {upper5['either_model_hit_rate']} vs BGE {upper5['bge_hit_rate']} "
        f"and Qwen3 {upper5['qwen3_hit_rate']}."
    )
    lines.append(
        "Working recommendation: keep rule-triage filtering, keep BGE as a recall "
        "candidate, and test Qwen3 as a high-grade ranker or reranker before replacing "
        "either model."
    )
    return lines


def build_outputs(
    *,
    bge_predictions_path: Path,
    qwen3_predictions_path: Path,
    qrels_path: Path,
    output_dir: Path,
    baseline_profile: str,
    challenger_profile: str,
    expected_split: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)

    bge_rows = read_jsonl(bge_predictions_path)
    qwen3_rows = read_jsonl(qwen3_predictions_path)
    qrels = read_jsonl(qrels_path)

    baseline_rows = select_profile(
        bge_rows,
        profile=baseline_profile,
        expected_split=expected_split,
    )
    challenger_rows = select_profile(
        qwen3_rows,
        profile=challenger_profile,
        expected_split=expected_split,
    )
    query_ids = {str(row["query_id"]) for row in baseline_rows}
    qrels_by_query = group_qrels(
        [row for row in qrels if str(row["query_id"]) in query_ids]
    )

    all_comparison_rows = compare_prediction_rows(
        baseline_rows=baseline_rows,
        challenger_rows=challenger_rows,
        qrels_by_query=qrels_by_query,
    )
    eligible_rows = [
        row for row in all_comparison_rows if bool(row.get("is_grade_2_eligible"))
    ]

    outcome_rows = outcome_count_rows(eligible_rows)
    domain_rows = domain_complementarity_rows(eligible_rows)
    advantage_rows = rank_advantage_rows(eligible_rows)
    upper_bound_rows = diagnostic_upper_bound_rows(eligible_rows)
    examples = example_rows(eligible_rows)
    interpretation = interpretation_rows(
        outcome_rows=outcome_rows,
        upper_bound_rows=upper_bound_rows,
    )

    outputs = {
        "query_comparison_all": output_dir
        / "retrieval_complementarity_query_comparison_all_dev.csv",
        "query_comparison_grade2_eligible": output_dir
        / "retrieval_complementarity_query_comparison_grade2_eligible_dev.csv",
        "outcome_counts": output_dir / "retrieval_complementarity_outcome_counts_dev.csv",
        "domain_counts": output_dir / "retrieval_complementarity_domain_counts_dev.csv",
        "rank_advantage": output_dir / "retrieval_complementarity_rank_advantage_dev.csv",
        "upper_bound": output_dir / "retrieval_complementarity_upper_bound_dev.csv",
        "examples": output_dir / "retrieval_complementarity_examples_dev.csv",
        "summary_json": output_dir / "retrieval_complementarity_summary_dev.json",
        "summary_md": output_dir / "retrieval_complementarity_summary_dev.md",
    }

    write_csv(outputs["query_comparison_all"], all_comparison_rows)
    write_csv(outputs["query_comparison_grade2_eligible"], eligible_rows)
    write_csv(outputs["outcome_counts"], outcome_rows)
    write_csv(outputs["domain_counts"], domain_rows)
    write_csv(outputs["rank_advantage"], advantage_rows)
    write_csv(outputs["upper_bound"], upper_bound_rows)
    write_csv(outputs["examples"], examples)

    summary = {
        "stage": "nonself_retrieval_complementarity_dev",
        "scope": "dev_only",
        "policy": [
            "Consumes existing non-self dev retrieval prediction artifacts.",
            "Compares rule-triage-filtered BGE with rule-triage-filtered Qwen3.",
            "Counts grade-2 complementarity only on queries with grade-2 positives.",
            "Does not embed, train, generate answers, or score holdout.",
        ],
        "inputs": {
            "bge_predictions": rel(bge_predictions_path),
            "qwen3_predictions": rel(qwen3_predictions_path),
            "qrels": rel(qrels_path),
            "baseline_profile": baseline_profile,
            "challenger_profile": challenger_profile,
            "split": expected_split,
        },
        "counts": {
            "bge_prediction_rows_loaded": len(bge_rows),
            "qwen3_prediction_rows_loaded": len(qwen3_rows),
            "baseline_profile_rows": len(baseline_rows),
            "challenger_profile_rows": len(challenger_rows),
            "all_compared_queries": len(all_comparison_rows),
            "grade_2_eligible_queries": len(eligible_rows),
            "grade_2_ineligible_queries": len(all_comparison_rows) - len(eligible_rows),
        },
        "outcome_counts": outcome_rows,
        "domain_counts": domain_rows,
        "rank_advantage": advantage_rows,
        "diagnostic_upper_bound": upper_bound_rows,
        "interpretation": interpretation,
        "outputs": {key: rel(path) for key, path in outputs.items()},
    }
    write_json(outputs["summary_json"], summary)

    lines = [
        "# Retrieval Complementarity Dev Analysis",
        "",
        "This stage compares existing dev-only BGE and Qwen3 retrieval predictions. "
        "It does not run embedding models, train classifiers, generate answers, or score holdout.",
        "",
        "## Policy",
        "",
        *[f"- {item}" for item in summary["policy"]],
        "",
        "## Inputs",
        "",
        md_table([summary["inputs"]], list(summary["inputs"])),
        "",
        "## Counts",
        "",
        md_table([summary["counts"]], list(summary["counts"])),
        "",
        "## Grade-2 Outcome Counts",
        "",
        md_table(outcome_rows, ["cutoff", "outcome", "queries", "share"]),
        "",
        "## Diagnostic Upper Bound",
        "",
        md_table(
            upper_bound_rows,
            [
                "cutoff",
                "bge_hit_rate",
                "qwen3_hit_rate",
                "either_model_hit_rate",
                "both_models_hit_rate",
                "queries",
            ],
        ),
        "",
        "## Rank Advantage",
        "",
        md_table(advantage_rows, ["rank_advantage", "queries", "share"]),
        "",
        "## Interpretation",
        "",
        *[f"- {item}" for item in interpretation],
        "",
        "## Outputs",
        "",
        *[f"- `{rel(path)}`" for path in outputs.values()],
    ]
    outputs["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bge-predictions", type=Path, default=DEFAULT_BGE_PREDICTIONS)
    parser.add_argument("--qwen3-predictions", type=Path, default=DEFAULT_QWEN3_PREDICTIONS)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--baseline-profile", default=DEFAULT_BASELINE_PROFILE)
    parser.add_argument("--challenger-profile", default=DEFAULT_CHALLENGER_PROFILE)
    parser.add_argument("--split", default=EXPECTED_SPLIT)
    parser.add_argument("--summary", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_outputs(
        bge_predictions_path=args.bge_predictions,
        qwen3_predictions_path=args.qwen3_predictions,
        qrels_path=args.qrels,
        output_dir=args.output_dir,
        baseline_profile=args.baseline_profile,
        challenger_profile=args.challenger_profile,
        expected_split=args.split,
    )
    print(f"wrote {summary['outputs']['summary_md']}")
    print(f"grade_2_eligible_queries={summary['counts']['grade_2_eligible_queries']}")
    for row in summary["diagnostic_upper_bound"]:
        print(
            f"grade_2@{row['cutoff']}: bge={row['bge_hit_rate']} "
            f"qwen3={row['qwen3_hit_rate']} either={row['either_model_hit_rate']}"
        )
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
