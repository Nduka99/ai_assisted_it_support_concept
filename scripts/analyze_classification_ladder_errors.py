"""Build a dev-only error review for the classification ladder baselines.

This script consumes chart/detail artifacts from the transparent classification
ladder step. It does not read holdout, train, embed, call an LLM, or generate
answers.
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

from it_support.classification.error_review import (
    multilabel_gap_count_rows,
    multilabel_gap_examples,
    primary_error_count_rows,
    primary_error_examples,
    ranked_primary_bucket_rows,
    safety_signal_gap_count_rows,
    safety_signal_gap_examples,
)
from it_support.config import DATA_DIR, PROJECT_ROOT


LADDER_DIR = DATA_DIR / "eval" / "classification_ladder_dev"
OUT = DATA_DIR / "eval" / "classification_ladder_error_review_dev"


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


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


def top_rows(rows: list[dict[str, Any]], *, limit: int = 10) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: int(row.get("records") or 0), reverse=True)[:limit]


def build_outputs() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)

    routing_details_path = LADDER_DIR / "classification_routing_case_details_dev.csv"
    safety_details_path = LADDER_DIR / "classification_safety_case_details_dev.csv"
    ladder_summary_path = LADDER_DIR / "classification_ladder_summary_dev.json"
    routing_rows = read_csv(routing_details_path)
    safety_rows = read_csv(safety_details_path)
    ladder_summary = json.loads(ladder_summary_path.read_text(encoding="utf-8"))
    if ladder_summary.get("scope") != "dev_only":
        raise ValueError(
            "Classification error review only accepts dev-only ladder summaries; "
            f"got {ladder_summary.get('scope')!r}"
        )

    primary_counts = primary_error_count_rows(routing_rows)
    primary_examples = primary_error_examples(routing_rows)
    multilabel_counts = multilabel_gap_count_rows(routing_rows)
    multilabel_examples = multilabel_gap_examples(routing_rows)
    ranked_buckets = ranked_primary_bucket_rows(routing_rows)
    safety_signal_counts = safety_signal_gap_count_rows(safety_rows)
    safety_signal_examples = safety_signal_gap_examples(safety_rows)

    outputs = {
        "primary_error_counts": OUT / "classification_primary_error_counts_dev.csv",
        "primary_error_examples": OUT / "classification_primary_error_examples_dev.csv",
        "multilabel_gap_counts": OUT / "classification_multilabel_gap_counts_dev.csv",
        "multilabel_gap_examples": OUT / "classification_multilabel_gap_examples_dev.csv",
        "ranked_primary_buckets": OUT / "classification_ranked_primary_buckets_dev.csv",
        "safety_signal_gap_counts": OUT / "classification_safety_signal_gap_counts_dev.csv",
        "safety_signal_gap_examples": OUT / "classification_safety_signal_gap_examples_dev.csv",
        "summary_json": OUT / "classification_error_review_summary_dev.json",
        "summary_md": OUT / "classification_error_review_summary_dev.md",
    }

    write_csv(
        outputs["primary_error_counts"],
        primary_counts,
        ["profile", "expected_primary_domain", "predicted_primary_domain", "records"],
    )
    write_csv(outputs["primary_error_examples"], primary_examples)
    write_csv(
        outputs["multilabel_gap_counts"],
        multilabel_counts,
        ["profile", "gap_type", "domain", "records"],
    )
    write_csv(outputs["multilabel_gap_examples"], multilabel_examples)
    write_csv(
        outputs["ranked_primary_buckets"],
        ranked_buckets,
        ["profile", "expected_primary_domain", "rank_bucket", "records"],
    )
    write_csv(
        outputs["safety_signal_gap_counts"],
        safety_signal_counts,
        ["profile", "gap_type", "safety_signal", "records"],
    )
    write_csv(outputs["safety_signal_gap_examples"], safety_signal_examples)

    primary_error_totals = {}
    for row in primary_counts:
        profile = str(row["profile"])
        primary_error_totals[profile] = primary_error_totals.get(profile, 0) + int(row["records"])
    safety_gap_totals = {}
    for row in safety_signal_counts:
        profile = str(row["profile"])
        safety_gap_totals[profile] = safety_gap_totals.get(profile, 0) + int(row["records"])

    interpretation = [
        "Primary-domain error review isolates where the first classifier label is wrong.",
        "Multi-label gap counts separate missing expected domains from extra predicted domains.",
        "Rank buckets show whether the expected primary domain is present lower in the ranked list.",
        "Safety-signal gaps may exist even when the higher-level safety behavior is correct.",
    ]

    summary = {
        "stage": "classification_ladder_error_review",
        "scope": "dev_only",
        "policy": [
            "Consumes classification ladder dev detail artifacts.",
            "Does not read holdout files.",
            "Does not train, embed, call an LLM, or generate answers.",
            "Writes hotspot counts and example queues for manual review.",
        ],
        "inputs": {
            "routing_details": rel(routing_details_path),
            "safety_details": rel(safety_details_path),
            "ladder_summary": rel(ladder_summary_path),
        },
        "counts": {
            "routing_detail_rows": len(routing_rows),
            "safety_detail_rows": len(safety_rows),
            "primary_error_pairs": len(primary_counts),
            "multilabel_gap_rows": len(multilabel_counts),
            "safety_signal_gap_rows": len(safety_signal_counts),
        },
        "primary_error_totals": primary_error_totals,
        "safety_signal_gap_totals": safety_gap_totals,
        "top_primary_errors": top_rows(primary_counts),
        "top_multilabel_gaps": top_rows(multilabel_counts),
        "top_safety_signal_gaps": top_rows(safety_signal_counts),
        "interpretation": interpretation,
        "outputs": {key: rel(path) for key, path in outputs.items()},
    }
    write_json(outputs["summary_json"], summary)

    lines = [
        "# Classification Ladder Error Review",
        "",
        "This stage reviews dev-only classification ladder errors. It does not train, "
        "embed, call an LLM, generate answers, or read holdout.",
        "",
        "## Policy",
        "",
        *[f"- {item}" for item in summary["policy"]],
        "",
        "## Counts",
        "",
        md_table([summary["counts"]], list(summary["counts"])),
        "",
        "## Primary Error Totals",
        "",
        md_table(
            [
                {"profile": profile, "primary_errors": count}
                for profile, count in sorted(primary_error_totals.items())
            ],
            ["profile", "primary_errors"],
        ),
        "",
        "## Top Primary Confusions",
        "",
        md_table(
            summary["top_primary_errors"],
            ["profile", "expected_primary_domain", "predicted_primary_domain", "records"],
        ),
        "",
        "## Top Multi-Label Gaps",
        "",
        md_table(
            summary["top_multilabel_gaps"],
            ["profile", "gap_type", "domain", "records"],
        ),
        "",
        "## Top Safety Signal Gaps",
        "",
        md_table(
            summary["top_safety_signal_gaps"],
            ["profile", "gap_type", "safety_signal", "records"],
        ),
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
    parser.add_argument("--summary", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_outputs()
    print(f"wrote {summary['outputs']['summary_md']}")
    print(f"primary_error_totals={summary['primary_error_totals']}")
    print(f"safety_signal_gap_totals={summary['safety_signal_gap_totals']}")
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
