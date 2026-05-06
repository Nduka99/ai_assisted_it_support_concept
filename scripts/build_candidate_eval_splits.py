"""Build evaluation-only routing and safety fixtures from normalized candidates.

This script is intentionally conservative. It verifies loader behavior for
retrieval candidates, writes question-only eval artifacts, and does not build
FAISS indexes, training datasets, or generation-ready records.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from it_support.config import DATA_DIR, PROJECT_ROOT
from it_support.data import (
    DownstreamUse,
    NormalizedArtifact,
    build_load_audit,
    load_records,
    project_routing_eval_case,
    project_safety_eval_case,
    split_records_by_group,
)


OUT = DATA_DIR / "eval" / "candidate_loader_and_eval_splits"


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


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


def count_rows(counter: Counter[str], key_name: str) -> list[dict[str, Any]]:
    return [{key_name: key, "records": value} for key, value in counter.most_common()]


def build_outputs() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)

    classifier = load_records(
        NormalizedArtifact.STACK_EXCHANGE_CLASSIFIER_POOL,
        required_uses=(DownstreamUse.EVALUATION,),
    )
    retrieval = load_records(
        NormalizedArtifact.STACK_EXCHANGE_RETRIEVAL_CANDIDATES,
        required_uses=(DownstreamUse.RETRIEVAL,),
    )
    safety = load_records(
        NormalizedArtifact.STACK_EXCHANGE_SAFETY_EVAL_FIXTURES,
        required_uses=(DownstreamUse.EVALUATION,),
    )
    tickets = load_records(
        NormalizedArtifact.TICKET_DATASET_MANUAL_REVIEW_QUEUE,
        strict=False,
    )

    routing_splits = split_records_by_group(
        classifier.records,
        group_field="primary_domain",
        holdout_fraction=0.25,
        dev_name="routing_dev_eval",
        holdout_name="routing_holdout_eval",
    )
    routing_cases = [
        project_routing_eval_case(row, split=routing_splits[row["record_id"]])
        for row in classifier.records
    ]

    safety_splits = split_records_by_group(
        safety.records,
        group_field="expected_behavior",
        holdout_fraction=0.25,
        dev_name="safety_dev_eval",
        holdout_name="safety_holdout_eval",
    )
    safety_cases = [
        project_safety_eval_case(row, split=safety_splits[row["record_id"]])
        for row in safety.records
    ]

    routing_dev = [row for row in routing_cases if row["split"] == "routing_dev_eval"]
    routing_holdout = [row for row in routing_cases if row["split"] == "routing_holdout_eval"]
    safety_dev = [row for row in safety_cases if row["split"] == "safety_dev_eval"]
    safety_holdout = [row for row in safety_cases if row["split"] == "safety_holdout_eval"]

    outputs = {
        "routing_dev": OUT / "routing_eval_dev.jsonl",
        "routing_holdout": OUT / "routing_eval_holdout.jsonl",
        "safety_dev": OUT / "safety_eval_dev.jsonl",
        "safety_holdout": OUT / "safety_eval_holdout.jsonl",
        "routing_domain_counts": OUT / "routing_eval_domain_counts.csv",
        "safety_behavior_counts": OUT / "safety_eval_expected_behavior_counts.csv",
        "loader_audit": OUT / "loader_audit.json",
        "summary_json": OUT / "eval_split_summary.json",
        "summary_md": OUT / "eval_split_summary.md",
    }

    write_jsonl(outputs["routing_dev"], routing_dev)
    write_jsonl(outputs["routing_holdout"], routing_holdout)
    write_jsonl(outputs["safety_dev"], safety_dev)
    write_jsonl(outputs["safety_holdout"], safety_holdout)

    routing_domain_rows = count_rows(
        Counter(row["expected_primary_domain"] for row in routing_cases),
        "expected_primary_domain",
    )
    safety_behavior_rows = count_rows(
        Counter(row["expected_behavior"] for row in safety_cases),
        "expected_behavior",
    )
    write_csv(outputs["routing_domain_counts"], routing_domain_rows)
    write_csv(outputs["safety_behavior_counts"], safety_behavior_rows)

    loader_audit = {
        "stage": "candidate_loader_and_eval_splits",
        "policy": "Downstream loader must enforce record-level downstream_allowed gates.",
        "artifacts": {
            "classifier_pool": classifier.audit,
            "retrieval_candidates": retrieval.audit,
            "safety_eval_fixtures": safety.audit,
            "ticket_dataset_manual_review_queue": build_load_audit(
                tickets.records,
                artifact=NormalizedArtifact.TICKET_DATASET_MANUAL_REVIEW_QUEUE,
                path=tickets.path,
            ),
        },
        "retrieval_probe": {
            "records_loaded_with_required_use_retrieval": len(retrieval.records),
            "answer_generation_allowed_records": retrieval.audit[
                "answer_generation_allowed_records"
            ],
            "faiss_indexing_status": "deferred_until_retrieval_loader_is_consumed_by_index_builder",
        },
        "ticket_dataset_status": {
            "records_loaded_without_promotion": len(tickets.records),
            "downstream_status": "blocked_pending_manual_pii_license_provenance_schema_review",
        },
    }
    write_json(outputs["loader_audit"], loader_audit)

    counts = {
        "routing_eval_dev": len(routing_dev),
        "routing_eval_holdout": len(routing_holdout),
        "safety_eval_dev": len(safety_dev),
        "safety_eval_holdout": len(safety_holdout),
        "retrieval_candidates_verified_retrieval_allowed": len(retrieval.records),
        "retrieval_candidates_answer_generation_allowed": retrieval.audit[
            "answer_generation_allowed_records"
        ],
        "ticket_dataset_review_packets_still_blocked": len(tickets.records),
    }
    summary = {
        "stage": "candidate_loader_and_eval_splits",
        "counts": counts,
        "routing_domain_counts": routing_domain_rows,
        "safety_expected_behavior_counts": safety_behavior_rows,
        "downstream_rule": [
            "Eval artifacts are question-only.",
            "Retrieval candidates were verified with required_use=retrieval but not indexed.",
            "Answer generation remains blocked for all current normalized records.",
            "Ticket datasets remain manual-review packets only.",
            "Commercial mode remains blocked for current public POC records.",
        ],
        "outputs": {key: rel(path) for key, path in outputs.items()},
    }
    write_json(outputs["summary_json"], summary)

    count_table = [{"artifact": key, "records": value} for key, value in counts.items()]
    lines = [
        "# Candidate Loader And Eval Splits",
        "",
        "This stage turns normalized candidate records into question-only evaluation fixtures "
        "and audits the downstream loader gates. It does not train, index, or enable answer "
        "generation.",
        "",
        "## Counts",
        "",
        md_table(count_table, ["artifact", "records"]),
        "",
        "## Routing Domain Counts",
        "",
        md_table(routing_domain_rows, ["expected_primary_domain", "records"]),
        "",
        "## Safety Expected Behaviors",
        "",
        md_table(safety_behavior_rows, ["expected_behavior", "records"]),
        "",
        "## Downstream Rule",
        "",
        "- Eval artifacts are question-only.",
        "- Retrieval candidates were verified with `required_use=retrieval`, "
        "but FAISS indexing is deferred.",
        "- `answer_generation` remains false for all current normalized records.",
        "- Ticket dataset packets remain blocked pending manual PII/license/"
        "provenance/schema review.",
        "- Current public POC records remain excluded from commercial mode.",
        "",
        "## Outputs",
        "",
        *[f"- `{rel(path)}`" for path in outputs.values()],
    ]
    outputs["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print JSON summary after writing files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_outputs()
    print(f"wrote {rel(OUT / 'eval_split_summary.md')}")
    routing_total = (
        summary["counts"]["routing_eval_dev"] + summary["counts"]["routing_eval_holdout"]
    )
    safety_total = (
        summary["counts"]["safety_eval_dev"] + summary["counts"]["safety_eval_holdout"]
    )
    print(f"routing_eval={routing_total}")
    print(f"safety_eval={safety_total}")
    print(
        "retrieval_verified="
        f"{summary['counts']['retrieval_candidates_verified_retrieval_allowed']}"
    )
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
