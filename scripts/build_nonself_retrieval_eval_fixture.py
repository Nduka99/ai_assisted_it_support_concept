"""Build a non-self retrieval evaluation fixture.

The fixture is intentionally metadata-derived and conservative: queries are
question-only, corpus documents are retrieval-allowed records, and relevance
judgments never point a query back to itself.
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
from it_support.data import DownstreamUse, NormalizedArtifact, load_records, split_records_by_group
from it_support.retrieval.eval_fixtures import (
    SPECIFIC_TAG_PEER_GRADE,
    assert_retrieval_fixture_policy,
    build_relevance_judgments,
    peer_group_key,
    positive_count_summary,
    project_retrieval_corpus_doc,
    project_retrieval_query_case,
    source_query_tags,
    specific_question_tags,
)


OUT = DATA_DIR / "eval" / "nonself_retrieval_eval_fixture"


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


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
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


def count_rows(counter: Counter[Any], key_name: str) -> list[dict[str, Any]]:
    return [{key_name: key, "records": value} for key, value in counter.most_common()]


def peer_group_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[peer_group_key(record)].append(record)
    rows = []
    for key, group_records in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        first = group_records[0]
        rows.append(
            {
                "peer_group": key,
                "primary_domain": first.get("primary_domain"),
                "source_query_tags": "|".join(source_query_tags(first)),
                "records": len(group_records),
            }
        )
    return rows


def positive_count_rows(qrels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in qrels:
        grouped[str(row["query_id"])].append(row)
    return [
        {
            "query_id": query_id,
            "positive_records": len(rows),
            "grade_2_positive_records": sum(
                1 for row in rows if int(row["relevance_grade"]) >= SPECIFIC_TAG_PEER_GRADE
            ),
            "primary_domain": rows[0].get("query_primary_domain"),
            "peer_group": rows[0].get("query_peer_group"),
        }
        for query_id, rows in sorted(grouped.items())
    ]


def build_outputs() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    load_result = load_records(
        NormalizedArtifact.STACK_EXCHANGE_RETRIEVAL_CANDIDATES,
        required_uses=(DownstreamUse.RETRIEVAL,),
    )
    records = load_result.records

    split_by_id = split_records_by_group(
        records,
        group_field="primary_domain",
        holdout_fraction=0.25,
        dev_name="retrieval_dev_eval",
        holdout_name="retrieval_holdout_eval",
    )
    corpus_docs = [project_retrieval_corpus_doc(row) for row in records]
    queries = [
        project_retrieval_query_case(row, split=split_by_id[row["record_id"]])
        for row in records
    ]
    qrels = build_relevance_judgments(records)
    assert_retrieval_fixture_policy(records=records, queries=queries, qrels=qrels)

    query_ids_with_qrels = {row["query_id"] for row in qrels}
    uncovered_queries = sorted(
        str(query["query_id"]) for query in queries if query["query_id"] not in query_ids_with_qrels
    )
    if uncovered_queries:
        raise RuntimeError(
            "Every retrieval query must have at least one non-self positive; "
            f"uncovered sample={uncovered_queries[:10]}"
        )

    query_domain_rows = count_rows(
        Counter(query["expected_primary_domain"] for query in queries),
        "primary_domain",
    )
    split_rows = count_rows(Counter(query["split"] for query in queries), "split")
    qrel_grade_rows = count_rows(Counter(row["relevance_grade"] for row in qrels), "grade")
    qrel_domain_rows = count_rows(
        Counter(row["query_primary_domain"] for row in qrels),
        "primary_domain",
    )
    peer_rows = peer_group_rows(records)
    positive_rows = positive_count_rows(qrels)

    outputs = {
        "corpus": OUT / "retrieval_corpus.jsonl",
        "queries": OUT / "retrieval_queries.jsonl",
        "qrels": OUT / "retrieval_qrels.jsonl",
        "query_domain_counts": OUT / "retrieval_query_domain_counts.csv",
        "query_split_counts": OUT / "retrieval_query_split_counts.csv",
        "qrel_grade_counts": OUT / "retrieval_qrel_grade_counts.csv",
        "qrel_domain_counts": OUT / "retrieval_qrel_domain_counts.csv",
        "peer_group_counts": OUT / "retrieval_peer_group_counts.csv",
        "positive_counts_by_query": OUT / "retrieval_positive_counts_by_query.csv",
        "summary_json": OUT / "retrieval_fixture_summary.json",
        "summary_md": OUT / "retrieval_fixture_summary.md",
    }
    write_jsonl(outputs["corpus"], corpus_docs)
    write_jsonl(outputs["queries"], queries)
    write_jsonl(outputs["qrels"], qrels)
    write_csv(outputs["query_domain_counts"], query_domain_rows)
    write_csv(outputs["query_split_counts"], split_rows)
    write_csv(outputs["qrel_grade_counts"], qrel_grade_rows)
    write_csv(outputs["qrel_domain_counts"], qrel_domain_rows)
    write_csv(outputs["peer_group_counts"], peer_rows)
    write_csv(outputs["positive_counts_by_query"], positive_rows)

    records_with_specific_tags = sum(1 for row in records if specific_question_tags(row))
    summary = {
        "stage": "nonself_retrieval_eval_fixture",
        "scope": "fixture_only_no_model_scoring",
        "policy": [
            "Loads only stack_exchange_retrieval_candidates with required_use=retrieval.",
            "Rejects answer-generation-enabled records.",
            "Queries are question-only and exclude accepted/top answer text.",
            "Relevance judgments never point a query to its own record.",
            "Grade 1 means same primary domain plus same source query tag.",
            "Grade 2 means same primary domain plus shared specific question tag.",
            "Holdout query rows are created but not scored in this stage.",
        ],
        "input_counts": {
            "retrieval_records": len(records),
            "answer_generation_allowed_records": (
                load_result.audit["answer_generation_allowed_records"]
            ),
            "records_with_specific_question_tags": records_with_specific_tags,
        },
        "fixture_counts": {
            "corpus_docs": len(corpus_docs),
            "queries": len(queries),
            "queries_with_nonself_positives": len(query_ids_with_qrels),
            "qrels": len(qrels),
            "peer_groups": len(peer_rows),
            "eligible_peer_groups": sum(1 for row in peer_rows if row["records"] >= 2),
        },
        "positive_summary": positive_count_summary(qrels),
        "query_split_counts": split_rows,
        "query_domain_counts": query_domain_rows,
        "qrel_grade_counts": qrel_grade_rows,
        "qrel_domain_counts": qrel_domain_rows,
        "outputs": {key: rel(path) for key, path in outputs.items()},
    }
    write_json(outputs["summary_json"], summary)

    count_rows_for_md = [
        {"artifact": key, "records": value}
        for key, value in summary["fixture_counts"].items()
    ]
    positive_md_rows = [
        {"metric": key, "value": value}
        for key, value in summary["positive_summary"].items()
    ]
    lines = [
        "# Non-Self Retrieval Evaluation Fixture",
        "",
        "This stage creates query/corpus/qrel artifacts for retrieval evaluation. It does "
        "not run an embedding model, score holdout, train, or generate answers.",
        "",
        "## Policy",
        "",
        *[f"- {item}" for item in summary["policy"]],
        "",
        "## Fixture Counts",
        "",
        md_table(count_rows_for_md, ["artifact", "records"]),
        "",
        "## Positive Coverage",
        "",
        md_table(positive_md_rows, ["metric", "value"]),
        "",
        "## Query Splits",
        "",
        md_table(split_rows, ["split", "records"]),
        "",
        "## Query Domains",
        "",
        md_table(query_domain_rows, ["primary_domain", "records"]),
        "",
        "## Qrel Grades",
        "",
        md_table(qrel_grade_rows, ["grade", "records"]),
        "",
        "## Largest Peer Groups",
        "",
        md_table(peer_rows[:15], ["peer_group", "primary_domain", "source_query_tags", "records"]),
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
    print(f"corpus_docs={summary['fixture_counts']['corpus_docs']}")
    print(f"queries={summary['fixture_counts']['queries']}")
    print(f"qrels={summary['fixture_counts']['qrels']}")
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
