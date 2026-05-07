"""Build and evaluate a tiny retrieval backbone smoke test.

This stage uses retrieval-allowed records only. It builds an in-memory CPU FAISS
index with local embedding model vectors, runs deterministic question-only
queries, and writes chart-ready audit artifacts. It does not train, score
holdout, generate answers, or enable answer-generation records.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from it_support.config import DATA_DIR, INDEX_DIR, MODEL_DIR, PROJECT_ROOT
from it_support.data import DownstreamUse, NormalizedArtifact, load_records
from it_support.data.candidate_sets import unique_expected_domains


OUT = DATA_DIR / "eval" / "retrieval_backbone_smoke_test"
MODEL_KEY = "bge_small_en_v15"
MODEL_PATH = MODEL_DIR / "bge-small-en-v1.5"
INDEX_PATH = INDEX_DIR / "retrieval_smoke_bge_small_en_v15.faiss"
DOCMAP_PATH = INDEX_DIR / "retrieval_smoke_bge_small_en_v15_docmap.jsonl"


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


def compact_text(*parts: str, limit: int = 2400) -> str:
    text = " ".join(part.strip() for part in parts if part and part.strip())
    text = " ".join(text.split())
    return text[:limit]


def answer_text(record: dict[str, Any]) -> str:
    accepted = record.get("accepted_answer") or {}
    top = record.get("top_answer") or {}
    accepted_text = accepted.get("answer_text") or ""
    top_text = top.get("answer_text") or ""
    if top_text and top_text != accepted_text:
        return compact_text(accepted_text, top_text, limit=1600)
    return compact_text(accepted_text, limit=1600)


def document_text(record: dict[str, Any]) -> str:
    return compact_text(
        record.get("title", ""),
        record.get("question_text", ""),
        answer_text(record),
    )


def query_text(record: dict[str, Any]) -> str:
    return compact_text(record.get("title", ""), record.get("question_text", ""), limit=900)


def select_balanced_queries(
    records: list[dict[str, Any]],
    *,
    max_queries: int,
    per_domain: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("primary_domain") or "unknown")].append(record)

    selected = []
    for domain in sorted(grouped):
        candidates = sorted(grouped[domain], key=lambda row: str(row.get("record_id")))
        selected.extend(candidates[:per_domain])

    if len(selected) < max_queries:
        selected_ids = {row["record_id"] for row in selected}
        remainder = [
            row
            for row in sorted(records, key=lambda item: str(item.get("record_id")))
            if row["record_id"] not in selected_ids
        ]
        selected.extend(remainder[: max_queries - len(selected)])

    return selected[:max_queries]


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype("float32")


def load_sentence_transformer(model_path: Path):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for the BGE retrieval smoke test"
        ) from exc

    return SentenceTransformer(str(model_path), local_files_only=True, device="cpu")


def encode_texts(model: Any, texts: list[str], *, batch_size: int) -> np.ndarray:
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype="float32")


def build_faiss_index(vectors: np.ndarray):
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required for the retrieval smoke test") from exc

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def search_index(
    index: Any,
    query_vectors: np.ndarray,
    *,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    scores, indices = index.search(query_vectors, top_k)
    return scores, indices


def index_for_domain(
    domain: str,
    records: list[dict[str, Any]],
    doc_vectors: np.ndarray,
) -> tuple[Any, list[int]]:
    member_indices = [
        index
        for index, record in enumerate(records)
        if domain in unique_expected_domains(record)
    ]
    if not member_indices:
        member_indices = list(range(len(records)))
    vectors = doc_vectors[np.array(member_indices, dtype="int64")]
    return build_faiss_index(vectors), member_indices


def prediction_rows(
    queries: list[dict[str, Any]],
    records: list[dict[str, Any]],
    scores: np.ndarray,
    indices: np.ndarray,
    *,
    profile: str,
    member_indices: list[list[int]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for query_index, query in enumerate(queries):
        retrieved = []
        for rank, doc_index in enumerate(indices[query_index], start=1):
            if doc_index < 0:
                continue
            source_index = (
                member_indices[query_index][int(doc_index)]
                if member_indices is not None
                else int(doc_index)
            )
            doc = records[source_index]
            retrieved.append(
                {
                    "rank": rank,
                    "record_id": doc["record_id"],
                    "score": round(float(scores[query_index][rank - 1]), 6),
                    "primary_domain": doc.get("primary_domain"),
                    "domain_labels": unique_expected_domains(doc),
                    "title": doc.get("title"),
                    "source_url": doc.get("source_url"),
                    "license": doc.get("license"),
                    "commercial_posture": doc.get("commercial_posture"),
                }
            )
        expected_id = query["record_id"]
        ranks = [
            item["rank"]
            for item in retrieved
            if item["record_id"] == expected_id
        ]
        self_rank = ranks[0] if ranks else None
        rows.append(
            {
                "profile": profile,
                "query_id": expected_id,
                "query_title": query.get("title"),
                "query_primary_domain": query.get("primary_domain"),
                "query_domain_labels": unique_expected_domains(query),
                "self_rank": self_rank,
                "hit_at_1": bool(self_rank and self_rank <= 1),
                "hit_at_3": bool(self_rank and self_rank <= 3),
                "hit_at_5": bool(self_rank and self_rank <= 5),
                "mrr": round(1.0 / self_rank, 6) if self_rank else 0.0,
                "retrieved": retrieved,
            }
        )
    return rows


def metrics_from_predictions(rows: list[dict[str, Any]], *, profile: str) -> dict[str, Any]:
    total = len(rows)
    return {
        "profile": profile,
        "queries": total,
        "hit_at_1": round(sum(row["hit_at_1"] for row in rows) / total, 4),
        "hit_at_3": round(sum(row["hit_at_3"] for row in rows) / total, 4),
        "hit_at_5": round(sum(row["hit_at_5"] for row in rows) / total, 4),
        "mrr": round(sum(float(row["mrr"]) for row in rows) / total, 4),
    }


def domain_metric_rows(rows: list[dict[str, Any]], *, profile: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("query_primary_domain") or "unknown")].append(row)
    return [
        {
            "profile": profile,
            "primary_domain": domain,
            **metrics_from_predictions(domain_rows, profile=profile),
        }
        for domain, domain_rows in sorted(grouped.items())
    ]


def build_outputs(
    *,
    max_queries: int,
    per_domain: int,
    top_k: int,
    batch_size: int,
) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    load_result = load_records(
        NormalizedArtifact.STACK_EXCHANGE_RETRIEVAL_CANDIDATES,
        required_uses=(DownstreamUse.RETRIEVAL,),
    )
    records = load_result.records
    queries = select_balanced_queries(records, max_queries=max_queries, per_domain=per_domain)

    if load_result.audit["answer_generation_allowed_records"]:
        raise RuntimeError("Retrieval smoke test refuses answer-generation-enabled records")

    model_start = time.perf_counter()
    model = load_sentence_transformer(MODEL_PATH)
    model_load_seconds = time.perf_counter() - model_start

    doc_texts = [document_text(record) for record in records]
    query_texts = [query_text(record) for record in queries]

    embed_start = time.perf_counter()
    doc_vectors = normalize_matrix(encode_texts(model, doc_texts, batch_size=batch_size))
    query_vectors = normalize_matrix(encode_texts(model, query_texts, batch_size=batch_size))
    embed_seconds = time.perf_counter() - embed_start

    flat_index = build_faiss_index(doc_vectors)
    flat_scores, flat_indices = search_index(flat_index, query_vectors, top_k=top_k)
    flat_predictions = prediction_rows(
        queries,
        records,
        flat_scores,
        flat_indices,
        profile="flat_bge_small_en_v15",
    )

    domain_scores = []
    domain_indices = []
    domain_member_indices = []
    domain_indexes: dict[str, tuple[Any, list[int]]] = {}
    for query_index, query in enumerate(queries):
        domain = str(query.get("primary_domain") or "unknown")
        if domain not in domain_indexes:
            domain_indexes[domain] = index_for_domain(domain, records, doc_vectors)
        domain_index, members = domain_indexes[domain]
        scores, indices = search_index(
            domain_index,
            query_vectors[query_index : query_index + 1],
            top_k=min(top_k, len(members)),
        )
        padded_scores = np.full((top_k,), -1.0, dtype="float32")
        padded_indices = np.full((top_k,), -1, dtype="int64")
        padded_scores[: scores.shape[1]] = scores[0]
        padded_indices[: indices.shape[1]] = indices[0]
        domain_scores.append(padded_scores)
        domain_indices.append(padded_indices)
        domain_member_indices.append(members)
    domain_predictions = prediction_rows(
        queries,
        records,
        np.vstack(domain_scores),
        np.vstack(domain_indices),
        profile="oracle_domain_filtered_bge_small_en_v15",
        member_indices=domain_member_indices,
    )

    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        import faiss

        faiss.write_index(flat_index, str(INDEX_PATH))
    except Exception as exc:  # pragma: no cover - index persistence is diagnostic.
        print(f"warning: could not persist FAISS index: {exc}", file=sys.stderr)
    write_jsonl(
        DOCMAP_PATH,
        [
            {
                "index": index,
                "record_id": record["record_id"],
                "primary_domain": record.get("primary_domain"),
                "source_url": record.get("source_url"),
                "license": record.get("license"),
                "commercial_posture": record.get("commercial_posture"),
            }
            for index, record in enumerate(records)
        ],
    )

    metric_rows = [
        metrics_from_predictions(flat_predictions, profile="flat_bge_small_en_v15"),
        metrics_from_predictions(
            domain_predictions,
            profile="oracle_domain_filtered_bge_small_en_v15",
        ),
    ]
    domain_rows = [
        *domain_metric_rows(flat_predictions, profile="flat_bge_small_en_v15"),
        *domain_metric_rows(
            domain_predictions,
            profile="oracle_domain_filtered_bge_small_en_v15",
        ),
    ]
    query_domain_rows = [
        {"primary_domain": domain, "queries": count}
        for domain, count in sorted(Counter(row.get("primary_domain") for row in queries).items())
    ]

    outputs = {
        "flat_predictions": OUT / "retrieval_predictions_flat_bge_small_en_v15.jsonl",
        "domain_predictions": (
            OUT / "retrieval_predictions_domain_filtered_bge_small_en_v15.jsonl"
        ),
        "metrics": OUT / "retrieval_metrics_bge_small_en_v15.csv",
        "domain_metrics": OUT / "retrieval_domain_metrics_bge_small_en_v15.csv",
        "query_domain_counts": OUT / "retrieval_query_domain_counts_bge_small_en_v15.csv",
        "summary_json": OUT / "retrieval_smoke_summary_bge_small_en_v15.json",
        "summary_md": OUT / "retrieval_smoke_summary_bge_small_en_v15.md",
        "faiss_index": INDEX_PATH,
        "faiss_docmap": DOCMAP_PATH,
    }
    write_jsonl(outputs["flat_predictions"], flat_predictions)
    write_jsonl(outputs["domain_predictions"], domain_predictions)
    write_csv(outputs["metrics"], metric_rows)
    write_csv(outputs["domain_metrics"], domain_rows)
    write_csv(outputs["query_domain_counts"], query_domain_rows)

    elapsed = time.perf_counter() - start
    summary = {
        "stage": "retrieval_backbone_smoke_test",
        "scope": "dev_smoke_retrieval_allowed_only",
        "policy": [
            "Loads only stack_exchange_retrieval_candidates with required_use=retrieval.",
            "Rejects answer-generation-enabled records.",
            "Uses local BGE embeddings and CPU FAISS.",
            "Does not train, score holdout, or generate troubleshooting answers.",
            "Domain-filtered profile is oracle-primary-domain filtering for smoke only.",
        ],
        "model": {
            "key": MODEL_KEY,
            "path": rel(MODEL_PATH),
            "backend": "sentence-transformers",
            "device": "cpu",
        },
        "input_counts": {
            "retrieval_records": len(records),
            "query_records": len(queries),
            "answer_generation_allowed_records": (
                load_result.audit["answer_generation_allowed_records"]
            ),
        },
        "runtime_seconds": {
            "model_load": round(model_load_seconds, 3),
            "embedding": round(embed_seconds, 3),
            "total": round(elapsed, 3),
        },
        "vector_shape": {
            "documents": list(doc_vectors.shape),
            "queries": list(query_vectors.shape),
        },
        "metrics": metric_rows,
        "query_domain_counts": query_domain_rows,
        "outputs": {key: rel(path) for key, path in outputs.items()},
    }
    write_json(outputs["summary_json"], summary)

    lines = [
        "# Retrieval Backbone Smoke Test",
        "",
        "This is a bounded retrieval sanity check over retrieval-allowed records only. "
        "It does not train, score holdout, or generate answers.",
        "",
        "## Policy",
        "",
        *[f"- {item}" for item in summary["policy"]],
        "",
        "## Model",
        "",
        md_table([summary["model"]], ["key", "backend", "device", "path"]),
        "",
        "## Counts",
        "",
        md_table([summary["input_counts"]], list(summary["input_counts"])),
        "",
        "## Metrics",
        "",
        md_table(metric_rows, ["profile", "queries", "hit_at_1", "hit_at_3", "hit_at_5", "mrr"]),
        "",
        "## Query Domains",
        "",
        md_table(query_domain_rows, ["primary_domain", "queries"]),
        "",
        "## Runtime Seconds",
        "",
        md_table([summary["runtime_seconds"]], ["model_load", "embedding", "total"]),
        "",
        "## Outputs",
        "",
        *[f"- `{rel(path)}`" for path in outputs.values()],
    ]
    outputs["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-queries", type=int, default=96)
    parser.add_argument("--per-domain", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--summary", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_outputs(
        max_queries=args.max_queries,
        per_domain=args.per_domain,
        top_k=args.top_k,
        batch_size=args.batch_size,
    )
    print(f"wrote {summary['outputs']['summary_md']}")
    print(f"retrieval_records={summary['input_counts']['retrieval_records']}")
    print(f"query_records={summary['input_counts']['query_records']}")
    for row in summary["metrics"]:
        print(
            f"{row['profile']}: hit@1={row['hit_at_1']} "
            f"hit@5={row['hit_at_5']} mrr={row['mrr']}"
        )
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
