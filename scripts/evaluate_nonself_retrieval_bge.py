"""Score dev-only non-self retrieval with local embedding models.

This script consumes the non-self retrieval fixture. It excludes the query's
own source document from each ranked list, compares flat retrieval with
rule-triage-filtered retrieval, and writes chart-ready metrics. Holdout rows are
left untouched unless a different split is explicitly requested.
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

from it_support.config import DATA_DIR, LOCAL_MODELS, PROJECT_ROOT
from it_support.retrieval.eval_scoring import (
    aggregate_metrics,
    domain_metric_rows,
    group_qrels,
    make_retrieved_items,
)
from it_support.triage.rule_based import predict_case


FIXTURE = DATA_DIR / "eval" / "nonself_retrieval_eval_fixture"
MODEL_EVAL_CONFIGS: dict[str, dict[str, str | None]] = {
    "bge_small_en_v15": {
        "output_dir": "nonself_retrieval_bge_dev",
        "file_suffix": "bge_small_en_v15",
        "profile_suffix": "bge_small_en_v15",
        "query_prompt_name": None,
        "document_prompt_name": None,
    },
    "qwen3_embedding_06b": {
        "output_dir": "nonself_retrieval_qwen3_dev",
        "file_suffix": "qwen3_embedding_06b",
        "profile_suffix": "qwen3_embedding_06b",
        "query_prompt_name": "query",
        "document_prompt_name": None,
    },
}


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


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


def model_eval_config(model_key: str) -> dict[str, str | None]:
    if model_key not in MODEL_EVAL_CONFIGS:
        known = ", ".join(sorted(MODEL_EVAL_CONFIGS))
        raise ValueError(f"Unsupported retrieval eval model {model_key!r}; expected {known}")
    if model_key not in LOCAL_MODELS:
        raise ValueError(f"Model {model_key!r} is not registered in LOCAL_MODELS")
    model = LOCAL_MODELS[model_key]
    if model.backend != "sentence-transformers":
        raise ValueError(f"Model {model_key!r} is not a sentence-transformers model")
    if not model.path.exists():
        raise FileNotFoundError(f"Model path does not exist for {model_key}: {model.path}")
    return MODEL_EVAL_CONFIGS[model_key]


def torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_device(requested_device: str, *, require_cuda: bool) -> str:
    if requested_device == "cpu":
        if require_cuda:
            raise RuntimeError("--require-cuda cannot be used with --device cpu")
        return "cpu"
    if requested_device == "cuda":
        if not torch_cuda_available():
            raise RuntimeError(
                "CUDA was requested but is not available in this Python environment. "
                "Install a CUDA-enabled torch build or run without --require-cuda."
            )
        return "cuda"
    if requested_device == "auto":
        if torch_cuda_available():
            return "cuda"
        if require_cuda:
            raise RuntimeError(
                "--device auto resolved to CPU because CUDA is not available. "
                "Refusing to run because --require-cuda was set."
            )
        return "cpu"
    raise ValueError(f"Unsupported device {requested_device!r}")


def load_sentence_transformer(model_path: Path, *, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for embedding retrieval scoring"
        ) from exc

    return SentenceTransformer(str(model_path), local_files_only=True, device=device)


def encode_texts(
    model: Any,
    texts: list[str],
    *,
    batch_size: int,
    prompt_name: str | None = None,
) -> np.ndarray:
    kwargs = {
        "batch_size": batch_size,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    if prompt_name:
        kwargs["prompt_name"] = prompt_name
    vectors = model.encode(texts, **kwargs)
    return np.asarray(vectors, dtype="float32")


def build_faiss_index(vectors: np.ndarray):
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required for BGE retrieval scoring") from exc

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def select_queries(
    queries: list[dict[str, Any]],
    *,
    split: str,
    max_queries: int | None,
) -> list[dict[str, Any]]:
    split_queries = [row for row in queries if row.get("split") == split]
    if not max_queries or len(split_queries) <= max_queries:
        return sorted(split_queries, key=lambda row: str(row["query_id"]))

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in split_queries:
        grouped[str(query.get("expected_primary_domain") or "unknown")].append(query)

    selected = []
    per_domain = max(1, max_queries // max(1, len(grouped)))
    for domain in sorted(grouped):
        selected.extend(sorted(grouped[domain], key=lambda row: str(row["query_id"]))[:per_domain])
    if len(selected) >= max_queries:
        return selected[:max_queries]

    selected_ids = {row["query_id"] for row in selected}
    remainder = [
        row
        for row in sorted(split_queries, key=lambda item: str(item["query_id"]))
        if row["query_id"] not in selected_ids
    ]
    selected.extend(remainder[: max_queries - len(selected)])
    return selected[:max_queries]


def query_case_for_triage(query: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": query["query_id"],
        "title": query.get("title", ""),
        "question_text": query.get("query_text", ""),
        "question_tags": query.get("question_tags", []),
        "query_tags": query.get("query_tags", []),
    }


def predicted_domains(query: dict[str, Any]) -> list[str]:
    prediction = predict_case(
        query_case_for_triage(query),
        include_source_tags=True,
        max_domains=3,
    )
    return [match.label for match in prediction.domains]


def member_indices_for_domains(docs: list[dict[str, Any]], domains: list[str]) -> list[int]:
    wanted = set(domains)
    members = [
        index
        for index, doc in enumerate(docs)
        if wanted.intersection(set(doc.get("domain_labels") or [doc.get("primary_domain")]))
    ]
    return members or list(range(len(docs)))


def search_profile(
    *,
    profile: str,
    queries: list[dict[str, Any]],
    docs: list[dict[str, Any]],
    doc_vectors: np.ndarray,
    query_vectors: np.ndarray,
    qrels_by_query: dict[str, dict[str, int]],
    top_k: int,
    search_k: int,
    domain_mode: str,
) -> list[dict[str, Any]]:
    flat_index = None
    domain_indexes: dict[tuple[str, ...], tuple[Any, list[int]]] = {}
    rows = []

    if domain_mode == "flat":
        flat_index = build_faiss_index(doc_vectors)

    for query_index, query in enumerate(queries):
        query_id = str(query["query_id"])
        triage_domains = predicted_domains(query)
        filter_domains: list[str] = []
        if domain_mode == "triage":
            filter_domains = triage_domains
        elif domain_mode == "oracle":
            filter_domains = [str(query.get("expected_primary_domain") or "unknown")]

        if domain_mode == "flat":
            assert flat_index is not None
            scores, indices = flat_index.search(
                query_vectors[query_index : query_index + 1],
                min(search_k, len(docs)),
            )
            member_indices = None
            pool_size = len(docs)
        else:
            members = member_indices_for_domains(docs, filter_domains)
            key = tuple(sorted(filter_domains)) or ("all",)
            if key not in domain_indexes:
                vectors = doc_vectors[np.array(members, dtype="int64")]
                domain_indexes[key] = (build_faiss_index(vectors), members)
            index, member_indices = domain_indexes[key]
            scores, indices = index.search(
                query_vectors[query_index : query_index + 1],
                min(search_k, len(member_indices)),
            )
            pool_size = len(member_indices)

        retrieved = make_retrieved_items(
            query_id=query_id,
            docs=docs,
            scores=[float(value) for value in scores[0]],
            indices=[int(value) for value in indices[0]],
            qrels_for_query=qrels_by_query.get(query_id, {}),
            member_indices=member_indices,
            top_k=top_k,
        )
        rows.append(
            {
                "profile": profile,
                "query_id": query_id,
                "split": query.get("split"),
                "title": query.get("title", ""),
                "expected_primary_domain": query.get("expected_primary_domain"),
                "expected_domains": query.get("expected_domains", []),
                "triage_predicted_domains": triage_domains,
                "filter_domains": filter_domains,
                "candidate_pool_size": pool_size,
                "retrieved": retrieved,
            }
        )
    return rows


def query_detail_rows(
    prediction_rows: list[dict[str, Any]],
    qrels_by_query: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    rows = []
    for row in prediction_rows:
        retrieved = row["retrieved"]
        first_any = next(
            (item["rank"] for item in retrieved if int(item["relevance_grade"]) >= 1),
            "",
        )
        first_grade_2 = next(
            (item["rank"] for item in retrieved if int(item["relevance_grade"]) >= 2),
            "",
        )
        qrels = qrels_by_query.get(str(row["query_id"]), {})
        rows.append(
            {
                "profile": row["profile"],
                "query_id": row["query_id"],
                "split": row["split"],
                "expected_primary_domain": row["expected_primary_domain"],
                "triage_predicted_domains": "|".join(row["triage_predicted_domains"]),
                "filter_domains": "|".join(row["filter_domains"]),
                "candidate_pool_size": row["candidate_pool_size"],
                "available_positive_docs": sum(1 for grade in qrels.values() if grade >= 1),
                "available_grade_2_docs": sum(1 for grade in qrels.values() if grade >= 2),
                "first_any_grade_rank": first_any,
                "first_grade_2_rank": first_grade_2,
                "top_doc_id": retrieved[0]["doc_id"] if retrieved else "",
                "top_relevance_grade": retrieved[0]["relevance_grade"] if retrieved else "",
            }
        )
    return rows


def build_outputs(
    *,
    model_key: str,
    split: str,
    max_queries: int | None,
    top_k: int,
    search_k: int,
    batch_size: int,
    include_oracle: bool,
    device: str,
    require_cuda: bool,
) -> dict[str, Any]:
    eval_config = model_eval_config(model_key)
    model_info = LOCAL_MODELS[model_key]
    output_dir = DATA_DIR / "eval" / str(eval_config["output_dir"])
    file_suffix = str(eval_config["file_suffix"])
    profile_suffix = str(eval_config["profile_suffix"])
    flat_profile = f"flat_{profile_suffix}"
    triage_profile = f"rule_triage_filtered_{profile_suffix}"
    oracle_profile = f"oracle_domain_filtered_{profile_suffix}"
    query_prompt_name = eval_config["query_prompt_name"]
    document_prompt_name = eval_config["document_prompt_name"]
    resolved_device = resolve_device(device, require_cuda=require_cuda)

    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    docs = read_jsonl(FIXTURE / "retrieval_corpus.jsonl")
    all_queries = read_jsonl(FIXTURE / "retrieval_queries.jsonl")
    qrels = read_jsonl(FIXTURE / "retrieval_qrels.jsonl")
    queries = select_queries(all_queries, split=split, max_queries=max_queries)
    qrels_by_query = group_qrels(qrels)

    model_start = time.perf_counter()
    model = load_sentence_transformer(model_info.path, device=resolved_device)
    model_load_seconds = time.perf_counter() - model_start

    embed_start = time.perf_counter()
    doc_vectors = encode_texts(
        model,
        [str(doc.get("document_text") or "") for doc in docs],
        batch_size=batch_size,
        prompt_name=str(document_prompt_name) if document_prompt_name else None,
    )
    query_vectors = encode_texts(
        model,
        [str(query.get("query_text") or "") for query in queries],
        batch_size=batch_size,
        prompt_name=str(query_prompt_name) if query_prompt_name else None,
    )
    embed_seconds = time.perf_counter() - embed_start

    profiles = [
        (flat_profile, "flat"),
        (triage_profile, "triage"),
    ]
    if include_oracle:
        profiles.append((oracle_profile, "oracle"))

    predictions_by_profile = {}
    for profile, mode in profiles:
        predictions_by_profile[profile] = search_profile(
            profile=profile,
            queries=queries,
            docs=docs,
            doc_vectors=doc_vectors,
            query_vectors=query_vectors,
            qrels_by_query=qrels_by_query,
            top_k=top_k,
            search_k=max(search_k, top_k + 5),
            domain_mode=mode,
        )

    prediction_rows = [
        row for profile_rows in predictions_by_profile.values() for row in profile_rows
    ]
    scopes = [("any_grade", 1), ("grade_2", 2)]
    metric_rows = []
    domain_rows = []
    for profile, rows in predictions_by_profile.items():
        for scope, min_grade in scopes:
            metric_rows.append(
                aggregate_metrics(
                    rows,
                    qrels_by_query,
                    profile=profile,
                    relevance_scope=scope,
                    min_grade=min_grade,
                    cutoff=top_k,
                )
            )
            domain_rows.extend(
                domain_metric_rows(
                    rows,
                    qrels_by_query,
                    profile=profile,
                    relevance_scope=scope,
                    min_grade=min_grade,
                    cutoff=top_k,
                )
            )

    detail_rows = query_detail_rows(prediction_rows, qrels_by_query)
    pool_rows = [
        {
            "profile": profile,
            "median_candidate_pool_size": float(
                np.median([row["candidate_pool_size"] for row in rows])
            ),
            "min_candidate_pool_size": min(row["candidate_pool_size"] for row in rows),
            "max_candidate_pool_size": max(row["candidate_pool_size"] for row in rows),
        }
        for profile, rows in predictions_by_profile.items()
    ]
    triage_domain_rows = [
        {"predicted_primary_domain": domain, "queries": count}
        for domain, count in Counter(
            row["triage_predicted_domains"][0] if row["triage_predicted_domains"] else "none"
            for row in predictions_by_profile[flat_profile]
        ).most_common()
    ]

    suffix = "sample" if max_queries else "dev"
    outputs = {
        "predictions": output_dir / f"retrieval_predictions_{suffix}_{file_suffix}.jsonl",
        "metrics": output_dir / f"retrieval_metrics_{suffix}_{file_suffix}.csv",
        "domain_metrics": output_dir / f"retrieval_domain_metrics_{suffix}_{file_suffix}.csv",
        "query_details": output_dir / f"retrieval_query_details_{suffix}_{file_suffix}.csv",
        "candidate_pool_metrics": output_dir / (
            f"retrieval_candidate_pool_metrics_{suffix}_{file_suffix}.csv"
        ),
        "triage_domain_counts": output_dir / (
            f"retrieval_triage_domain_counts_{suffix}_{file_suffix}.csv"
        ),
        "summary_json": output_dir / f"retrieval_eval_summary_{suffix}_{file_suffix}.json",
        "summary_md": output_dir / f"retrieval_eval_summary_{suffix}_{file_suffix}.md",
    }
    write_jsonl(outputs["predictions"], prediction_rows)
    write_csv(outputs["metrics"], metric_rows)
    write_csv(outputs["domain_metrics"], domain_rows)
    write_csv(outputs["query_details"], detail_rows)
    write_csv(outputs["candidate_pool_metrics"], pool_rows)
    write_csv(outputs["triage_domain_counts"], triage_domain_rows)

    elapsed = time.perf_counter() - start
    summary = {
        "stage": "nonself_retrieval_embedding_dev",
        "scope": "dev_only" if not max_queries else "sample_dev_only",
        "split": split,
        "policy": [
            "Consumes the non-self retrieval fixture.",
            "Scores only the requested query split.",
            "Removes the query's own source document from ranked results.",
            f"Uses local {model_key} embeddings and CPU FAISS.",
            "Compares flat retrieval with rule-triage-filtered retrieval.",
            "Oracle-domain-filtered retrieval is diagnostic only when included.",
            "Does not score holdout unless a holdout split is explicitly requested.",
            "Does not train or generate answers.",
        ],
        "model": {
            "key": model_key,
            "path": rel(model_info.path),
            "backend": model_info.backend,
            "device": resolved_device,
            "requested_device": device,
            "require_cuda": require_cuda,
            "query_prompt_name": query_prompt_name or "",
            "document_prompt_name": document_prompt_name or "",
        },
        "input_counts": {
            "corpus_docs": len(docs),
            "available_queries": len([row for row in all_queries if row.get("split") == split]),
            "scored_queries": len(queries),
            "qrels": len(qrels),
        },
        "runtime_seconds": {
            "model_load": round(model_load_seconds, 3),
            "embedding": round(embed_seconds, 3),
            "total": round(elapsed, 3),
        },
        "settings": {
            "top_k": top_k,
            "search_k": max(search_k, top_k + 5),
            "batch_size": batch_size,
            "include_oracle": include_oracle,
            "max_queries": max_queries,
            "requested_device": device,
            "resolved_device": resolved_device,
            "require_cuda": require_cuda,
        },
        "metrics": metric_rows,
        "candidate_pool_metrics": pool_rows,
        "outputs": {key: rel(path) for key, path in outputs.items()},
    }
    write_json(outputs["summary_json"], summary)

    lines = [
        f"# Non-Self Retrieval {model_key} Dev Evaluation",
        "",
        "This stage scores dev queries against non-self qrels. It does not train, "
        "generate answers, or score holdout.",
        "",
        "## Policy",
        "",
        *[f"- {item}" for item in summary["policy"]],
        "",
        "## Model",
        "",
        md_table(
            [summary["model"]],
            [
                "key",
                "backend",
                "device",
                "requested_device",
                "require_cuda",
                "path",
                "query_prompt_name",
            ],
        ),
        "",
        "## Counts",
        "",
        md_table([summary["input_counts"]], list(summary["input_counts"])),
        "",
        "## Metrics",
        "",
        md_table(
            metric_rows,
            [
                "profile",
                "relevance_scope",
                "queries",
                "queries_without_positives",
                "hit_at_1",
                "hit_at_3",
                "hit_at_5",
                "hit_at_10",
                "mrr",
                "ndcg",
            ],
        ),
        "",
        "## Candidate Pools",
        "",
        md_table(
            pool_rows,
            [
                "profile",
                "median_candidate_pool_size",
                "min_candidate_pool_size",
                "max_candidate_pool_size",
            ],
        ),
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
    parser.add_argument(
        "--model-key",
        choices=sorted(MODEL_EVAL_CONFIGS),
        default="bge_small_en_v15",
    )
    parser.add_argument("--split", default="retrieval_dev_eval")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--search-k", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
        help="Embedding device. Use auto or cuda for GPU runs.",
    )
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail fast if CUDA is unavailable instead of falling back to CPU.",
    )
    parser.add_argument("--no-oracle", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_outputs(
        model_key=args.model_key,
        split=args.split,
        max_queries=args.max_queries,
        top_k=args.top_k,
        search_k=args.search_k,
        batch_size=args.batch_size,
        include_oracle=not args.no_oracle,
        device=args.device,
        require_cuda=args.require_cuda,
    )
    print(f"wrote {summary['outputs']['summary_md']}")
    print(f"scored_queries={summary['input_counts']['scored_queries']}")
    for row in summary["metrics"]:
        print(
            f"{row['profile']} {row['relevance_scope']}: "
            f"hit@5={row['hit_at_5']} mrr={row['mrr']} ndcg={row['ndcg']}"
        )
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
