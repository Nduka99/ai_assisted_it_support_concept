"""Run a dev-only embedding nearest-neighbor classifier baseline.

Defaults are intentionally bounded for smoke testing. Full dev runs and larger
models must be requested explicitly. This script does not read holdout, train,
call an LLM, or generate answers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from it_support.classification.embedding_classifier import (
    compact_case_text,
    prediction_record_from_neighbors,
    select_balanced_cases,
    top_neighbors,
)
from it_support.classification.ladder import (
    multilabel_domain_report,
    primary_multiclass_report,
    ranked_domain_report,
    routing_case_detail_rows,
)
from it_support.config import DATA_DIR, LOCAL_MODELS, PROJECT_ROOT
from it_support.schemas import DOMAIN_LABELS


SPLITS_DIR = DATA_DIR / "eval" / "candidate_loader_and_eval_splits"
OUT_ROOT = DATA_DIR / "eval" / "embedding_classifier_dev"
MODEL_CONFIGS = {
    "bge_small_en_v15": {
        "file_suffix": "bge_small_en_v15",
        "query_prompt_name": None,
        "document_prompt_name": None,
    },
    "qwen3_embedding_06b": {
        "file_suffix": "qwen3_embedding_06b",
        "query_prompt_name": "query",
        "document_prompt_name": None,
    },
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
            raise RuntimeError("CUDA was requested but is not available.")
        return "cuda"
    if requested_device == "auto":
        if torch_cuda_available():
            return "cuda"
        if require_cuda:
            raise RuntimeError("--device auto resolved to CPU but --require-cuda was set.")
        return "cpu"
    raise ValueError(f"Unsupported device {requested_device!r}")


def model_config(model_key: str) -> dict[str, str | None]:
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported embedding classifier model: {model_key}")
    if model_key not in LOCAL_MODELS:
        raise ValueError(f"Model {model_key!r} is not registered")
    model = LOCAL_MODELS[model_key]
    if model.backend != "sentence-transformers":
        raise ValueError(f"Model {model_key!r} is not a sentence-transformers model")
    if not model.path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model.path}")
    return MODEL_CONFIGS[model_key]


def load_sentence_transformer(model_path: Path, *, device: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required") from exc
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
    return np.asarray(model.encode(texts, **kwargs), dtype="float32")


def build_outputs(
    *,
    model_key: str,
    include_tags: bool,
    full_dev: bool,
    max_eval_cases: int,
    max_reference_cases: int,
    neighbor_k: int,
    max_domains: int,
    label_threshold_ratio: float,
    batch_size: int,
    device: str,
    require_cuda: bool,
) -> dict[str, Any]:
    start = time.perf_counter()
    config = model_config(model_key)
    model_info = LOCAL_MODELS[model_key]
    resolved_device = resolve_device(device, require_cuda=require_cuda)
    profile = (
        f"embedding_knn_metadata_augmented_{config['file_suffix']}"
        if include_tags
        else f"embedding_knn_text_only_{config['file_suffix']}"
    )
    scope = "dev_full" if full_dev else "dev_sample"
    output_dir = OUT_ROOT / str(config["file_suffix"])
    output_dir.mkdir(parents=True, exist_ok=True)

    routing_cases_path = SPLITS_DIR / "routing_eval_dev.jsonl"
    all_cases = read_jsonl(routing_cases_path)
    if any(row.get("split") != "routing_dev_eval" for row in all_cases):
        raise ValueError("Embedding classifier baseline only accepts routing_dev_eval rows")

    eval_cases = (
        select_balanced_cases(all_cases, max_cases=None)
        if full_dev
        else select_balanced_cases(all_cases, max_cases=max_eval_cases)
    )
    reference_cases = (
        select_balanced_cases(all_cases, max_cases=None)
        if full_dev
        else select_balanced_cases(all_cases, max_cases=max_reference_cases)
    )

    model_start = time.perf_counter()
    model = load_sentence_transformer(model_info.path, device=resolved_device)
    model_load_seconds = time.perf_counter() - model_start

    embed_start = time.perf_counter()
    document_prompt = config["document_prompt_name"]
    query_prompt = config["query_prompt_name"]
    reference_vectors = encode_texts(
        model,
        [compact_case_text(case, include_tags=include_tags) for case in reference_cases],
        batch_size=batch_size,
        prompt_name=str(document_prompt) if document_prompt else None,
    )
    eval_vectors = encode_texts(
        model,
        [compact_case_text(case, include_tags=include_tags) for case in eval_cases],
        batch_size=batch_size,
        prompt_name=str(query_prompt) if query_prompt else None,
    )
    embed_seconds = time.perf_counter() - embed_start

    scores = np.matmul(eval_vectors, reference_vectors.T)
    predictions = []
    for index, case in enumerate(eval_cases):
        neighbors = top_neighbors(
            query_case=case,
            reference_cases=reference_cases,
            scores=scores[index],
            top_k=neighbor_k,
        )
        predictions.append(
            prediction_record_from_neighbors(
                query_case=case,
                neighbors=neighbors,
                profile=profile,
                max_domains=max_domains,
                label_threshold_ratio=label_threshold_ratio,
            )
        )

    primary_summary, primary_rows, confusion_rows = primary_multiclass_report(
        predictions,
        labels=DOMAIN_LABELS,
        profile=profile,
    )
    multilabel_summary, multilabel_rows = multilabel_domain_report(
        predictions,
        labels=DOMAIN_LABELS,
        profile=profile,
    )
    ranked_summary = ranked_domain_report(predictions, profile=profile)
    detail_rows = routing_case_detail_rows(predictions, profile=profile)

    suffix = f"{scope}_{'metadata' if include_tags else 'text'}_{config['file_suffix']}"
    outputs = {
        "predictions": output_dir / f"embedding_classifier_predictions_{suffix}.jsonl",
        "primary_metrics": output_dir / f"embedding_classifier_primary_metrics_{suffix}.csv",
        "primary_per_domain": output_dir / f"embedding_classifier_primary_per_domain_{suffix}.csv",
        "primary_confusion": output_dir / f"embedding_classifier_primary_confusion_{suffix}.csv",
        "multilabel_metrics": output_dir / f"embedding_classifier_multilabel_metrics_{suffix}.csv",
        "multilabel_per_domain": output_dir
        / f"embedding_classifier_multilabel_per_domain_{suffix}.csv",
        "ranked_metrics": output_dir / f"embedding_classifier_ranked_metrics_{suffix}.csv",
        "routing_details": output_dir / f"embedding_classifier_routing_details_{suffix}.csv",
        "summary_json": output_dir / f"embedding_classifier_summary_{suffix}.json",
        "summary_md": output_dir / f"embedding_classifier_summary_{suffix}.md",
    }
    write_jsonl(outputs["predictions"], predictions)
    write_csv(outputs["primary_metrics"], [primary_summary])
    write_csv(outputs["primary_per_domain"], primary_rows)
    write_csv(outputs["primary_confusion"], confusion_rows)
    write_csv(outputs["multilabel_metrics"], [multilabel_summary])
    write_csv(outputs["multilabel_per_domain"], multilabel_rows)
    write_csv(outputs["ranked_metrics"], [ranked_summary])
    write_csv(outputs["routing_details"], detail_rows)

    elapsed = time.perf_counter() - start
    summary = {
        "stage": "embedding_classifier_baseline",
        "scope": scope,
        "policy": [
            "Consumes routing dev fixtures only.",
            "Uses local sentence-transformers embeddings with nearest-neighbor label voting.",
            "Excludes the query case itself from its neighbor list.",
            "Does not read holdout, train, call an LLM, or generate answers.",
            "Default mode is a bounded smoke sample; full dev requires --full-dev.",
        ],
        "model": {
            "key": model_key,
            "path": rel(model_info.path),
            "backend": model_info.backend,
            "requested_device": device,
            "resolved_device": resolved_device,
            "require_cuda": require_cuda,
            "query_prompt_name": query_prompt or "",
            "document_prompt_name": document_prompt or "",
        },
        "settings": {
            "include_tags": include_tags,
            "neighbor_k": neighbor_k,
            "max_domains": max_domains,
            "label_threshold_ratio": label_threshold_ratio,
            "batch_size": batch_size,
            "full_dev": full_dev,
            "max_eval_cases": None if full_dev else max_eval_cases,
            "max_reference_cases": None if full_dev else max_reference_cases,
        },
        "counts": {
            "available_routing_dev_cases": len(all_cases),
            "eval_cases": len(eval_cases),
            "reference_cases": len(reference_cases),
            "predictions": len(predictions),
        },
        "runtime_seconds": {
            "model_load": round(model_load_seconds, 3),
            "embedding": round(embed_seconds, 3),
            "total": round(elapsed, 3),
        },
        "primary_metrics": primary_summary,
        "multilabel_metrics": multilabel_summary,
        "ranked_metrics": ranked_summary,
        "outputs": {key: rel(path) for key, path in outputs.items()},
    }
    write_json(outputs["summary_json"], summary)

    overview = {
        "profile": profile,
        "scope": scope,
        "eval_cases": len(eval_cases),
        "reference_cases": len(reference_cases),
        "primary_accuracy": primary_summary["accuracy"],
        "multilabel_micro_f1": multilabel_summary["micro_f1"],
        "multilabel_exact_match": multilabel_summary["exact_match_ratio"],
        "ranked_primary_hit_at_3": ranked_summary["primary_hit_at_3"],
        "ranked_graded_ndcg_at_3": ranked_summary["graded_ndcg_at_3"],
    }
    lines = [
        "# Embedding Classifier Baseline",
        "",
        "This is a dev-only nearest-neighbor classifier baseline over local embeddings. "
        "It does not train, read holdout, call an LLM, or generate answers.",
        "",
        "## Policy",
        "",
        *[f"- {item}" for item in summary["policy"]],
        "",
        "## Overview",
        "",
        md_table([overview], list(overview)),
        "",
        "## Model",
        "",
        md_table([summary["model"]], list(summary["model"])),
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
    parser.add_argument("--model-key", choices=sorted(MODEL_CONFIGS), default="bge_small_en_v15")
    parser.add_argument("--include-tags", action="store_true")
    parser.add_argument("--full-dev", action="store_true")
    parser.add_argument("--max-eval-cases", type=int, default=48)
    parser.add_argument("--max-reference-cases", type=int, default=192)
    parser.add_argument("--neighbor-k", type=int, default=7)
    parser.add_argument("--max-domains", type=int, default=4)
    parser.add_argument("--label-threshold-ratio", type=float, default=0.35)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_outputs(
        model_key=args.model_key,
        include_tags=args.include_tags,
        full_dev=args.full_dev,
        max_eval_cases=args.max_eval_cases,
        max_reference_cases=args.max_reference_cases,
        neighbor_k=args.neighbor_k,
        max_domains=args.max_domains,
        label_threshold_ratio=args.label_threshold_ratio,
        batch_size=args.batch_size,
        device=args.device,
        require_cuda=args.require_cuda,
    )
    print(f"wrote {summary['outputs']['summary_md']}")
    print(
        f"scope={summary['scope']} eval_cases={summary['counts']['eval_cases']} "
        f"reference_cases={summary['counts']['reference_cases']}"
    )
    print(
        f"primary_acc={summary['primary_metrics']['accuracy']} "
        f"multilabel_micro_f1={summary['multilabel_metrics']['micro_f1']} "
        f"ranked_hit@3={summary['ranked_metrics']['primary_hit_at_3']}"
    )
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
