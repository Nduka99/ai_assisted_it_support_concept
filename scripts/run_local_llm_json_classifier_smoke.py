"""Run a tiny local Transformers LLM JSON-classifier smoke on saved requests.

Defaults are intentionally small. This script consumes dev request artifacts,
loads one local Transformers model, writes saved response rows, and does not
read holdout, train, fine-tune, or generate troubleshooting answers.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from it_support.config import DATA_DIR, LOCAL_MODELS, PROJECT_ROOT  # noqa: E402


DEFAULT_REQUESTS = (
    DATA_DIR
    / "eval"
    / "llm_json_classifier_dry_run"
    / "llm_json_classifier_requests_dev_sample_metadata.jsonl"
)
OUT = DATA_DIR / "eval" / "llm_json_classifier_responses"
SUPPORTED_BACKENDS = {"transformers"}


def rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


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


def request_rows(path: Path, *, max_cases: int) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if max_cases <= 0:
        raise ValueError("--max-cases must be positive")
    selected = rows[:max_cases]
    if not selected:
        raise ValueError(f"No request rows found in {path}")
    if any(row.get("split") != "routing_dev_eval" for row in selected):
        raise ValueError("LLM smoke only accepts routing_dev_eval request rows")
    return selected


def torch_cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_dtype(dtype: str) -> Any:
    if dtype == "auto":
        return "auto"
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return mapping[dtype]


def load_transformers_model(
    model_path: Path,
    *,
    dtype: str,
    device_map: str,
) -> tuple[Any, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=False,
        dtype=resolve_dtype(dtype),
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return processor, model


def generation_inputs(processor: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    block_messages = [
        {
            "role": message["role"],
            "content": [{"type": "text", "text": message["content"]}],
        }
        for message in messages
    ]
    inputs = processor.apply_chat_template(
        block_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return dict(inputs)


def input_device(model: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def move_inputs(inputs: dict[str, Any], device: Any) -> dict[str, Any]:
    moved = {}
    for key, value in inputs.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def decode_generated_text(processor: Any, output_ids: Any, *, prompt_length: int) -> str:
    generated = output_ids[0][prompt_length:]
    decoder = getattr(processor, "decode", None)
    if decoder is None:
        decoder = processor.tokenizer.decode
    return decoder(generated, skip_special_tokens=True).strip()


def generate_one(
    *,
    processor: Any,
    model: Any,
    request: dict[str, Any],
    max_new_tokens: int,
) -> dict[str, Any]:
    import torch

    started = time.perf_counter()
    inputs = generation_inputs(processor, request["messages"])
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    inputs = move_inputs(inputs, input_device(model))
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    response_text = decode_generated_text(processor, output_ids, prompt_length=prompt_tokens)
    return {
        "case_id": request["case_id"],
        "split": request.get("split"),
        "profile": request.get("profile"),
        "schema_name": request.get("schema_name"),
        "response_text": response_text,
        "input_tokens": prompt_tokens,
        "output_tokens": int(output_ids.shape[-1] - prompt_tokens),
        "runtime_seconds": round(time.perf_counter() - started, 3),
    }


def build_outputs(
    *,
    model_key: str,
    requests_path: Path,
    max_cases: int,
    max_new_tokens: int,
    dtype: str,
    device_map: str,
    require_cuda: bool,
    run_name: str,
    preflight_only: bool,
) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    if model_key not in LOCAL_MODELS:
        raise ValueError(f"Unknown local model key: {model_key}")
    model_info = LOCAL_MODELS[model_key]
    if model_info.backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"{model_key} uses backend {model_info.backend!r}; "
            f"supported smoke backends are {sorted(SUPPORTED_BACKENDS)}"
        )
    if not model_info.path.exists():
        raise FileNotFoundError(f"Local model path is missing: {model_info.path}")
    if require_cuda and not torch_cuda_available():
        raise RuntimeError("--require-cuda was set but CUDA is not available")

    rows = request_rows(requests_path, max_cases=max_cases)
    suffix = f"{run_name}_{model_key}"
    responses_path = OUT / f"llm_json_classifier_responses_{suffix}.jsonl"
    summary_path = OUT / f"llm_json_classifier_response_smoke_summary_{suffix}.json"
    summary_md_path = OUT / f"llm_json_classifier_response_smoke_summary_{suffix}.md"

    started = time.perf_counter()
    responses = []
    load_seconds = 0.0
    status = "preflight_only"
    error = ""
    if not preflight_only:
        try:
            load_started = time.perf_counter()
            processor, model = load_transformers_model(
                model_info.path,
                dtype=dtype,
                device_map=device_map,
            )
            load_seconds = time.perf_counter() - load_started
            for row in rows:
                responses.append(
                    generate_one(
                        processor=processor,
                        model=model,
                        request=row,
                        max_new_tokens=max_new_tokens,
                    )
                )
            write_jsonl(responses_path, responses)
            status = "completed"
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
            write_jsonl(responses_path, responses)

    summary = {
        "stage": "local_llm_json_classifier_smoke",
        "scope": "routing_dev_eval_tiny_smoke",
        "status": status,
        "policy": [
            "Consumes dev request artifacts only.",
            "Does not read holdout files.",
            "Does not train, fine-tune, or generate troubleshooting answers.",
            "Writes raw saved response rows for the separate evaluator.",
        ],
        "model": {
            "key": model_info.key,
            "path": rel(model_info.path),
            "backend": model_info.backend,
            "dtype": dtype,
            "device_map": device_map,
            "require_cuda": require_cuda,
        },
        "settings": {
            "max_cases": max_cases,
            "max_new_tokens": max_new_tokens,
            "preflight_only": preflight_only,
            "run_name": run_name,
        },
        "counts": {
            "request_rows_selected": len(rows),
            "responses_written": len(responses),
        },
        "runtime_seconds": {
            "model_load": round(load_seconds, 3),
            "total": round(time.perf_counter() - started, 3),
        },
        "error": error,
        "inputs": {
            "requests": rel(requests_path),
        },
        "outputs": {
            "responses": rel(responses_path),
            "summary_json": rel(summary_path),
            "summary_md": rel(summary_md_path),
        },
    }
    write_json(summary_path, summary)

    lines = [
        "# Local LLM JSON Classifier Smoke",
        "",
        "This tiny smoke writes raw local LLM classifier responses for the saved-response "
        "evaluator. It is not a fine-tune, not holdout scoring, and not answer generation.",
        "",
        "## Status",
        "",
        f"- status: `{status}`",
        f"- error: `{error}`",
        "",
        "## Counts",
        "",
        f"- request rows selected: {len(rows)}",
        f"- responses written: {len(responses)}",
        "",
        "## Outputs",
        "",
        f"- `{rel(responses_path)}`",
        f"- `{rel(summary_path)}`",
    ]
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", default="gemma4_e2b_it")
    parser.add_argument("--requests", type=Path, default=DEFAULT_REQUESTS)
    parser.add_argument("--max-cases", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--dtype", choices=("auto", "bfloat16", "float16", "float32"), default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--run-name", default="tiny_dev_smoke")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_outputs(
        model_key=args.model_key,
        requests_path=args.requests,
        max_cases=args.max_cases,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
        device_map=args.device_map,
        require_cuda=args.require_cuda,
        run_name=args.run_name,
        preflight_only=args.preflight_only,
    )
    print(f"wrote {summary['outputs']['summary_md']}")
    print(
        f"status={summary['status']} model={summary['model']['key']} "
        f"responses={summary['counts']['responses_written']} "
        f"error={summary['error']}"
    )
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
