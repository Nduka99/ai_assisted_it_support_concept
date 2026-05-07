"""Prepare a dry-run prompt/schema contract for local LLM classification.

This script writes dev-only request artifacts for a future local Gemma/Qwen JSON
classifier pass. It does not load a model, call an LLM, train, read holdout, or
generate troubleshooting answers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from it_support.classification.embedding_classifier import select_balanced_cases  # noqa: E402
from it_support.classification.llm_json_classifier import (  # noqa: E402
    SCHEMA_NAME,
    build_classifier_request,
    eval_key,
    llm_classifier_response_schema,
)
from it_support.config import DATA_DIR, PROJECT_ROOT  # noqa: E402


SPLITS_DIR = DATA_DIR / "eval" / "candidate_loader_and_eval_splits"
OUT = DATA_DIR / "eval" / "llm_json_classifier_dry_run"
ROUTING_DEV_SPLIT = "routing_dev_eval"


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


def assert_routing_dev(rows: list[dict[str, Any]], *, path: Path) -> None:
    splits = sorted({str(row.get("split")) for row in rows})
    if splits != [ROUTING_DEV_SPLIT]:
        raise ValueError(f"{path} must contain only {ROUTING_DEV_SPLIT!r}; got {splits}")


def build_outputs(*, max_cases: int, include_tags: bool) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    routing_cases_path = SPLITS_DIR / "routing_eval_dev.jsonl"
    all_cases = read_jsonl(routing_cases_path)
    assert_routing_dev(all_cases, path=routing_cases_path)

    cases = select_balanced_cases(all_cases, max_cases=max_cases)
    profile = "llm_json_classifier_metadata_dry_run" if include_tags else (
        "llm_json_classifier_text_dry_run"
    )
    scope = "dev_sample"
    schema = llm_classifier_response_schema()
    requests = [
        build_classifier_request(case, include_tags=include_tags, profile=profile)
        for case in cases
    ]
    eval_keys = [eval_key(case, profile=profile) for case in cases]

    suffix = f"{scope}_{'metadata' if include_tags else 'text'}"
    outputs = {
        "schema": OUT / f"llm_json_classifier_schema_{SCHEMA_NAME}.json",
        "requests": OUT / f"llm_json_classifier_requests_{suffix}.jsonl",
        "eval_keys": OUT / f"llm_json_classifier_eval_keys_{suffix}.jsonl",
        "summary_json": OUT / f"llm_json_classifier_dry_run_summary_{suffix}.json",
        "summary_md": OUT / f"llm_json_classifier_dry_run_summary_{suffix}.md",
    }
    write_json(outputs["schema"], schema)
    write_jsonl(outputs["requests"], requests)
    write_jsonl(outputs["eval_keys"], eval_keys)

    policy = [
        "Consumes routing dev fixtures only.",
        "Writes prompt/request and evaluation-key artifacts for future local LLM use.",
        "Keeps expected labels out of model request rows; labels are stored separately.",
        "Does not read holdout, train, load a model, call an LLM, or generate answers.",
    ]
    summary = {
        "stage": "llm_json_classifier_dry_run",
        "scope": scope,
        "profile": profile,
        "schema_name": SCHEMA_NAME,
        "policy": policy,
        "settings": {
            "include_tags": include_tags,
            "max_cases": max_cases,
        },
        "counts": {
            "available_routing_dev_cases": len(all_cases),
            "request_cases": len(requests),
            "eval_key_cases": len(eval_keys),
        },
        "inputs": {
            "routing_cases": rel(routing_cases_path),
        },
        "outputs": {key: rel(path) for key, path in outputs.items()},
        "future_heavy_commands_for_user": [
            "Run a local Gemma/Qwen JSON classifier against the requests artifact only "
            "after explicitly approving LLM inference.",
            "Evaluate parsed responses against the eval-keys artifact on dev only before "
            "any holdout scoring.",
        ],
    }
    write_json(outputs["summary_json"], summary)

    lines = [
        "# LLM JSON Classifier Dry Run",
        "",
        "This stage prepares a strict JSON classifier contract for future local LLM "
        "classification. It does not load or call a model.",
        "",
        "## Policy",
        "",
        *[f"- {item}" for item in policy],
        "",
        "## Counts",
        "",
        md_table([summary["counts"]], list(summary["counts"])),
        "",
        "## Settings",
        "",
        md_table([summary["settings"]], list(summary["settings"])),
        "",
        "## Outputs",
        "",
        *[f"- `{path}`" for path in summary["outputs"].values()],
        "",
        "## Next Heavy Step",
        "",
        "The next heavy step is local LLM inference against the request artifact. "
        "That should remain explicit and dev-only until the JSON parser/evaluator is ready.",
    ]
    outputs["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-cases", type=int, default=24)
    parser.add_argument("--include-tags", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_outputs(max_cases=args.max_cases, include_tags=args.include_tags)
    print(f"wrote {summary['outputs']['summary_md']}")
    print(
        f"profile={summary['profile']} request_cases={summary['counts']['request_cases']} "
        f"schema={summary['schema_name']}"
    )
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
