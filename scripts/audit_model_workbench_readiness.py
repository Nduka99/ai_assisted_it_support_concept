"""Audit local downloaded model readiness for the next model-backed work.

This is an inventory/readiness script. It does not load large models, train,
score holdout, call an LLM, or generate answers.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from it_support.config import DATA_DIR, LOCAL_MODELS, PROJECT_ROOT


OUT = DATA_DIR / "eval" / "model_workbench_readiness"
CORE_MODEL_KEYS = [
    "bge_small_en_v15",
    "qwen3_embedding_06b",
    "gemma4_e2b_it",
    "qwen35_4b",
    "ministral3_3b_instruct_q4km",
    "gemma4_e4b_it_q4km",
]


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


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


def path_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def has_module(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def torch_runtime_row() -> dict[str, Any]:
    row: dict[str, Any] = {
        "package": "torch",
        "available": False,
        "version": "",
        "cuda_available": False,
        "cuda_version": "",
        "device_count": 0,
        "device_name": "",
    }
    try:
        import torch
    except ImportError:
        return row
    row["available"] = True
    row["version"] = str(torch.__version__)
    row["cuda_available"] = bool(torch.cuda.is_available())
    row["cuda_version"] = str(torch.version.cuda or "")
    row["device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    row["device_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    return row


def runtime_rows() -> list[dict[str, Any]]:
    rows = []
    for package, module_name in [
        ("sentence-transformers", "sentence_transformers"),
        ("transformers", "transformers"),
        ("faiss", "faiss"),
        ("llama-cpp-python", "llama_cpp"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
    ]:
        rows.append(
            {
                "package": package,
                "module": module_name,
                "available": has_module(module_name),
            }
        )
    return rows


def model_rows(model_keys: list[str]) -> list[dict[str, Any]]:
    rows = []
    for key in model_keys:
        model = LOCAL_MODELS[key]
        size = path_size_bytes(model.path)
        rows.append(
            {
                "key": key,
                "role": model.role,
                "backend": model.backend,
                "registry_status": model.status,
                "exists": model.path.exists(),
                "path": rel(model.path),
                "path_type": "dir" if model.path.is_dir() else "file" if model.path.is_file() else "missing",
                "size_gb": round(size / (1024**3), 3),
                "notes": model.notes,
            }
        )
    return rows


def build_outputs(*, all_models: bool = False) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    keys = sorted(LOCAL_MODELS) if all_models else CORE_MODEL_KEYS
    models = model_rows(keys)
    runtimes = runtime_rows()
    torch_row = torch_runtime_row()

    outputs = {
        "model_inventory": OUT / "model_workbench_inventory.csv",
        "runtime_inventory": OUT / "model_workbench_runtime.csv",
        "summary_json": OUT / "model_workbench_readiness_summary.json",
        "summary_md": OUT / "model_workbench_readiness_summary.md",
    }
    write_csv(
        outputs["model_inventory"],
        models,
        [
            "key",
            "role",
            "backend",
            "registry_status",
            "exists",
            "path_type",
            "size_gb",
            "path",
            "notes",
        ],
    )
    write_csv(outputs["runtime_inventory"], runtimes)

    ready_sentence_transformers = [
        row["key"]
        for row in models
        if row["backend"] == "sentence-transformers"
        and row["exists"]
        and any(runtime["module"] == "sentence_transformers" and runtime["available"] for runtime in runtimes)
    ]
    local_llm_assets = [
        row["key"]
        for row in models
        if row["backend"] in {"transformers", "llama.cpp"} and row["exists"]
    ]
    next_commands = [
        ".\\.it_support\\Scripts\\python.exe scripts\\evaluate_embedding_classifier_baseline.py --model-key bge_small_en_v15 --summary",
        ".\\.it_support\\Scripts\\python.exe scripts\\evaluate_embedding_classifier_baseline.py --model-key bge_small_en_v15 --include-tags --summary",
        ".\\.it_support\\Scripts\\python.exe scripts\\evaluate_embedding_classifier_baseline.py --model-key qwen3_embedding_06b --device auto --require-cuda --batch-size 4 --summary",
        ".\\.it_support\\Scripts\\python.exe scripts\\evaluate_embedding_classifier_baseline.py --model-key bge_small_en_v15 --full-dev --summary",
    ]
    summary = {
        "stage": "model_workbench_readiness",
        "scope": "inventory_only",
        "policy": [
            "Audits local model files and runtime packages.",
            "Does not load large models.",
            "Does not read holdout, train, call an LLM, or generate answers.",
            "Embedding classifier scripts default to bounded dev samples; full dev requires --full-dev.",
        ],
        "counts": {
            "models_audited": len(models),
            "models_present": sum(1 for row in models if row["exists"]),
            "ready_sentence_transformer_models": len(ready_sentence_transformers),
            "local_llm_assets_present": len(local_llm_assets),
        },
        "torch": torch_row,
        "ready_sentence_transformer_models": ready_sentence_transformers,
        "local_llm_assets_present": local_llm_assets,
        "next_commands": next_commands,
        "outputs": {key: rel(path) for key, path in outputs.items()},
    }
    write_json(outputs["summary_json"], summary)

    lines = [
        "# Model Workbench Readiness",
        "",
        "This audit checks downloaded model assets and runtime packages. It does not "
        "load large models, train, score holdout, call an LLM, or generate answers.",
        "",
        "## Counts",
        "",
        md_table([summary["counts"]], list(summary["counts"])),
        "",
        "## Torch Runtime",
        "",
        md_table([torch_row], list(torch_row)),
        "",
        "## Model Inventory",
        "",
        md_table(
            models,
            ["key", "role", "backend", "registry_status", "exists", "path_type", "size_gb"],
        ),
        "",
        "## Starter Commands",
        "",
        *[f"```powershell\n{command}\n```" for command in next_commands],
        "",
        "## Outputs",
        "",
        *[f"- `{rel(path)}`" for path in outputs.values()],
    ]
    outputs["summary_md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-models", action="store_true")
    parser.add_argument("--summary", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_outputs(all_models=args.all_models)
    print(f"wrote {summary['outputs']['summary_md']}")
    print(f"models_present={summary['counts']['models_present']}")
    print(f"ready_sentence_transformers={summary['ready_sentence_transformer_models']}")
    print(f"cuda_available={summary['torch']['cuda_available']}")
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
