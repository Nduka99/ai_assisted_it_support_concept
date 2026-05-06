"""Download the IT-SUPPORT model ladder from Hugging Face.

This script is intentionally outside the notebooks. Downloads are big, resumable, and
environment-oriented; notebooks should start at data acquisition and experiment output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "model_downloads.json"


@dataclass(frozen=True)
class DownloadSpec:
    key: str
    phase: str
    priority: int
    repo_id: str
    local_dir: Path
    backend: str
    role: str
    license: str
    estimated_size: str
    reason: str
    gated: bool
    allow_patterns: list[str] | None
    ignore_patterns: list[str] | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DownloadSpec":
        return cls(
            key=raw["key"],
            phase=raw["phase"],
            priority=int(raw["priority"]),
            repo_id=raw["repo_id"],
            local_dir=(PROJECT_ROOT / raw["local_dir"]).resolve(),
            backend=raw["backend"],
            role=raw["role"],
            license=raw["license"],
            estimated_size=raw.get("estimated_size", "unknown"),
            reason=raw["reason"],
            gated=bool(raw.get("gated", False)),
            allow_patterns=raw.get("allow_patterns"),
            ignore_patterns=raw.get("ignore_patterns"),
        )


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_manifest(path: Path) -> list[DownloadSpec]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    specs = [DownloadSpec.from_dict(item) for item in raw["models"]]
    return sorted(specs, key=lambda item: item.priority)


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def size_label(size_bytes: int) -> str:
    if size_bytes <= 0:
        return "missing"
    return f"{size_bytes / (1024**3):.2f} GB"


def selected_specs(
    specs: list[DownloadSpec],
    *,
    phase: str | None,
    keys: list[str],
) -> list[DownloadSpec]:
    if keys:
        wanted = set(keys)
        found = [spec for spec in specs if spec.key in wanted]
        missing = sorted(wanted - {spec.key for spec in found})
        if missing:
            raise SystemExit(f"Unknown model key(s): {', '.join(missing)}")
        return found
    if phase and phase != "all":
        return [spec for spec in specs if spec.phase == phase]
    return specs


def print_specs(specs: list[DownloadSpec]) -> None:
    for spec in specs:
        marker = "present" if spec.local_dir.exists() else "missing"
        print(
            f"{spec.priority}. {spec.key} [{spec.phase}] {marker} "
            f"({size_label(dir_size(spec.local_dir))}, expected {spec.estimated_size})"
        )
        print(f"   repo: {spec.repo_id}")
        print(f"   path: {spec.local_dir.relative_to(PROJECT_ROOT)}")
        print(f"   role: {spec.role}")
        print(f"   why:  {spec.reason}")
        if spec.gated:
            print("   note: gated/terms model; set HF_TOKEN after accepting access on Hugging Face")


def download(spec: DownloadSpec, *, token: str | None, force: bool, max_workers: int) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install requirements first:\n"
            r"  .\.it_support\Scripts\python.exe -m pip install -r requirements.txt"
        ) from exc

    if spec.local_dir.exists() and dir_size(spec.local_dir) > 0 and not force:
        print(f"SKIP {spec.key}: already present at {spec.local_dir}")
        return

    if spec.gated and not token:
        raise SystemExit(
            f"{spec.key} is marked as gated. Accept access for {spec.repo_id} on Hugging Face, "
            "then put HF_TOKEN=... in .env or set it in the shell."
        )

    spec.local_dir.parent.mkdir(parents=True, exist_ok=True)
    print(f"DOWNLOAD {spec.key}: {spec.repo_id} -> {spec.local_dir}")
    snapshot_download(
        repo_id=spec.repo_id,
        local_dir=str(spec.local_dir),
        allow_patterns=spec.allow_patterns,
        ignore_patterns=spec.ignore_patterns,
        token=token,
        max_workers=max_workers,
    )
    print(f"DONE {spec.key}: {size_label(dir_size(spec.local_dir))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--phase", choices=["starter", "classifier", "stretch", "all"], default="starter")
    parser.add_argument("--model", action="append", default=[], help="Specific model key to download.")
    parser.add_argument("--list", action="store_true", help="Print matching models and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print matching models without downloading.")
    parser.add_argument("--force", action="store_true", help="Re-download even if the local path exists.")
    parser.add_argument("--max-workers", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    specs = selected_specs(load_manifest(args.manifest), phase=args.phase, keys=args.model)

    if args.list or args.dry_run:
        print_specs(specs)
        return 0

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    for spec in specs:
        download(spec, token=token, force=args.force, max_workers=args.max_workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
