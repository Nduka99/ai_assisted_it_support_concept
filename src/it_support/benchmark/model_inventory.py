"""Print a lightweight inventory of known local models."""

from __future__ import annotations

from it_support.config import LOCAL_MODELS, TARGET_DOWNLOADS


def _size_label(path) -> str:
    if not path.exists():
        return "missing"
    if path.is_dir():
        return "directory"
    size_gb = path.stat().st_size / (1024**3)
    return f"{size_gb:.2f} GB"


def main() -> None:
    print("Present / expected local models")
    for model in LOCAL_MODELS.values():
        marker = "OK" if model.exists else "MISSING"
        print(f"- {marker} {model.key}: {model.role} ({_size_label(model.path)})")

    print("\nTarget downloads not currently registered as present")
    for key, reason in TARGET_DOWNLOADS.items():
        print(f"- {key}: {reason}")


if __name__ == "__main__":
    main()
