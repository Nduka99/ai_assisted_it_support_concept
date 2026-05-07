"""Review dev-only rule-baseline safety false negatives.

This is an audit helper, not a model or rule tuner. It reads the question-only
dev safety fixtures plus the generated false-negative table and separates likely
rule gaps from conservative fixture construction. Holdout files are not read.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from it_support.config import DATA_DIR, PROJECT_ROOT


EVAL_IN = DATA_DIR / "eval" / "candidate_loader_and_eval_splits"
BASELINE = DATA_DIR / "eval" / "rule_based_triage_safety_baseline"
PROFILE = "metadata_assisted"

SECURITY_STRONG_TERMS = [
    "backdoor",
    "badusb",
    "breach",
    "cache poisoning",
    "compromise",
    "compromised",
    "cve",
    "exploit",
    "hacked",
    "identity theft",
    "keylogger",
    "malicious",
    "malware",
    "phishing",
    "ransomware",
    "scam",
    "spoofing",
    "spyware",
    "superfish",
    "suspicious",
    "unauthorized",
    "virus",
    "vulnerability",
]
SECURITY_BROAD_TERMS = [
    "acl",
    "admin rights",
    "certificate",
    "firewall",
    "open port",
    "password",
    "port",
    "private key",
    "security",
    "token",
]
FIRMWARE_STRONG_TERMS = [
    "bios",
    "bios update",
    "bricked",
    "cmos",
    "efi",
    "firmware",
    "firmware update",
    "flash bios",
    "flashing bios",
    "nvram",
    "secure boot",
    "uefi",
]
FIRMWARE_BROAD_TERMS = [
    "boot",
    "bootloader",
    "driver",
    "drivers",
    "grub",
    "grub2",
    "kernel",
    "nomodeset",
    "nouveau",
    "nvidia",
]


def rel(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def normalize(value: str) -> str:
    value = value.lower()
    value = value.replace("-", " ")
    value = re.sub(r"[^a-z0-9+#./_]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def has_term(text: str, term: str) -> bool:
    needle = normalize(term)
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text) is not None


def term_hits(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if has_term(text, term)]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or sorted({field for row in rows for field in row})
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


def question_only_text(case: dict[str, Any]) -> str:
    parts = [
        case.get("title", ""),
        case.get("question_text", ""),
        " ".join(case.get("question_tags") or []),
        " ".join(case.get("query_tags") or []),
    ]
    return normalize(" ".join(str(part) for part in parts if part))


def classify_review(
    miss: dict[str, str],
    case: dict[str, Any],
) -> dict[str, str]:
    text = question_only_text(case)
    expected = miss["expected_behavior"]
    predicted = miss["predicted_behavior"]
    security_strong = term_hits(text, SECURITY_STRONG_TERMS)
    security_broad = term_hits(text, SECURITY_BROAD_TERMS)
    firmware_strong = term_hits(text, FIRMWARE_STRONG_TERMS)
    firmware_broad = term_hits(text, FIRMWARE_BROAD_TERMS)
    predicted_other_safety = predicted in {
        "credential_or_access_review",
        "data_preservation_review",
        "structured_firmware_escalation",
        "security_triage_or_escalation_after_filter",
    }

    if expected == "security_triage_or_escalation_after_filter":
        if predicted_other_safety and predicted != "route_to_expected_domains":
            review_bucket = "priority_or_multisafety_mismatch"
            recommendation = "Review behavior precedence before adding rules."
        elif security_strong:
            review_bucket = "candidate_rule_gap_question_visible"
            recommendation = "Add or weight visible security cue in rule baseline."
        elif security_broad:
            review_bucket = "conservative_security_fixture"
            recommendation = "Keep as review/eval-only or relabel after manual inspection."
        else:
            review_bucket = "likely_gate_context_mismatch"
            recommendation = (
                "Do not tune rules from this row until source-gate trigger is inspected."
            )
    elif expected == "structured_firmware_escalation":
        if firmware_strong:
            review_bucket = "candidate_rule_gap_question_visible"
            recommendation = "Add or weight visible firmware cue in rule baseline."
        elif firmware_broad:
            review_bucket = "conservative_firmware_fixture"
            recommendation = "Likely OS/driver/bootloader support, not vendor firmware escalation."
        else:
            review_bucket = "likely_gate_context_mismatch"
            recommendation = (
                "Do not tune rules from this row until source-gate trigger is inspected."
            )
    else:
        review_bucket = "unexpected_expected_behavior"
        recommendation = "Inspect fixture generation before tuning."

    return {
        "review_bucket": review_bucket,
        "recommendation": recommendation,
        "visible_security_strong_terms": ";".join(security_strong),
        "visible_security_broad_terms": ";".join(security_broad),
        "visible_firmware_strong_terms": ";".join(firmware_strong),
        "visible_firmware_broad_terms": ";".join(firmware_broad),
        "safety_flags": json.dumps(case.get("safety_flags", {}), sort_keys=True),
    }


def build_review() -> dict[str, Any]:
    cases = {row["case_id"]: row for row in read_jsonl(EVAL_IN / "safety_eval_dev.jsonl")}
    misses = [
        row
        for row in read_csv(BASELINE / "safety_false_negatives_dev_only.csv")
        if row["profile"] == PROFILE
    ]
    review_rows: list[dict[str, Any]] = []
    for miss in misses:
        case = cases[miss["case_id"]]
        review = classify_review(miss, case)
        review_rows.append(
            {
                **miss,
                **review,
                "question_text_preview": re.sub(
                    r"\s+",
                    " ",
                    case.get("question_text", ""),
                )[:360],
                "gate_decision": case.get("gate_decision"),
                "blocked_reason": case.get("blocked_reason"),
            }
        )

    bucket_counts = Counter(row["review_bucket"] for row in review_rows)
    expected_bucket_counts = Counter(
        (row["expected_behavior"], row["review_bucket"]) for row in review_rows
    )
    predicted_bucket_counts = Counter(
        (row["predicted_behavior"], row["review_bucket"]) for row in review_rows
    )
    tag_bucket_counts = Counter()
    for row in review_rows:
        tags = [tag for tag in row.get("question_tags", "").split(";") if tag]
        for tag in tags or ["<no_question_tag>"]:
            tag_bucket_counts[(row["review_bucket"], tag)] += 1

    bucket_rows = [
        {"review_bucket": bucket, "records": count}
        for bucket, count in sorted(bucket_counts.items())
    ]
    expected_rows = [
        {
            "expected_behavior": expected,
            "review_bucket": bucket,
            "records": count,
        }
        for (expected, bucket), count in sorted(expected_bucket_counts.items())
    ]
    predicted_rows = [
        {
            "predicted_behavior": predicted,
            "review_bucket": bucket,
            "records": count,
        }
        for (predicted, bucket), count in sorted(predicted_bucket_counts.items())
    ]
    tag_rows = [
        {"review_bucket": bucket, "question_tag": tag, "records": count}
        for (bucket, tag), count in sorted(
            tag_bucket_counts.items(),
            key=lambda item: (item[0][0], -item[1], item[0][1]),
        )
    ]

    outputs = {
        "review_rows": BASELINE / "safety_false_negative_review_dev_only.csv",
        "review_bucket_counts": (
            BASELINE / "safety_false_negative_review_bucket_counts_dev_only.csv"
        ),
        "review_expected_bucket_counts": (
            BASELINE / "safety_false_negative_review_expected_bucket_counts_dev_only.csv"
        ),
        "review_predicted_bucket_counts": (
            BASELINE / "safety_false_negative_review_predicted_bucket_counts_dev_only.csv"
        ),
        "review_tag_bucket_counts": (
            BASELINE / "safety_false_negative_review_tag_bucket_counts_dev_only.csv"
        ),
        "review_summary": BASELINE / "safety_false_negative_review_summary_dev_only.md",
    }
    write_csv(
        outputs["review_rows"],
        review_rows,
        fields=[
            "blocked_reason",
            "case_id",
            "expected_behavior",
            "expected_primary_domain",
            "gate_decision",
            "predicted_behavior",
            "predicted_domains",
            "predicted_primary_domain",
            "profile",
            "query_tags",
            "question_tags",
            "question_text_preview",
            "recommendation",
            "review_bucket",
            "safety_flags",
            "source_url",
            "title",
            "visible_firmware_broad_terms",
            "visible_firmware_strong_terms",
            "visible_security_broad_terms",
            "visible_security_strong_terms",
        ],
    )
    write_csv(
        outputs["review_bucket_counts"],
        bucket_rows,
        fields=["review_bucket", "records"],
    )
    write_csv(
        outputs["review_expected_bucket_counts"],
        expected_rows,
        fields=["expected_behavior", "review_bucket", "records"],
    )
    write_csv(
        outputs["review_predicted_bucket_counts"],
        predicted_rows,
        fields=["predicted_behavior", "review_bucket", "records"],
    )
    write_csv(
        outputs["review_tag_bucket_counts"],
        tag_rows,
        fields=["review_bucket", "question_tag", "records"],
    )

    candidate_rule_gaps = [
        row
        for row in review_rows
        if row["review_bucket"] == "candidate_rule_gap_question_visible"
    ][:12]
    conservative_examples = [
        row
        for row in review_rows
        if row["review_bucket"] in {
            "conservative_firmware_fixture",
            "conservative_security_fixture",
            "likely_gate_context_mismatch",
        }
    ][:12]
    priority_examples = [
        row
        for row in review_rows
        if row["review_bucket"] == "priority_or_multisafety_mismatch"
    ][:12]

    lines = [
        "# Safety False-Negative Review",
        "",
        "Scope: `dev_only`, profile: `metadata_assisted`.",
        "",
        "This review does not change rules, use holdout, train a model, build an index, "
        "or enable answer generation.",
        "",
        "## Review Buckets",
        "",
        md_table(bucket_rows, ["review_bucket", "records"]),
        "",
        *(
            ["No metadata-assisted safety false negatives remain in the dev-only baseline.", ""]
            if not review_rows
            else []
        ),
        "## By Expected Behavior",
        "",
        md_table(expected_rows, ["expected_behavior", "review_bucket", "records"]),
        "",
        "## Interpretation",
        "",
        "- `candidate_rule_gap_question_visible`: the question-only fixture visibly contains "
        "a strong security or firmware cue; these are legitimate candidates for rule changes.",
        "- `conservative_security_fixture`: the source gate flagged broad security posture, "
        "but question-only text may not require security escalation.",
        "- `conservative_firmware_fixture`: driver, GRUB, bootloader, or kernel language "
        "was treated as firmware escalation; many are probably OS/driver support cases.",
        "- `likely_gate_context_mismatch`: the visible question-only case does not show "
        "the safety cue strongly enough; inspect the gate context before tuning.",
        "- `priority_or_multisafety_mismatch`: the baseline did flag a safety behavior, "
        "just not the expected one; this is a precedence/calibration question.",
        "",
        "## Candidate Rule-Gap Examples",
        "",
        md_table(
            candidate_rule_gaps,
            [
                "case_id",
                "expected_behavior",
                "predicted_behavior",
                "visible_security_strong_terms",
                "visible_firmware_strong_terms",
                "title",
            ],
        ),
        "",
        "## Conservative Fixture Examples",
        "",
        md_table(
            conservative_examples,
            [
                "case_id",
                "expected_behavior",
                "review_bucket",
                "visible_security_broad_terms",
                "visible_firmware_broad_terms",
                "title",
            ],
        ),
        "",
        "## Priority/Multi-Safety Examples",
        "",
        md_table(
            priority_examples,
            [
                "case_id",
                "expected_behavior",
                "predicted_behavior",
                "visible_security_strong_terms",
                "visible_firmware_strong_terms",
                "title",
            ],
        ),
        "",
        "## Outputs",
        "",
        *[f"- `{rel(path)}`" for path in outputs.values()],
    ]
    outputs["review_summary"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "stage": "rule_based_safety_false_negative_review",
        "scope": "dev_only",
        "profile": PROFILE,
        "records": len(review_rows),
        "bucket_counts": bucket_rows,
        "outputs": {key: rel(path) for key, path in outputs.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="store_true", help="Print JSON summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_review()
    print(f"wrote {summary['outputs']['review_summary']}")
    print(f"records={summary['records']}")
    for row in summary["bucket_counts"]:
        print(f"{row['review_bucket']}={row['records']}")
    if args.summary:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
