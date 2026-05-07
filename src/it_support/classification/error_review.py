"""Error-review helpers for dev-only classification ladder artifacts."""

from __future__ import annotations

from collections import Counter
from typing import Any


def pipe_values(value: Any) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    return [part for part in text.split("|") if part]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def rank_bucket(value: Any, *, cutoff: int = 3) -> str:
    text = str(value or "").strip()
    if not text:
        return "not_predicted"
    try:
        rank = int(float(text))
    except ValueError:
        return "not_predicted"
    if rank <= cutoff:
        return f"rank_{rank}"
    return f"rank_gt_{cutoff}"


def primary_error_count_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (
            row.get("profile", ""),
            row.get("expected_primary_domain", ""),
            row.get("predicted_primary_domain", ""),
        )
        for row in rows
        if not parse_bool(row.get("primary_correct"))
    )
    return [
        {
            "profile": profile,
            "expected_primary_domain": expected,
            "predicted_primary_domain": predicted,
            "records": count,
        }
        for (profile, expected, predicted), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def primary_error_examples(
    rows: list[dict[str, Any]],
    *,
    limit_per_pair: int = 5,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if parse_bool(row.get("primary_correct")):
            continue
        key = (
            str(row.get("profile", "")),
            str(row.get("expected_primary_domain", "")),
            str(row.get("predicted_primary_domain", "")),
        )
        grouped.setdefault(key, []).append(row)

    examples = []
    for key, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        profile, expected, predicted = key
        for row in sorted(group, key=lambda item: str(item.get("case_id", "")))[:limit_per_pair]:
            examples.append(
                {
                    "profile": profile,
                    "expected_primary_domain": expected,
                    "predicted_primary_domain": predicted,
                    "case_id": row.get("case_id", ""),
                    "title": row.get("title", ""),
                    "expected_domains": row.get("expected_domains", ""),
                    "predicted_domains": row.get("predicted_domains", ""),
                    "first_primary_rank": row.get("first_primary_rank", ""),
                    "source_url": row.get("source_url", ""),
                }
            )
    return examples


def multilabel_gap_count_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter()
    for row in rows:
        expected = set(pipe_values(row.get("expected_domains")))
        predicted = set(pipe_values(row.get("predicted_domains")))
        profile = row.get("profile", "")
        for domain in expected - predicted:
            counts[(profile, "missing_expected", domain)] += 1
        for domain in predicted - expected:
            counts[(profile, "extra_predicted", domain)] += 1
    return [
        {
            "profile": profile,
            "gap_type": gap_type,
            "domain": domain,
            "records": count,
        }
        for (profile, gap_type, domain), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def multilabel_gap_examples(
    rows: list[dict[str, Any]],
    *,
    limit_per_gap: int = 8,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        expected = set(pipe_values(row.get("expected_domains")))
        predicted = set(pipe_values(row.get("predicted_domains")))
        profile = str(row.get("profile", ""))
        for domain in expected - predicted:
            grouped.setdefault((profile, "missing_expected", domain), []).append(row)
        for domain in predicted - expected:
            grouped.setdefault((profile, "extra_predicted", domain), []).append(row)

    examples = []
    for key, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        profile, gap_type, domain = key
        for row in sorted(group, key=lambda item: str(item.get("case_id", "")))[:limit_per_gap]:
            examples.append(
                {
                    "profile": profile,
                    "gap_type": gap_type,
                    "domain": domain,
                    "case_id": row.get("case_id", ""),
                    "expected_primary_domain": row.get("expected_primary_domain", ""),
                    "predicted_primary_domain": row.get("predicted_primary_domain", ""),
                    "title": row.get("title", ""),
                    "expected_domains": row.get("expected_domains", ""),
                    "predicted_domains": row.get("predicted_domains", ""),
                    "source_url": row.get("source_url", ""),
                }
            )
    return examples


def ranked_primary_bucket_rows(
    rows: list[dict[str, Any]],
    *,
    cutoff: int = 3,
) -> list[dict[str, Any]]:
    counts = Counter(
        (
            row.get("profile", ""),
            row.get("expected_primary_domain", ""),
            rank_bucket(row.get("first_primary_rank"), cutoff=cutoff),
        )
        for row in rows
    )
    return [
        {
            "profile": profile,
            "expected_primary_domain": expected,
            "rank_bucket": bucket,
            "records": count,
        }
        for (profile, expected, bucket), count in sorted(
            counts.items(), key=lambda item: (item[0][0], item[0][1], item[0][2])
        )
    ]


def safety_signal_gap_count_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter()
    for row in rows:
        profile = row.get("profile", "")
        for signal in pipe_values(row.get("missed_safety_signals")):
            counts[(profile, "missed_expected", signal)] += 1
        for signal in pipe_values(row.get("extra_safety_signals")):
            counts[(profile, "extra_predicted", signal)] += 1
    return [
        {
            "profile": profile,
            "gap_type": gap_type,
            "safety_signal": signal,
            "records": count,
        }
        for (profile, gap_type, signal), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]


def safety_signal_gap_examples(
    rows: list[dict[str, Any]],
    *,
    limit_per_gap: int = 8,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        profile = str(row.get("profile", ""))
        for signal in pipe_values(row.get("missed_safety_signals")):
            grouped.setdefault((profile, "missed_expected", signal), []).append(row)
        for signal in pipe_values(row.get("extra_safety_signals")):
            grouped.setdefault((profile, "extra_predicted", signal), []).append(row)

    examples = []
    for key, group in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        profile, gap_type, signal = key
        for row in sorted(group, key=lambda item: str(item.get("case_id", "")))[:limit_per_gap]:
            examples.append(
                {
                    "profile": profile,
                    "gap_type": gap_type,
                    "safety_signal": signal,
                    "case_id": row.get("case_id", ""),
                    "expected_behavior": row.get("expected_behavior", ""),
                    "predicted_behavior": row.get("predicted_behavior", ""),
                    "behavior_correct": row.get("behavior_correct", ""),
                    "title": row.get("title", ""),
                    "expected_safety_signals": row.get("expected_safety_signals", ""),
                    "predicted_safety_signals": row.get("predicted_safety_signals", ""),
                    "source_url": row.get("source_url", ""),
                }
            )
    return examples
