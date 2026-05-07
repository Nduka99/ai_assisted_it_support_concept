"""Embedding nearest-neighbor helpers for dev-only classifier baselines."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def compact_case_text(case: dict[str, Any], *, include_tags: bool) -> str:
    parts = [
        str(case.get("title") or ""),
        str(case.get("question_text") or ""),
    ]
    if include_tags:
        tags = list(case.get("question_tags") or []) + list(case.get("query_tags") or [])
        parts.append(" ".join(str(tag) for tag in tags))
    return " ".join(" ".join(parts).split())


def select_balanced_cases(
    cases: list[dict[str, Any]],
    *,
    max_cases: int | None,
    group_field: str = "expected_primary_domain",
) -> list[dict[str, Any]]:
    sorted_cases = sorted(cases, key=lambda row: str(row["case_id"]))
    if not max_cases or len(sorted_cases) <= max_cases:
        return sorted_cases

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in sorted_cases:
        grouped[str(case.get(group_field) or "unknown")].append(case)

    selected = []
    per_group = max(1, max_cases // max(1, len(grouped)))
    for group in sorted(grouped):
        selected.extend(grouped[group][:per_group])
    if len(selected) >= max_cases:
        return selected[:max_cases]

    selected_ids = {row["case_id"] for row in selected}
    remainder = [row for row in sorted_cases if row["case_id"] not in selected_ids]
    selected.extend(remainder[: max_cases - len(selected)])
    return selected[:max_cases]


def normalized_positive_weight(score: float, *, floor: float = 0.0) -> float:
    return max(float(score), floor)


def predict_domains_from_neighbors(
    *,
    neighbors: list[dict[str, Any]],
    max_domains: int,
    label_threshold_ratio: float,
) -> list[dict[str, Any]]:
    primary_scores: dict[str, float] = defaultdict(float)
    label_scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[str]] = defaultdict(list)

    for neighbor in neighbors:
        weight = normalized_positive_weight(float(neighbor.get("score", 0.0)))
        if weight <= 0:
            continue
        case = neighbor["case"]
        primary = str(case.get("expected_primary_domain") or "")
        if primary:
            primary_scores[primary] += weight
            evidence[primary].append(str(case.get("case_id") or ""))
        for label in case.get("expected_domains") or []:
            label = str(label)
            label_scores[label] += weight
            evidence[label].append(str(case.get("case_id") or ""))

    if not label_scores:
        return []

    primary_label = max(
        primary_scores or label_scores,
        key=lambda label: (primary_scores.get(label, 0.0), label_scores.get(label, 0.0), label),
    )
    top_score = max(label_scores.values())
    threshold = top_score * label_threshold_ratio
    ranked_labels = [
        label
        for label, score in sorted(label_scores.items(), key=lambda item: (-item[1], item[0]))
        if score >= threshold
    ]
    if primary_label not in ranked_labels:
        ranked_labels.insert(0, primary_label)
    ranked_labels = sorted(
        ranked_labels,
        key=lambda label: (
            0 if label == primary_label else 1,
            -label_scores.get(label, 0.0),
            label,
        ),
    )[:max_domains]

    return [
        {
            "label": label,
            "score": round(float(label_scores[label]), 6),
            "confidence": round(float(label_scores[label] / top_score), 4) if top_score else 0.0,
            "evidence": sorted(set(evidence[label]))[:5],
        }
        for label in ranked_labels
    ]


def top_neighbors(
    *,
    query_case: dict[str, Any],
    reference_cases: list[dict[str, Any]],
    scores: np.ndarray,
    top_k: int,
) -> list[dict[str, Any]]:
    order = np.argsort(-scores)
    neighbors = []
    for index in order:
        case = reference_cases[int(index)]
        if str(case.get("case_id")) == str(query_case.get("case_id")):
            continue
        neighbors.append(
            {
                "rank": len(neighbors) + 1,
                "case_id": case.get("case_id"),
                "score": round(float(scores[int(index)]), 6),
                "case": case,
            }
        )
        if len(neighbors) >= top_k:
            break
    return neighbors


def prediction_record_from_neighbors(
    *,
    query_case: dict[str, Any],
    neighbors: list[dict[str, Any]],
    profile: str,
    max_domains: int,
    label_threshold_ratio: float,
) -> dict[str, Any]:
    predicted_domains = predict_domains_from_neighbors(
        neighbors=neighbors,
        max_domains=max_domains,
        label_threshold_ratio=label_threshold_ratio,
    )
    primary = predicted_domains[0]["label"] if predicted_domains else None
    return {
        "case_id": query_case.get("case_id"),
        "split": query_case.get("split"),
        "profile": profile,
        "title": query_case.get("title", ""),
        "expected_primary_domain": query_case.get("expected_primary_domain"),
        "expected_domains": query_case.get("expected_domains", []),
        "predicted_primary_domain": primary,
        "predicted_domains": predicted_domains,
        "expected_behavior": query_case.get("expected_behavior"),
        "predicted_behavior": "route_to_expected_domains",
        "source_url": query_case.get("source_url"),
        "question_tags": query_case.get("question_tags", []),
        "query_tags": query_case.get("query_tags", []),
        "neighbor_case_ids": [neighbor["case_id"] for neighbor in neighbors],
        "neighbor_scores": [neighbor["score"] for neighbor in neighbors],
    }
