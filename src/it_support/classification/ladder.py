"""Metrics for the dev-only classification ladder."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def round_metric(value: float) -> float:
    return round(value, 4)


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ranked_prediction_labels(row: dict[str, Any]) -> list[str]:
    labels = []
    seen = set()
    for item in row.get("predicted_domains") or []:
        label = str(item.get("label") or "").strip()
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def first_rank(labels: list[str], wanted: set[str]) -> int | None:
    for index, label in enumerate(labels, start=1):
        if label in wanted:
            return index
    return None


def hit_at(rank: int | None, cutoff: int) -> bool:
    return rank is not None and rank <= cutoff


def dcg(grades: list[int], *, cutoff: int) -> float:
    return sum(
        ((2**grade - 1) / math.log2(rank + 1))
        for rank, grade in enumerate(grades[:cutoff], start=1)
    )


def _multiclass_report(
    prediction_rows: list[dict[str, Any]],
    *,
    labels: list[str],
    profile: str,
    expected_field: str,
    predicted_field: str,
    label_column: str,
    confusion_expected_column: str,
    confusion_predicted_column: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    totals = Counter()
    confusion = Counter()
    correct = 0

    for row in prediction_rows:
        expected = str(row.get(expected_field) or "")
        predicted = str(row.get(predicted_field) or "")
        if expected == predicted:
            correct += 1
        confusion[(expected, predicted)] += 1
        for label in labels:
            if expected == label and predicted == label:
                totals[(label, "tp")] += 1
            elif expected != label and predicted == label:
                totals[(label, "fp")] += 1
            elif expected == label and predicted != label:
                totals[(label, "fn")] += 1
            else:
                totals[(label, "tn")] += 1

    per_label = []
    weighted_f1_numerator = 0.0
    cases = len(prediction_rows)
    for label in labels:
        tp = totals[(label, "tp")]
        fp = totals[(label, "fp")]
        fn = totals[(label, "fn")]
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        label_f1 = f1(precision, recall)
        weighted_f1_numerator += label_f1 * support
        per_label.append(
            {
                "profile": profile,
                label_column: label,
                "support": support,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round_metric(precision),
                "recall": round_metric(recall),
                "f1": round_metric(label_f1),
            }
        )

    macro_f1 = sum(row["f1"] for row in per_label) / len(per_label) if per_label else 0.0
    summary = {
        "profile": profile,
        "cases": cases,
        "accuracy": round_metric(correct / cases) if cases else 0.0,
        "macro_f1": round_metric(macro_f1),
        "weighted_f1": round_metric(weighted_f1_numerator / cases) if cases else 0.0,
    }
    confusion_rows = [
            {
                "profile": profile,
                confusion_expected_column: expected,
                confusion_predicted_column: predicted,
                "records": count,
            }
        for (expected, predicted), count in sorted(confusion.items())
    ]
    return summary, per_label, confusion_rows


def primary_multiclass_report(
    prediction_rows: list[dict[str, Any]],
    *,
    labels: list[str],
    profile: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    return _multiclass_report(
        prediction_rows,
        labels=labels,
        profile=profile,
        expected_field="expected_primary_domain",
        predicted_field="predicted_primary_domain",
        label_column="domain",
        confusion_expected_column="expected_primary_domain",
        confusion_predicted_column="predicted_primary_domain",
    )


def multilabel_domain_report(
    prediction_rows: list[dict[str, Any]],
    *,
    labels: list[str],
    profile: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = Counter()
    exact_matches = 0
    total_expected = 0
    total_predicted = 0

    for row in prediction_rows:
        expected = set(row.get("expected_domains") or [])
        predicted = set(ranked_prediction_labels(row))
        total_expected += len(expected)
        total_predicted += len(predicted)
        if expected == predicted:
            exact_matches += 1
        for label in labels:
            expected_has = label in expected
            predicted_has = label in predicted
            if expected_has and predicted_has:
                totals[(label, "tp")] += 1
                totals[("micro", "tp")] += 1
            elif not expected_has and predicted_has:
                totals[(label, "fp")] += 1
                totals[("micro", "fp")] += 1
            elif expected_has and not predicted_has:
                totals[(label, "fn")] += 1
                totals[("micro", "fn")] += 1
            else:
                totals[(label, "tn")] += 1

    per_label = []
    hamming_errors = 0
    for label in labels:
        tp = totals[(label, "tp")]
        fp = totals[(label, "fp")]
        fn = totals[(label, "fn")]
        tn = totals[(label, "tn")]
        hamming_errors += fp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label.append(
            {
                "profile": profile,
                "domain": label,
                "support": tp + fn,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": round_metric(precision),
                "recall": round_metric(recall),
                "f1": round_metric(f1(precision, recall)),
            }
        )

    micro_tp = totals[("micro", "tp")]
    micro_fp = totals[("micro", "fp")]
    micro_fn = totals[("micro", "fn")]
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    cases = len(prediction_rows)
    macro_f1 = sum(row["f1"] for row in per_label) / len(per_label) if per_label else 0.0
    summary = {
        "profile": profile,
        "cases": cases,
        "exact_match_ratio": round_metric(exact_matches / cases) if cases else 0.0,
        "micro_precision": round_metric(micro_precision),
        "micro_recall": round_metric(micro_recall),
        "micro_f1": round_metric(f1(micro_precision, micro_recall)),
        "macro_f1": round_metric(macro_f1),
        "hamming_loss": round_metric(hamming_errors / (cases * len(labels)))
        if cases and labels
        else 0.0,
        "avg_expected_labels": round_metric(total_expected / cases) if cases else 0.0,
        "avg_predicted_labels": round_metric(total_predicted / cases) if cases else 0.0,
    }
    return summary, per_label


def ranked_domain_report(
    prediction_rows: list[dict[str, Any]],
    *,
    profile: str,
    cutoffs: tuple[int, ...] = (1, 2, 3, 5),
) -> dict[str, Any]:
    cases = len(prediction_rows)
    totals = Counter()
    primary_rr = 0.0
    any_rr = 0.0
    graded_ndcg = {cutoff: 0.0 for cutoff in cutoffs}

    for row in prediction_rows:
        labels = ranked_prediction_labels(row)
        expected_primary = str(row.get("expected_primary_domain") or "")
        expected_domains = set(row.get("expected_domains") or [])
        primary_rank = first_rank(labels, {expected_primary})
        any_rank = first_rank(labels, expected_domains)
        if primary_rank:
            primary_rr += 1.0 / primary_rank
        if any_rank:
            any_rr += 1.0 / any_rank

        for cutoff in cutoffs:
            totals[("primary", cutoff)] += int(hit_at(primary_rank, cutoff))
            totals[("any_expected", cutoff)] += int(hit_at(any_rank, cutoff))

            actual_grades = [
                2 if label == expected_primary else 1 if label in expected_domains else 0
                for label in labels
            ]
            ideal_grades = [2] if expected_primary else []
            ideal_grades.extend([1] * max(0, len(expected_domains - {expected_primary})))
            ideal = dcg(ideal_grades, cutoff=cutoff)
            graded_ndcg[cutoff] += dcg(actual_grades, cutoff=cutoff) / ideal if ideal else 0.0

    row: dict[str, Any] = {
        "profile": profile,
        "cases": cases,
        "primary_mrr": round_metric(primary_rr / cases) if cases else 0.0,
        "any_expected_mrr": round_metric(any_rr / cases) if cases else 0.0,
    }
    for cutoff in cutoffs:
        row[f"primary_hit_at_{cutoff}"] = (
            round_metric(totals[("primary", cutoff)] / cases) if cases else 0.0
        )
        row[f"any_expected_hit_at_{cutoff}"] = round_metric(
            totals[("any_expected", cutoff)] / cases
        ) if cases else 0.0
        row[f"graded_ndcg_at_{cutoff}"] = (
            round_metric(graded_ndcg[cutoff] / cases) if cases else 0.0
        )
    return row


def safety_behavior_report(
    prediction_rows: list[dict[str, Any]],
    *,
    labels: list[str],
    profile: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    return _multiclass_report(
        prediction_rows,
        labels=labels,
        profile=profile,
        expected_field="expected_behavior",
        predicted_field="predicted_behavior",
        label_column="expected_behavior",
        confusion_expected_column="expected_behavior",
        confusion_predicted_column="predicted_behavior",
    )


def safety_signal_report(
    *,
    safety_cases: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    signal_labels: list[str],
    profile: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    cases_by_id = {str(row["case_id"]): row for row in safety_cases}
    totals = Counter()
    details = []

    for prediction in prediction_rows:
        case_id = str(prediction["case_id"])
        case = cases_by_id[case_id]
        expected_flags = case.get("safety_flags") or {}
        predicted_flags = prediction.get("safety_signals") or {}
        expected_true = {
            label for label in signal_labels if bool(expected_flags.get(label, False))
        }
        predicted_true = {
            label for label in signal_labels if bool(predicted_flags.get(label, False))
        }
        details.append(
            {
                "profile": profile,
                "case_id": case_id,
                "expected_behavior": case.get("expected_behavior"),
                "predicted_behavior": prediction.get("predicted_behavior"),
                "behavior_correct": case.get("expected_behavior")
                == prediction.get("predicted_behavior"),
                "expected_safety_signals": "|".join(sorted(expected_true)),
                "predicted_safety_signals": "|".join(sorted(predicted_true)),
                "missed_safety_signals": "|".join(sorted(expected_true - predicted_true)),
                "extra_safety_signals": "|".join(sorted(predicted_true - expected_true)),
                "title": prediction.get("title", ""),
                "source_url": prediction.get("source_url", ""),
            }
        )
        for label in signal_labels:
            expected_has = label in expected_true
            predicted_has = label in predicted_true
            if expected_has and predicted_has:
                totals[(label, "tp")] += 1
                totals[("micro", "tp")] += 1
            elif not expected_has and predicted_has:
                totals[(label, "fp")] += 1
                totals[("micro", "fp")] += 1
            elif expected_has and not predicted_has:
                totals[(label, "fn")] += 1
                totals[("micro", "fn")] += 1
            else:
                totals[(label, "tn")] += 1

    per_label = []
    for label in signal_labels:
        tp = totals[(label, "tp")]
        fp = totals[(label, "fp")]
        fn = totals[(label, "fn")]
        tn = totals[(label, "tn")]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        per_label.append(
            {
                "profile": profile,
                "safety_signal": label,
                "support": tp + fn,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": round_metric(precision),
                "recall": round_metric(recall),
                "f1": round_metric(f1(precision, recall)),
            }
        )

    micro_tp = totals[("micro", "tp")]
    micro_fp = totals[("micro", "fp")]
    micro_fn = totals[("micro", "fn")]
    micro_precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    micro_recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    macro_f1 = sum(row["f1"] for row in per_label) / len(per_label) if per_label else 0.0
    summary = {
        "profile": profile,
        "cases": len(prediction_rows),
        "micro_precision": round_metric(micro_precision),
        "micro_recall": round_metric(micro_recall),
        "micro_f1": round_metric(f1(micro_precision, micro_recall)),
        "macro_f1": round_metric(macro_f1),
    }
    return summary, per_label, details


def routing_case_detail_rows(
    prediction_rows: list[dict[str, Any]],
    *,
    profile: str,
) -> list[dict[str, Any]]:
    rows = []
    for row in prediction_rows:
        labels = ranked_prediction_labels(row)
        expected_primary = str(row.get("expected_primary_domain") or "")
        expected_domains = set(row.get("expected_domains") or [])
        primary_rank = first_rank(labels, {expected_primary})
        any_rank = first_rank(labels, expected_domains)
        rows.append(
            {
                "profile": profile,
                "case_id": row.get("case_id"),
                "expected_primary_domain": expected_primary,
                "predicted_primary_domain": row.get("predicted_primary_domain"),
                "primary_correct": expected_primary == row.get("predicted_primary_domain"),
                "expected_domains": "|".join(row.get("expected_domains") or []),
                "predicted_domains": "|".join(labels),
                "exact_match": expected_domains == set(labels),
                "first_primary_rank": primary_rank or "",
                "first_any_expected_rank": any_rank or "",
                "title": row.get("title", ""),
                "source_url": row.get("source_url", ""),
            }
        )
    return rows
