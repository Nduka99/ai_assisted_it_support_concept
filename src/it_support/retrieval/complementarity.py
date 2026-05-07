"""Query-level comparison helpers for retrieval model complementarity."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def first_rank_at_min_grade(
    retrieved: list[dict[str, Any]],
    *,
    min_grade: int,
) -> int | None:
    for item in retrieved:
        if int(item.get("relevance_grade", 0)) >= min_grade:
            return int(item["rank"])
    return None


def hit_at(rank: int | None, cutoff: int) -> bool:
    return rank is not None and rank <= cutoff


def outcome_at_cutoff(
    *,
    baseline_rank: int | None,
    challenger_rank: int | None,
    cutoff: int,
    baseline_label: str = "bge",
    challenger_label: str = "qwen3",
) -> str:
    baseline_hit = hit_at(baseline_rank, cutoff)
    challenger_hit = hit_at(challenger_rank, cutoff)
    if baseline_hit and challenger_hit:
        return "both_hit"
    if baseline_hit:
        return f"{baseline_label}_only"
    if challenger_hit:
        return f"{challenger_label}_only"
    return "neither_hit"


def rank_advantage(
    *,
    baseline_rank: int | None,
    challenger_rank: int | None,
    baseline_label: str = "bge",
    challenger_label: str = "qwen3",
) -> str:
    if baseline_rank is None and challenger_rank is None:
        return "both_miss_top10"
    if baseline_rank is None:
        return f"{challenger_label}_only_top10"
    if challenger_rank is None:
        return f"{baseline_label}_only_top10"
    if baseline_rank < challenger_rank:
        return f"{baseline_label}_higher_rank"
    if challenger_rank < baseline_rank:
        return f"{challenger_label}_higher_rank"
    return "same_first_rank"


def best_retrieved_item(
    retrieved: list[dict[str, Any]],
    *,
    min_grade: int,
) -> dict[str, Any] | None:
    for item in retrieved:
        if int(item.get("relevance_grade", 0)) >= min_grade:
            return item
    return None


def top_item(retrieved: list[dict[str, Any]]) -> dict[str, Any]:
    if not retrieved:
        return {}
    return retrieved[0]


def compare_prediction_rows(
    *,
    baseline_rows: list[dict[str, Any]],
    challenger_rows: list[dict[str, Any]],
    qrels_by_query: dict[str, dict[str, int]],
    min_grade: int = 2,
    cutoffs: tuple[int, ...] = (1, 5, 10),
    baseline_label: str = "bge",
    challenger_label: str = "qwen3",
) -> list[dict[str, Any]]:
    baseline_by_query = {str(row["query_id"]): row for row in baseline_rows}
    challenger_by_query = {str(row["query_id"]): row for row in challenger_rows}
    if set(baseline_by_query) != set(challenger_by_query):
        missing_baseline = sorted(set(challenger_by_query) - set(baseline_by_query))
        missing_challenger = sorted(set(baseline_by_query) - set(challenger_by_query))
        raise ValueError(
            "Prediction query sets differ: "
            f"missing_baseline={missing_baseline[:5]} "
            f"missing_challenger={missing_challenger[:5]}"
        )

    rows = []
    for query_id in sorted(baseline_by_query):
        baseline = baseline_by_query[query_id]
        challenger = challenger_by_query[query_id]
        baseline_retrieved = baseline.get("retrieved", [])
        challenger_retrieved = challenger.get("retrieved", [])
        baseline_rank = first_rank_at_min_grade(baseline_retrieved, min_grade=min_grade)
        challenger_rank = first_rank_at_min_grade(challenger_retrieved, min_grade=min_grade)
        baseline_top = top_item(baseline_retrieved)
        challenger_top = top_item(challenger_retrieved)
        baseline_best = best_retrieved_item(baseline_retrieved, min_grade=min_grade) or {}
        challenger_best = best_retrieved_item(challenger_retrieved, min_grade=min_grade) or {}
        qrels = qrels_by_query.get(query_id, {})
        available_min_grade = sum(1 for grade in qrels.values() if int(grade) >= min_grade)

        row = {
            "query_id": query_id,
            "split": baseline.get("split"),
            "title": baseline.get("title", ""),
            "expected_primary_domain": baseline.get("expected_primary_domain"),
            "expected_domains": "|".join(str(value) for value in baseline.get("expected_domains", [])),
            "triage_predicted_domains": "|".join(
                str(value) for value in baseline.get("triage_predicted_domains", [])
            ),
            "available_grade_2_docs": available_min_grade,
            "is_grade_2_eligible": available_min_grade > 0,
            "baseline_profile": baseline.get("profile"),
            "challenger_profile": challenger.get("profile"),
            f"{baseline_label}_first_grade_2_rank": baseline_rank or "",
            f"{challenger_label}_first_grade_2_rank": challenger_rank or "",
            f"{baseline_label}_top_doc_id": baseline_top.get("doc_id", ""),
            f"{baseline_label}_top_title": baseline_top.get("title", ""),
            f"{baseline_label}_top_relevance_grade": baseline_top.get("relevance_grade", ""),
            f"{challenger_label}_top_doc_id": challenger_top.get("doc_id", ""),
            f"{challenger_label}_top_title": challenger_top.get("title", ""),
            f"{challenger_label}_top_relevance_grade": challenger_top.get("relevance_grade", ""),
            f"{baseline_label}_first_grade_2_doc_id": baseline_best.get("doc_id", ""),
            f"{baseline_label}_first_grade_2_title": baseline_best.get("title", ""),
            f"{challenger_label}_first_grade_2_doc_id": challenger_best.get("doc_id", ""),
            f"{challenger_label}_first_grade_2_title": challenger_best.get("title", ""),
            "rank_advantage": rank_advantage(
                baseline_rank=baseline_rank,
                challenger_rank=challenger_rank,
                baseline_label=baseline_label,
                challenger_label=challenger_label,
            ),
        }
        for cutoff in cutoffs:
            row[f"{baseline_label}_grade_2_hit_at_{cutoff}"] = hit_at(baseline_rank, cutoff)
            row[f"{challenger_label}_grade_2_hit_at_{cutoff}"] = hit_at(challenger_rank, cutoff)
            row[f"outcome_at_{cutoff}"] = outcome_at_cutoff(
                baseline_rank=baseline_rank,
                challenger_rank=challenger_rank,
                cutoff=cutoff,
                baseline_label=baseline_label,
                challenger_label=challenger_label,
            )
        rows.append(row)
    return rows


def outcome_count_rows(
    comparison_rows: list[dict[str, Any]],
    *,
    cutoffs: tuple[int, ...] = (1, 5, 10),
) -> list[dict[str, Any]]:
    rows = []
    total = len(comparison_rows)
    for cutoff in cutoffs:
        counts = Counter(str(row[f"outcome_at_{cutoff}"]) for row in comparison_rows)
        for outcome in ("both_hit", "bge_only", "qwen3_only", "neither_hit"):
            count = counts.get(outcome, 0)
            rows.append(
                {
                    "cutoff": cutoff,
                    "outcome": outcome,
                    "queries": count,
                    "share": round(count / total, 4) if total else 0.0,
                }
            )
    return rows


def domain_complementarity_rows(
    comparison_rows: list[dict[str, Any]],
    *,
    cutoffs: tuple[int, ...] = (1, 5, 10),
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        grouped[str(row.get("expected_primary_domain") or "unknown")].append(row)

    rows = []
    for domain, domain_rows in sorted(grouped.items()):
        out: dict[str, Any] = {
            "expected_primary_domain": domain,
            "eligible_queries": len(domain_rows),
        }
        for cutoff in cutoffs:
            counts = Counter(str(row[f"outcome_at_{cutoff}"]) for row in domain_rows)
            out[f"both_hit_at_{cutoff}"] = counts.get("both_hit", 0)
            out[f"bge_only_at_{cutoff}"] = counts.get("bge_only", 0)
            out[f"qwen3_only_at_{cutoff}"] = counts.get("qwen3_only", 0)
            out[f"neither_hit_at_{cutoff}"] = counts.get("neither_hit", 0)
        rows.append(out)
    return rows


def rank_advantage_rows(comparison_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(comparison_rows)
    counts = Counter(str(row["rank_advantage"]) for row in comparison_rows)
    return [
        {
            "rank_advantage": advantage,
            "queries": count,
            "share": round(count / total, 4) if total else 0.0,
        }
        for advantage, count in sorted(counts.items())
    ]


def diagnostic_upper_bound_rows(
    comparison_rows: list[dict[str, Any]],
    *,
    cutoffs: tuple[int, ...] = (1, 5, 10),
    baseline_label: str = "bge",
    challenger_label: str = "qwen3",
) -> list[dict[str, Any]]:
    total = len(comparison_rows)
    rows = []
    for cutoff in cutoffs:
        baseline_hits = sum(
            1 for row in comparison_rows if bool(row[f"{baseline_label}_grade_2_hit_at_{cutoff}"])
        )
        challenger_hits = sum(
            1
            for row in comparison_rows
            if bool(row[f"{challenger_label}_grade_2_hit_at_{cutoff}"])
        )
        either_hits = sum(1 for row in comparison_rows if row[f"outcome_at_{cutoff}"] != "neither_hit")
        both_hits = sum(1 for row in comparison_rows if row[f"outcome_at_{cutoff}"] == "both_hit")
        rows.append(
            {
                "cutoff": cutoff,
                f"{baseline_label}_hit_rate": round(baseline_hits / total, 4) if total else 0.0,
                f"{challenger_label}_hit_rate": round(challenger_hits / total, 4) if total else 0.0,
                "either_model_hit_rate": round(either_hits / total, 4) if total else 0.0,
                "both_models_hit_rate": round(both_hits / total, 4) if total else 0.0,
                "either_model_hits": either_hits,
                "both_model_hits": both_hits,
                "queries": total,
            }
        )
    return rows


def example_rows(
    comparison_rows: list[dict[str, Any]],
    *,
    limit_per_type: int = 12,
) -> list[dict[str, Any]]:
    example_specs = [
        ("qwen3_rank1_bge_miss", "outcome_at_1", "qwen3_only"),
        ("bge_rank1_qwen3_miss", "outcome_at_1", "bge_only"),
        ("qwen3_top5_bge_miss", "outcome_at_5", "qwen3_only"),
        ("bge_top5_qwen3_miss", "outcome_at_5", "bge_only"),
        ("both_top5_miss", "outcome_at_5", "neither_hit"),
    ]
    rows = []
    for example_type, field, value in example_specs:
        matches = [row for row in comparison_rows if row.get(field) == value]
        matches = sorted(
            matches,
            key=lambda row: (
                str(row.get("expected_primary_domain") or ""),
                str(row.get("query_id") or ""),
            ),
        )
        for row in matches[:limit_per_type]:
            rows.append(
                {
                    "example_type": example_type,
                    "query_id": row["query_id"],
                    "expected_primary_domain": row.get("expected_primary_domain", ""),
                    "title": row.get("title", ""),
                    "bge_first_grade_2_rank": row.get("bge_first_grade_2_rank", ""),
                    "qwen3_first_grade_2_rank": row.get("qwen3_first_grade_2_rank", ""),
                    "bge_top_title": row.get("bge_top_title", ""),
                    "qwen3_top_title": row.get("qwen3_top_title", ""),
                    "bge_first_grade_2_title": row.get("bge_first_grade_2_title", ""),
                    "qwen3_first_grade_2_title": row.get("qwen3_first_grade_2_title", ""),
                }
            )
    return rows
