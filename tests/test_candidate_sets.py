from __future__ import annotations

import pytest

from it_support.data import (
    DownstreamUse,
    DownstreamUseBlocked,
    NormalizedArtifact,
    load_records,
    project_routing_eval_case,
    split_records_by_group,
)


def test_retrieval_candidates_require_retrieval_gate() -> None:
    result = load_records(
        NormalizedArtifact.STACK_EXCHANGE_RETRIEVAL_CANDIDATES,
        required_uses=(DownstreamUse.RETRIEVAL,),
    )

    assert len(result.records) == 516
    assert result.audit["answer_generation_allowed_records"] == 0
    assert result.audit["blocked_by_required_use"]["retrieval"] == 0


def test_safety_fixtures_are_not_retrieval_allowed() -> None:
    with pytest.raises(DownstreamUseBlocked):
        load_records(
            NormalizedArtifact.STACK_EXCHANGE_SAFETY_EVAL_FIXTURES,
            required_uses=(DownstreamUse.RETRIEVAL,),
        )


def test_routing_eval_projection_is_question_only() -> None:
    result = load_records(
        NormalizedArtifact.STACK_EXCHANGE_RETRIEVAL_CANDIDATES,
        required_uses=(DownstreamUse.EVALUATION,),
    )
    case = project_routing_eval_case(result.records[0], split="routing_dev_eval")

    assert case["case_id"] == result.records[0]["record_id"]
    assert case["downstream_allowed"]["answer_generation"] is False
    assert "accepted_answer" not in case
    assert "top_answer" not in case


def test_split_records_by_group_is_deterministic() -> None:
    result = load_records(
        NormalizedArtifact.STACK_EXCHANGE_CLASSIFIER_POOL,
        required_uses=(DownstreamUse.EVALUATION,),
    )

    first = split_records_by_group(result.records, group_field="primary_domain")
    second = split_records_by_group(result.records, group_field="primary_domain")

    assert first == second
    assert set(first.values()) == {"dev", "holdout"}
