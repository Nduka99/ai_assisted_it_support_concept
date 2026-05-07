from __future__ import annotations

import numpy as np

from it_support.classification.embedding_classifier import (
    compact_case_text,
    predict_domains_from_neighbors,
    select_balanced_cases,
    top_neighbors,
)


def test_compact_case_text_optionally_includes_tags() -> None:
    case = {
        "title": "VPN fails",
        "question_text": "Cannot connect",
        "question_tags": ["vpn"],
        "query_tags": ["networking"],
    }

    assert compact_case_text(case, include_tags=False) == "VPN fails Cannot connect"
    assert compact_case_text(case, include_tags=True) == (
        "VPN fails Cannot connect vpn networking"
    )


def test_select_balanced_cases_samples_across_domains() -> None:
    cases = [
        {"case_id": f"n{i}", "expected_primary_domain": "network_connectivity"}
        for i in range(5)
    ] + [
        {"case_id": f"h{i}", "expected_primary_domain": "hardware"}
        for i in range(5)
    ]

    selected = select_balanced_cases(cases, max_cases=4)

    assert len(selected) == 4
    assert {
        row["expected_primary_domain"] for row in selected
    } == {"network_connectivity", "hardware"}


def test_top_neighbors_excludes_query_case() -> None:
    query = {"case_id": "q1"}
    refs = [{"case_id": "q1"}, {"case_id": "q2"}, {"case_id": "q3"}]
    neighbors = top_neighbors(
        query_case=query,
        reference_cases=refs,
        scores=np.asarray([0.99, 0.8, 0.7]),
        top_k=2,
    )

    assert [row["case_id"] for row in neighbors] == ["q2", "q3"]


def test_predict_domains_from_neighbors_votes_primary_and_secondary_labels() -> None:
    neighbors = [
        {
            "score": 0.9,
            "case": {
                "case_id": "n1",
                "expected_primary_domain": "network_connectivity",
                "expected_domains": ["network_connectivity", "os_kernel_drivers"],
            },
        },
        {
            "score": 0.8,
            "case": {
                "case_id": "n2",
                "expected_primary_domain": "network_connectivity",
                "expected_domains": ["network_connectivity", "hardware"],
            },
        },
    ]

    predicted = predict_domains_from_neighbors(
        neighbors=neighbors,
        max_domains=3,
        label_threshold_ratio=0.35,
    )

    assert predicted[0]["label"] == "network_connectivity"
    assert {row["label"] for row in predicted} == {
        "network_connectivity",
        "hardware",
        "os_kernel_drivers",
    }
