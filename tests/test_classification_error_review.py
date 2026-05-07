from __future__ import annotations

from it_support.classification.error_review import (
    multilabel_gap_count_rows,
    primary_error_count_rows,
    rank_bucket,
    ranked_primary_bucket_rows,
    safety_signal_gap_count_rows,
)


def test_primary_error_count_rows_excludes_correct_predictions() -> None:
    rows = [
        {
            "profile": "p",
            "expected_primary_domain": "network_connectivity",
            "predicted_primary_domain": "network_connectivity",
            "primary_correct": "True",
        },
        {
            "profile": "p",
            "expected_primary_domain": "network_connectivity",
            "predicted_primary_domain": "application_software",
            "primary_correct": "False",
        },
        {
            "profile": "p",
            "expected_primary_domain": "network_connectivity",
            "predicted_primary_domain": "application_software",
            "primary_correct": "False",
        },
    ]

    counts = primary_error_count_rows(rows)

    assert counts == [
        {
            "profile": "p",
            "expected_primary_domain": "network_connectivity",
            "predicted_primary_domain": "application_software",
            "records": 2,
        }
    ]


def test_multilabel_gap_count_rows_separates_missing_and_extra_domains() -> None:
    rows = [
        {
            "profile": "p",
            "expected_domains": "network_connectivity|hardware",
            "predicted_domains": "network_connectivity|application_software",
        }
    ]

    counts = multilabel_gap_count_rows(rows)

    assert {
        (row["gap_type"], row["domain"]): row["records"]
        for row in counts
    } == {
        ("missing_expected", "hardware"): 1,
        ("extra_predicted", "application_software"): 1,
    }


def test_rank_bucket_and_ranked_primary_bucket_rows() -> None:
    rows = [
        {
            "profile": "p",
            "expected_primary_domain": "network_connectivity",
            "first_primary_rank": "1",
        },
        {
            "profile": "p",
            "expected_primary_domain": "network_connectivity",
            "first_primary_rank": "",
        },
    ]

    assert rank_bucket("4", cutoff=3) == "rank_gt_3"
    assert rank_bucket("", cutoff=3) == "not_predicted"
    assert {
        row["rank_bucket"]: row["records"]
        for row in ranked_primary_bucket_rows(rows, cutoff=3)
    } == {"rank_1": 1, "not_predicted": 1}


def test_safety_signal_gap_count_rows_counts_pipe_separated_gaps() -> None:
    rows = [
        {
            "profile": "p",
            "missed_safety_signals": "data_loss_risk|needs_human_review",
            "extra_safety_signals": "",
        },
        {
            "profile": "p",
            "missed_safety_signals": "data_loss_risk",
            "extra_safety_signals": "credential_secret_risk",
        },
    ]

    counts = safety_signal_gap_count_rows(rows)

    assert {
        (row["gap_type"], row["safety_signal"]): row["records"]
        for row in counts
    } == {
        ("missed_expected", "data_loss_risk"): 2,
        ("missed_expected", "needs_human_review"): 1,
        ("extra_predicted", "credential_secret_risk"): 1,
    }
