"""Load normalized POC candidate sets with explicit downstream-use gates.

The normalized candidate artifacts are intentionally not training data, indexes,
or generation-ready records. This module keeps that boundary executable: callers
must name the downstream use they need, and records are rejected unless the
record-level ``downstream_allowed`` map explicitly permits that use.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable

from it_support.config import DATA_DIR


NORMALIZED_CANDIDATE_DIR = DATA_DIR / "processed" / "normalized_candidate_sets"


class DownstreamUse(StrEnum):
    """Named downstream gates stored on normalized candidate records."""

    CLASSIFIER_POOL = "classifier_pool"
    RETRIEVAL = "retrieval"
    ANSWER_GENERATION = "answer_generation"
    FINE_TUNING = "fine_tuning"
    EVALUATION = "evaluation"
    COMMERCIAL_MODE = "commercial_mode"


class NormalizedArtifact(StrEnum):
    """Known normalized candidate-set artifacts."""

    STACK_EXCHANGE_CLASSIFIER_POOL = "stack_exchange_classifier_pool"
    STACK_EXCHANGE_RETRIEVAL_CANDIDATES = "stack_exchange_retrieval_candidates"
    STACK_EXCHANGE_SAFETY_EVAL_FIXTURES = "stack_exchange_safety_eval_fixtures"
    STACK_EXCHANGE_MANUAL_REVIEW_QUEUE = "stack_exchange_manual_review_queue"
    TICKET_DATASET_MANUAL_REVIEW_QUEUE = "ticket_dataset_manual_review_queue"


ARTIFACT_FILES: dict[NormalizedArtifact, str] = {
    NormalizedArtifact.STACK_EXCHANGE_CLASSIFIER_POOL: "stack_exchange_classifier_pool.jsonl",
    NormalizedArtifact.STACK_EXCHANGE_RETRIEVAL_CANDIDATES: (
        "stack_exchange_retrieval_candidates.jsonl"
    ),
    NormalizedArtifact.STACK_EXCHANGE_SAFETY_EVAL_FIXTURES: (
        "stack_exchange_safety_eval_fixtures.jsonl"
    ),
    NormalizedArtifact.STACK_EXCHANGE_MANUAL_REVIEW_QUEUE: (
        "stack_exchange_manual_review_queue.jsonl"
    ),
    NormalizedArtifact.TICKET_DATASET_MANUAL_REVIEW_QUEUE: (
        "ticket_dataset_manual_review_queue.jsonl"
    ),
}

ANSWER_EVIDENCE_KEYS = ("accepted_answer", "top_answer")
ANSWER_TEXT_FIELDS = ("answer_text", "answer_html", "accepted_answer", "top_answer")


class CandidateSetError(RuntimeError):
    """Base error for normalized candidate-set loading."""


class DownstreamUseBlocked(CandidateSetError):
    """Raised when a record is not allowed for a requested downstream use."""


class CandidateSetIntegrityError(CandidateSetError):
    """Raised when a candidate set violates the expected forensic contract."""


@dataclass(frozen=True)
class LoadResult:
    """Loaded records plus an audit summary."""

    artifact: NormalizedArtifact
    path: Path
    required_uses: tuple[DownstreamUse, ...]
    records: list[dict[str, Any]]
    audit: dict[str, Any]


def coerce_artifact(value: str | NormalizedArtifact) -> NormalizedArtifact:
    try:
        return value if isinstance(value, NormalizedArtifact) else NormalizedArtifact(value)
    except ValueError as exc:
        known = ", ".join(item.value for item in NormalizedArtifact)
        raise CandidateSetError(
            f"Unknown normalized artifact {value!r}; expected one of {known}"
        ) from exc


def coerce_use(value: str | DownstreamUse) -> DownstreamUse:
    try:
        return value if isinstance(value, DownstreamUse) else DownstreamUse(value)
    except ValueError as exc:
        known = ", ".join(item.value for item in DownstreamUse)
        raise CandidateSetError(
            f"Unknown downstream use {value!r}; expected one of {known}"
        ) from exc


def artifact_path(
    artifact: str | NormalizedArtifact,
    *,
    base_dir: Path = NORMALIZED_CANDIDATE_DIR,
) -> Path:
    normalized = coerce_artifact(artifact)
    return base_dir / ARTIFACT_FILES[normalized]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise CandidateSetIntegrityError(
                    f"Invalid JSONL in {path} at line {line_number}: {exc}"
                ) from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise CandidateSetError(f"Candidate artifact does not exist: {path}")
    return list(iter_jsonl(path))


def downstream_map(record: dict[str, Any]) -> dict[str, bool]:
    raw = record.get("downstream_allowed")
    if not isinstance(raw, dict):
        return {}
    return {str(key): bool(value) for key, value in raw.items()}


def is_allowed(record: dict[str, Any], use: str | DownstreamUse) -> bool:
    return downstream_map(record).get(coerce_use(use).value, False)


def contains_answer_evidence(record: dict[str, Any]) -> bool:
    return any(record.get(key) for key in ANSWER_EVIDENCE_KEYS)


def assert_allowed(record: dict[str, Any], uses: Iterable[DownstreamUse]) -> None:
    blocked = [use.value for use in uses if not is_allowed(record, use)]
    if blocked:
        record_id = record.get("record_id") or record.get("dataset_id") or "<unknown>"
        raise DownstreamUseBlocked(
            f"Record {record_id!r} is blocked for downstream use(s): {', '.join(blocked)}"
        )


def validate_record(record: dict[str, Any], artifact: NormalizedArtifact) -> list[str]:
    errors: list[str] = []
    identifier = record.get("record_id") or record.get("dataset_id")
    if not identifier:
        errors.append("missing record_id/dataset_id")
    if not isinstance(record.get("downstream_allowed"), dict):
        errors.append(f"{identifier}: missing downstream_allowed map")

    if artifact == NormalizedArtifact.STACK_EXCHANGE_RETRIEVAL_CANDIDATES:
        if not is_allowed(record, DownstreamUse.RETRIEVAL):
            errors.append(f"{identifier}: retrieval candidate is not retrieval-allowed")
        if is_allowed(record, DownstreamUse.ANSWER_GENERATION):
            errors.append(f"{identifier}: answer_generation unexpectedly allowed")
        if not contains_answer_evidence(record):
            errors.append(f"{identifier}: retrieval candidate has no answer evidence")

    if artifact == NormalizedArtifact.STACK_EXCHANGE_SAFETY_EVAL_FIXTURES:
        if record.get("answer_text_included") is not False:
            errors.append(f"{identifier}: safety fixture must declare answer_text_included=false")
        if contains_answer_evidence(record):
            errors.append(f"{identifier}: safety fixture includes answer evidence")
        if not is_allowed(record, DownstreamUse.EVALUATION):
            errors.append(f"{identifier}: safety fixture is not evaluation-allowed")

    if is_allowed(record, DownstreamUse.COMMERCIAL_MODE):
        errors.append(
            f"{identifier}: commercial_mode should be false for current public POC records"
        )

    return errors


def build_load_audit(
    records: list[dict[str, Any]],
    *,
    artifact: str | NormalizedArtifact,
    path: Path | None = None,
    required_uses: Iterable[str | DownstreamUse] = (),
) -> dict[str, Any]:
    normalized_artifact = coerce_artifact(artifact)
    uses = tuple(coerce_use(use) for use in required_uses)
    downstream_counts: dict[str, int] = {}
    for use in DownstreamUse:
        downstream_counts[use.value] = sum(1 for row in records if is_allowed(row, use))

    blocked_by_use: dict[str, int] = {}
    blocked_samples: dict[str, list[str]] = {}
    for use in uses:
        blocked = [
            str(row.get("record_id") or row.get("dataset_id") or "<unknown>")
            for row in records
            if not is_allowed(row, use)
        ]
        blocked_by_use[use.value] = len(blocked)
        blocked_samples[use.value] = blocked[:10]

    validation_errors = [
        error for row in records for error in validate_record(row, normalized_artifact)
    ]

    answer_evidence_records = sum(1 for row in records if contains_answer_evidence(row))
    answer_generation_allowed = sum(
        1 for row in records if is_allowed(row, DownstreamUse.ANSWER_GENERATION)
    )

    return {
        "artifact": normalized_artifact.value,
        "path": str(path) if path else None,
        "records": len(records),
        "required_uses": [use.value for use in uses],
        "blocked_by_required_use": blocked_by_use,
        "blocked_samples": blocked_samples,
        "downstream_allowed_counts": downstream_counts,
        "domain_counts": dict(Counter(row.get("primary_domain", "unknown") for row in records)),
        "gate_decision_counts": dict(
            Counter(row.get("gate_decision", "unknown") for row in records)
        ),
        "license_counts": dict(Counter(row.get("license", "unknown") for row in records)),
        "commercial_posture_counts": dict(
            Counter(row.get("commercial_posture", "unknown") for row in records)
        ),
        "answer_evidence_records": answer_evidence_records,
        "answer_generation_allowed_records": answer_generation_allowed,
        "validation_error_count": len(validation_errors),
        "validation_errors_sample": validation_errors[:25],
    }


def load_records(
    artifact: str | NormalizedArtifact,
    *,
    required_uses: Iterable[str | DownstreamUse] = (),
    base_dir: Path = NORMALIZED_CANDIDATE_DIR,
    strict: bool = True,
) -> LoadResult:
    normalized_artifact = coerce_artifact(artifact)
    uses = tuple(coerce_use(use) for use in required_uses)
    path = artifact_path(normalized_artifact, base_dir=base_dir)
    records = read_jsonl(path)
    audit = build_load_audit(records, artifact=normalized_artifact, path=path, required_uses=uses)

    if strict and audit["validation_error_count"]:
        sample = "\n".join(audit["validation_errors_sample"])
        raise CandidateSetIntegrityError(
            f"{normalized_artifact.value} failed validation with "
            f"{audit['validation_error_count']} error(s):\n{sample}"
        )

    blocked = {
        use: count
        for use, count in audit["blocked_by_required_use"].items()
        if count
    }
    if strict and blocked:
        details = "; ".join(f"{use}={count}" for use, count in blocked.items())
        raise DownstreamUseBlocked(
            f"{normalized_artifact.value} has records blocked for requested use(s): {details}"
        )

    if uses:
        filtered = [row for row in records if all(is_allowed(row, use) for use in uses)]
    else:
        filtered = records

    return LoadResult(
        artifact=normalized_artifact,
        path=path,
        required_uses=uses,
        records=filtered,
        audit=audit,
    )


def unique_expected_domains(record: dict[str, Any]) -> list[str]:
    domains: list[str] = []
    primary = record.get("primary_domain")
    if primary:
        domains.append(str(primary))
    for domain in record.get("secondary_domains") or []:
        if domain and domain not in domains:
            domains.append(str(domain))
    return domains


def attribution_refs_for_question(record: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    for ref in record.get("attribution_refs") or []:
        if ref.get("record_type") == "question":
            refs.append(ref)
    return refs[:1]


def project_routing_eval_case(record: dict[str, Any], *, split: str) -> dict[str, Any]:
    """Create a question-only routing eval case from a normalized record."""

    assert_allowed(record, (DownstreamUse.EVALUATION,))
    case = {
        "case_id": record["record_id"],
        "split": split,
        "source_family_id": record.get("source_family_id"),
        "source_run_id": record.get("source_run_id"),
        "source_url": record.get("source_url"),
        "site": record.get("site"),
        "question_id": record.get("question_id"),
        "title": record.get("title", ""),
        "question_text": record.get("question_text", ""),
        "question_tags": record.get("question_tags", []),
        "query_tags": record.get("query_tags", []),
        "expected_primary_domain": record.get("primary_domain"),
        "expected_domains": unique_expected_domains(record),
        "ranked_domains": record.get("ranked_domains", []),
        "safety_flags": record.get("safety_flags", {}),
        "gate_decision": record.get("gate_decision"),
        "expected_behavior": "route_to_expected_domains",
        "license": record.get("license"),
        "commercial_posture": record.get("commercial_posture"),
        "commercial_reuse_allowed": False,
        "attribution_refs": attribution_refs_for_question(record),
        "downstream_allowed": {
            "evaluation": True,
            "commercial_mode": False,
            "answer_generation": False,
            "fine_tuning": False,
        },
    }
    return case


def project_safety_eval_case(record: dict[str, Any], *, split: str) -> dict[str, Any]:
    """Create a question-only safety/escalation eval case."""

    assert_allowed(record, (DownstreamUse.EVALUATION,))
    case = project_routing_eval_case(record, split=split)
    case.update(
        {
            "expected_behavior": record.get("expected_behavior"),
            "blocked_reason": record.get("blocked_reason"),
            "answer_text_included": False,
            "retrieval_allowed": bool(is_allowed(record, DownstreamUse.RETRIEVAL)),
        }
    )
    return case


def stable_bucket(value: str, *, modulus: int = 100) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


def split_records_by_group(
    records: Iterable[dict[str, Any]],
    *,
    group_field: str,
    holdout_fraction: float = 0.25,
    dev_name: str = "dev",
    holdout_name: str = "holdout",
) -> dict[str, str]:
    """Create deterministic, roughly stratified dev/holdout assignments."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(group_field) or "unknown")].append(record)

    assignments: dict[str, str] = {}
    for group_records in grouped.values():
        sorted_records = sorted(
            group_records,
            key=lambda row: (
                stable_bucket(str(row.get("record_id") or row.get("case_id"))),
                str(row),
            ),
        )
        holdout_count = round(len(sorted_records) * holdout_fraction)
        if len(sorted_records) > 1:
            holdout_count = max(1, holdout_count)
        else:
            holdout_count = 0
        holdout_ids = {
            str(row.get("record_id") or row.get("case_id"))
            for row in sorted_records[:holdout_count]
        }
        for row in sorted_records:
            record_id = str(row.get("record_id") or row.get("case_id"))
            assignments[record_id] = holdout_name if record_id in holdout_ids else dev_name
    return assignments
