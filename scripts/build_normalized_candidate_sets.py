"""Build normalized POC candidate and review sets from preprocessing gates.

This stage promotes only gate-approved records into normalized POC candidates.
It also preserves signal from higher-risk records as question-only safety/eval
fixtures and review queues. It does not build indexes, train models, or create
generation-ready data from unreviewed security/firmware/ticket content.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
GATE = DATA / "processed" / "source_specific_preprocessing_gate"
OUT = DATA / "processed" / "normalized_candidate_sets"


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def read_json(path: Path) -> Any:
    return json.loads(read_text(path))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        values = [str(row.get(field, "")).replace("|", "\\|").replace("\n", " ") for field in fields]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"<(br|p|li|pre)[^>]*>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def compact_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def parse_json_field(value: str, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def answer_site_from_file(name: str) -> str:
    match = re.match(r"answers_(.+?)_chunk", name)
    if not match:
        raise ValueError(f"Cannot parse Stack Exchange site from {name}")
    return match.group(1)


def load_stack_raw(run_id: str) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]], dict[tuple[str, str, int, int | None], list[dict[str, Any]]]]:
    raw_dir = DATA / "raw" / "stack_exchange_it_support_sites" / run_id
    manifest = read_json(raw_dir / "run_manifest.json")
    questions: dict[tuple[str, int], dict[str, Any]] = {}
    answers: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)

    for filename in manifest["raw_question_files"]:
        payload = read_json(raw_dir / filename)
        query = payload["query"]
        site = query["site"]
        for item in payload["response"].get("items", []):
            if not item.get("question_id"):
                continue
            key = (site, int(item["question_id"]))
            record = questions.setdefault(
                key,
                {
                    "site": site,
                    "question_id": int(item["question_id"]),
                    "title": html.unescape(item.get("title", "")),
                    "question_text": strip_html(item.get("body")),
                    "question_html": item.get("body", ""),
                    "question_tags": item.get("tags", []),
                    "question_score": item.get("score"),
                    "question_url": item.get("link"),
                    "accepted_answer_id": item.get("accepted_answer_id"),
                    "creation_date": item.get("creation_date"),
                    "api_content_license": item.get("content_license"),
                },
            )
            record.setdefault("query_specs", []).append(query)

    for filename in manifest["raw_answer_files"]:
        site = answer_site_from_file(filename)
        payload = read_json(raw_dir / filename)
        for item in payload["response"].get("items", []):
            if not item.get("question_id") or not item.get("answer_id"):
                continue
            key = (site, int(item["question_id"]))
            if all(existing.get("answer_id") != item["answer_id"] for existing in answers[key]):
                answer = dict(item)
                answer["answer_text"] = strip_html(item.get("body"))
                answer["answer_html"] = item.get("body", "")
                answers[key].append(answer)

    attr_rows = read_jsonl(ROOT / manifest["attribution_path"])
    attr: dict[tuple[str, str, int, int | None], list[dict[str, Any]]] = defaultdict(list)
    for row in attr_rows:
        if not row.get("site") or row.get("question_id") is None:
            continue
        answer_id = row.get("answer_id")
        key = (row.get("record_type"), row["site"], int(row["question_id"]), int(answer_id) if answer_id else None)
        attr[key].append(row)
    return questions, answers, attr


def attribution_refs(attr: dict[tuple[str, str, int, int | None], list[dict[str, Any]]], site: str, question_id: int, answer_ids: list[int | None]) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    refs.extend(attr.get(("question", site, question_id, None), [])[:1])
    for answer_id in answer_ids:
        if answer_id is not None:
            refs.extend(attr.get(("answer", site, question_id, int(answer_id)), [])[:1])
    return [
        {
            "record_type": row.get("record_type"),
            "site": row.get("site"),
            "question_id": row.get("question_id"),
            "answer_id": row.get("answer_id"),
            "source_url": row.get("source_url"),
            "owner_display_name": row.get("owner_display_name"),
            "owner_link": row.get("owner_link"),
            "license": row.get("license"),
            "api_content_license": row.get("api_content_license"),
            "derived_license_by_creation_date": row.get("derived_license_by_creation_date"),
        }
        for row in refs
    ]


def stack_record_base(gate_row: dict[str, str], question: dict[str, Any]) -> dict[str, Any]:
    ranked = parse_json_field(gate_row.get("ranked_domains_json", ""), [])
    flags = parse_json_field(gate_row.get("safety_flags_json", ""), {})
    return {
        "record_id": gate_row["record_id"],
        "source_family_id": "stack_exchange_it_support_sites",
        "source_run_id": gate_row["run_id"],
        "source_url": gate_row["source_url"],
        "site": gate_row["site"],
        "question_id": int(gate_row["question_id"]),
        "title": question["title"],
        "question_text": question["question_text"],
        "question_tags": question["question_tags"],
        "query_tags": [tag for tag in gate_row.get("query_tags", "").split(";") if tag],
        "primary_domain": gate_row.get("primary_domain", ""),
        "secondary_domains": [domain for domain in gate_row.get("secondary_domains", "").split(";") if domain],
        "ranked_domains": ranked,
        "safety_flags": flags,
        "gate_decision": gate_row.get("gate_decision"),
        "use_lanes": [lane for lane in gate_row.get("use_lanes", "").split(";") if lane],
        "license": gate_row.get("license"),
        "commercial_posture": gate_row.get("commercial_posture"),
        "commercial_reuse_allowed": False,
        "attribution_required": True,
        "share_alike_required": True,
    }


def build_stack_sets(stack_run: str) -> dict[str, Any]:
    gate_rows = read_csv(GATE / "stack_exchange_scaled_run_gate.csv")
    questions, answers, attr = load_stack_raw(stack_run)

    classifier_pool: list[dict[str, Any]] = []
    retrieval_candidates: list[dict[str, Any]] = []
    safety_fixtures: list[dict[str, Any]] = []
    manual_review: list[dict[str, Any]] = []

    for gate_row in gate_rows:
        site = gate_row["site"]
        question_id = int(gate_row["question_id"])
        question = questions.get((site, question_id))
        if not question:
            continue
        answer_list = sorted(answers.get((site, question_id), []), key=lambda row: row.get("score") or 0, reverse=True)
        accepted_id = question.get("accepted_answer_id")
        accepted = next((row for row in answer_list if row.get("answer_id") == accepted_id), None)
        top = answer_list[0] if answer_list else None
        answer_refs = []
        if accepted:
            answer_refs.append(int(accepted["answer_id"]))
        if top and (not accepted or top["answer_id"] != accepted["answer_id"]):
            answer_refs.append(int(top["answer_id"]))

        base = stack_record_base(gate_row, question)
        base["attribution_refs"] = attribution_refs(attr, site, question_id, answer_refs)
        base["promotion_status"] = (
            "promoted_poc_candidate"
            if gate_row["gate_decision"] == "candidate_after_license_review"
            else "review_or_eval_only_pending_manual_promotion"
        )
        base["downstream_allowed"] = {
            "classifier_pool": True,
            "retrieval": gate_row["gate_decision"] == "candidate_after_license_review" and bool(accepted or top),
            "answer_generation": False,
            "fine_tuning": False,
            "evaluation": True,
            "commercial_mode": False,
        }
        classifier_pool.append(base)

        if gate_row["gate_decision"] == "candidate_after_license_review" and (accepted or top):
            retrieval = dict(base)
            retrieval["unit_type"] = "question_with_answer_evidence"
            retrieval["accepted_answer"] = (
                {
                    "answer_id": int(accepted["answer_id"]),
                    "answer_text": accepted["answer_text"],
                    "answer_score": accepted.get("score"),
                    "is_accepted": True,
                }
                if accepted
                else None
            )
            retrieval["top_answer"] = (
                {
                    "answer_id": int(top["answer_id"]),
                    "answer_text": top["answer_text"],
                    "answer_score": top.get("score"),
                    "is_accepted": bool(accepted and top["answer_id"] == accepted["answer_id"]),
                }
                if top
                else None
            )
            retrieval["downstream_allowed"] = {
                "classifier_pool": True,
                "retrieval": True,
                "answer_generation": False,
                "fine_tuning": False,
                "evaluation": True,
                "commercial_mode": False,
            }
            retrieval_candidates.append(retrieval)
        elif gate_row["gate_decision"] in {
            "requires_record_level_security_filter",
            "requires_firmware_escalation_policy_review",
        }:
            fixture = dict(base)
            fixture["unit_type"] = "question_only_safety_fixture"
            fixture["answer_text_included"] = False
            fixture["expected_behavior"] = (
                "structured_firmware_escalation"
                if gate_row["gate_decision"] == "requires_firmware_escalation_policy_review"
                else "security_triage_or_escalation_after_filter"
            )
            fixture["blocked_reason"] = gate_row["gate_decision"]
            safety_fixtures.append(fixture)
        else:
            review = {
                "record_id": base["record_id"],
                "title": base["title"],
                "source_url": base["source_url"],
                "site": base["site"],
                "primary_domain": base["primary_domain"],
                "secondary_domains": base["secondary_domains"],
                "safety_flags": base["safety_flags"],
                "gate_decision": base["gate_decision"],
                "manual_review_reason": "record_level_review_required_before_promotion",
                "commercial_reuse_allowed": False,
            }
            manual_review.append(review)

    return {
        "classifier_pool": classifier_pool,
        "retrieval_candidates": retrieval_candidates,
        "safety_fixtures": safety_fixtures,
        "manual_review": manual_review,
    }


def build_ticket_review_queue() -> list[dict[str, Any]]:
    intake_rows = read_csv(GATE / "ticket_dataset_intake.csv")
    profile_rows = read_csv(GATE / "ticket_dataset_file_profiles.csv")
    preview_rows = read_jsonl(GATE / "ticket_dataset_redacted_preview.jsonl")
    profiles_by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    previews_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in profile_rows:
        profiles_by_dataset[row["dataset_id"]].append(row)
    for row in preview_rows:
        previews_by_dataset[row["dataset_id"]].append(row)

    queue = []
    for row in intake_rows:
        dataset_id = row["dataset_id"]
        profiles = profiles_by_dataset.get(dataset_id, [])
        queue.append(
            {
                "dataset_id": dataset_id,
                "dataset_dir": row.get("dataset_dir"),
                "declared_license": row.get("declared_license"),
                "declared_pii_removed": compact_bool(row.get("declared_pii_removed")),
                "declared_synthetic": compact_bool(row.get("declared_synthetic")),
                "text_file_count": int(row.get("text_file_count") or 0),
                "label_file_count": int(row.get("label_file_count") or 0),
                "pii_signal_count": int(row.get("pii_signal_count") or 0),
                "secret_like_signal_count": int(row.get("secret_like_signal_count") or 0),
                "provisional_use_lanes": [lane for lane in row.get("provisional_use_lanes", "").split(";") if lane],
                "promotion_status": "blocked_pending_manual_pii_license_provenance_schema_review",
                "downstream_allowed": {
                    "classifier_pool": False,
                    "retrieval": False,
                    "answer_generation": False,
                    "fine_tuning": False,
                    "evaluation": False,
                    "commercial_mode": False,
                },
                "file_profiles": [
                    {
                        "path": profile.get("path"),
                        "read_status": profile.get("read_status"),
                        "row_count": profile.get("row_count"),
                        "text_columns": [item for item in profile.get("text_columns", "").split(";") if item],
                        "label_columns": [item for item in profile.get("label_columns", "").split(";") if item],
                        "pii_signal_count": int(profile.get("pii_signal_count") or 0),
                        "secret_like_signal_count": int(profile.get("secret_like_signal_count") or 0),
                    }
                    for profile in profiles
                ],
                "redacted_previews": previews_by_dataset.get(dataset_id, [])[:5],
                "next_review_questions": [
                    "Is the license compatible with intended POC use?",
                    "Is the source real, synthetic, anonymized, or mixed?",
                    "Are PII and secret-like signals false positives or real residual sensitive data?",
                    "Which text and label columns map cleanly to IT-SUPPORT domains?",
                    "Should this dataset be classifier-only, eval-only, retrieval-safe, or excluded?",
                ],
            }
        )
    return queue


def domain_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(row.get("primary_domain", "") for row in rows)
    return [{"primary_domain": key or "unknown", "records": value} for key, value in counts.most_common()]


def write_schema() -> None:
    schema = {
        "schema_version": "normalized_candidate_sets_v1",
        "status": "poc_candidate_not_training_or_indexing_artifact",
        "record_types": {
            "stack_exchange_classifier_pool": "Question-only classification/evaluation pool; includes all gated Stack Exchange records with promotion status.",
            "stack_exchange_retrieval_candidates": "Question plus accepted/top-answer evidence for records that passed candidate_after_license_review.",
            "stack_exchange_safety_eval_fixtures": "Question-only security/firmware fixtures. No answer text included.",
            "ticket_dataset_manual_review_queue": "Dataset/file-level review queue. No training promotion yet.",
        },
        "commercial_mode": "All Stack Exchange and non-commercial ticket sources remain excluded from commercial-mode runs.",
    }
    write_json(OUT / "normalized_candidate_schema_v1.json", schema)


def write_summary(stack_sets: dict[str, list[dict[str, Any]]], ticket_queue: list[dict[str, Any]]) -> None:
    outputs = [
        OUT / "normalization_summary.md",
        OUT / "normalization_summary.json",
        OUT / "normalized_candidate_schema_v1.json",
        OUT / "stack_exchange_classifier_pool.jsonl",
        OUT / "stack_exchange_retrieval_candidates.jsonl",
        OUT / "stack_exchange_safety_eval_fixtures.jsonl",
        OUT / "stack_exchange_manual_review_queue.jsonl",
        OUT / "ticket_dataset_manual_review_queue.jsonl",
        OUT / "stack_exchange_candidate_domain_counts.csv",
    ]
    domain_rows = domain_counts(stack_sets["classifier_pool"])
    write_csv(OUT / "stack_exchange_candidate_domain_counts.csv", domain_rows)

    summary = {
        "stage": "normalized_candidate_sets",
        "policy": "Promote clean POC candidates only; preserve security/firmware/ticket signal as eval fixtures or review queues.",
        "counts": {
            "stack_exchange_classifier_pool": len(stack_sets["classifier_pool"]),
            "stack_exchange_retrieval_candidates": len(stack_sets["retrieval_candidates"]),
            "stack_exchange_safety_eval_fixtures": len(stack_sets["safety_fixtures"]),
            "stack_exchange_manual_review_queue": len(stack_sets["manual_review"]),
            "ticket_dataset_manual_review_queue": len(ticket_queue),
        },
        "retrieval_candidate_gate_decisions": dict(Counter(row["gate_decision"] for row in stack_sets["retrieval_candidates"])),
        "safety_fixture_expected_behavior": dict(Counter(row["expected_behavior"] for row in stack_sets["safety_fixtures"])),
        "domain_counts": domain_rows,
        "outputs": [rel(path) for path in outputs],
    }
    write_json(OUT / "normalization_summary.json", summary)

    count_rows = [{"artifact": key, "records": value} for key, value in summary["counts"].items()]
    lines = [
        "# Normalized Candidate Sets",
        "",
        "This stage creates POC candidate and review artifacts from the preprocessing gate. It does not train, index, or enable answer generation.",
        "",
        "## Counts",
        "",
        md_table(count_rows, ["artifact", "records"]),
        "",
        "## Primary Domain Counts",
        "",
        md_table(domain_rows, ["primary_domain", "records"]),
        "",
        "## Outputs",
        "",
        *[f"- `{rel(path)}`" for path in outputs[1:]],
        "",
        "## Downstream Rule",
        "",
        "- `stack_exchange_retrieval_candidates.jsonl` is POC/share-alike retrieval candidate material only; it is not commercial-ready and not answer-generation-enabled yet.",
        "- `stack_exchange_safety_eval_fixtures.jsonl` is question-only safety/eval material; no security or firmware answer text is promoted.",
        "- `ticket_dataset_manual_review_queue.jsonl` remains blocked from training until manual dataset-level review promotes a subset.",
    ]
    (OUT / "normalization_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-run", default="20260506T145259Z")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stack_sets = build_stack_sets(args.stack_run)
    ticket_queue = build_ticket_review_queue()
    write_jsonl(OUT / "stack_exchange_classifier_pool.jsonl", stack_sets["classifier_pool"])
    write_jsonl(OUT / "stack_exchange_retrieval_candidates.jsonl", stack_sets["retrieval_candidates"])
    write_jsonl(OUT / "stack_exchange_safety_eval_fixtures.jsonl", stack_sets["safety_fixtures"])
    write_jsonl(OUT / "stack_exchange_manual_review_queue.jsonl", stack_sets["manual_review"])
    write_jsonl(OUT / "ticket_dataset_manual_review_queue.jsonl", ticket_queue)
    write_schema()
    write_summary(stack_sets, ticket_queue)
    print(f"wrote {rel(OUT / 'normalization_summary.md')}")
    print(f"retrieval_candidates={len(stack_sets['retrieval_candidates'])}")
    print(f"safety_fixtures={len(stack_sets['safety_fixtures'])}")
    print(f"ticket_review_datasets={len(ticket_queue)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
