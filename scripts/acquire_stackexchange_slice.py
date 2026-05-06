"""Acquire a small Stack Exchange IT-support Q&A slice.

This is deliberately source-specific. Do not generalize it until a second source
forces shared acquisition logic.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = "stack_exchange_it_support_sites"
API_ROOT = "https://api.stackexchange.com/2.3"
CREATED_CC_BY_SA_30 = 1302220800  # 2011-04-08 UTC
CREATED_CC_BY_SA_40 = 1525219200  # 2018-05-02 UTC
SITE_HOSTS = {
    "askubuntu": "askubuntu.com",
    "dba": "dba.stackexchange.com",
    "networkengineering": "networkengineering.stackexchange.com",
    "security": "security.stackexchange.com",
    "serverfault": "serverfault.com",
    "superuser": "superuser.com",
    "unix": "unix.stackexchange.com",
}

QUERY_SPECS = [
    {"site": "superuser", "tag": "windows-10", "domains": ["os_kernel_drivers", "application_software"]},
    {"site": "superuser", "tag": "windows-11", "domains": ["os_kernel_drivers", "application_software"]},
    {"site": "superuser", "tag": "networking", "domains": ["network_connectivity"]},
    {"site": "superuser", "tag": "vpn", "domains": ["network_connectivity", "identity_access_accounts"]},
    {"site": "superuser", "tag": "hard-drive", "domains": ["hardware", "storage_data_backup"]},
    {"site": "superuser", "tag": "backup", "domains": ["storage_data_backup"]},
    {"site": "superuser", "tag": "printer", "domains": ["hardware", "os_kernel_drivers"]},
    {"site": "superuser", "tag": "bios", "domains": ["hardware", "firmware_bios_uefi"]},
    {"site": "superuser", "tag": "uefi", "domains": ["hardware", "firmware_bios_uefi"]},
    {"site": "askubuntu", "tag": "drivers", "domains": ["os_kernel_drivers", "hardware"]},
    {"site": "askubuntu", "tag": "networking", "domains": ["network_connectivity"]},
    {"site": "askubuntu", "tag": "wireless", "domains": ["network_connectivity", "os_kernel_drivers"]},
    {"site": "askubuntu", "tag": "nvidia", "domains": ["hardware", "os_kernel_drivers"]},
    {"site": "askubuntu", "tag": "grub2", "domains": ["os_kernel_drivers", "storage_data_backup"]},
    {"site": "serverfault", "tag": "active-directory", "domains": ["identity_access_accounts"]},
    {"site": "serverfault", "tag": "group-policy", "domains": ["identity_access_accounts", "application_software"]},
    {"site": "serverfault", "tag": "domain-name-system", "domains": ["network_connectivity"]},
    {"site": "serverfault", "tag": "windows-server-2019", "domains": ["os_kernel_drivers", "identity_access_accounts"]},
    {"site": "serverfault", "tag": "vpn", "domains": ["network_connectivity", "identity_access_accounts"]},
    {"site": "networkengineering", "tag": "routing", "domains": ["network_connectivity"]},
    {"site": "networkengineering", "tag": "dns", "domains": ["network_connectivity"]},
    {"site": "unix", "tag": "permissions", "domains": ["identity_access_accounts", "os_kernel_drivers"]},
    {"site": "unix", "tag": "systemd", "domains": ["os_kernel_drivers", "application_software"]},
    {"site": "unix", "tag": "mount", "domains": ["storage_data_backup", "os_kernel_drivers"]},
    {"site": "unix", "tag": "disk", "domains": ["storage_data_backup", "hardware"]},
    {"site": "security", "tag": "malware", "domains": ["security_malware"]},
    {"site": "security", "tag": "phishing", "domains": ["security_malware", "identity_access_accounts"]},
    {"site": "security", "tag": "ransomware", "domains": ["security_malware", "storage_data_backup"]},
    {"site": "dba", "tag": "backup", "domains": ["storage_data_backup", "application_software"]},
]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def request_json(session: requests.Session, path: str, params: dict[str, Any]) -> dict[str, Any]:
    response = session.get(f"{API_ROOT}{path}", params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get("backoff"):
        time.sleep(int(payload["backoff"]) + 1)
    return payload


def license_for(created: int | None) -> str:
    if created is None:
        return "CC BY-SA unknown; creation_date missing"
    if created < CREATED_CC_BY_SA_30:
        return "CC BY-SA 2.5"
    if created < CREATED_CC_BY_SA_40:
        return "CC BY-SA 3.0"
    return "CC BY-SA 4.0"


def owner_meta(item: dict[str, Any]) -> dict[str, Any]:
    owner = item.get("owner") or {}
    return {
        "owner_user_id": owner.get("user_id"),
        "owner_display_name": owner.get("display_name"),
        "owner_link": owner.get("link"),
        "owner_user_type": owner.get("user_type"),
    }


def attribution_row(
    *,
    record_type: str,
    site: str,
    item: dict[str, Any],
    source_family: dict[str, Any],
    domains: list[str],
) -> dict[str, Any]:
    created = item.get("creation_date")
    derived_license = license_for(created)
    api_license = item.get("content_license")
    source_url = item.get("link")
    if not source_url and record_type == "answer" and item.get("answer_id"):
        source_url = f"https://{SITE_HOSTS[site]}/a/{item['answer_id']}"
    row = {
        "source_family_id": SOURCE_ID,
        "record_type": record_type,
        "site": site,
        "question_id": item.get("question_id"),
        "answer_id": item.get("answer_id"),
        "title": item.get("title"),
        "source_url": source_url,
        "score": item.get("score"),
        "creation_date": created,
        "license": api_license or derived_license,
        "api_content_license": api_license,
        "derived_license_by_creation_date": derived_license,
        "commercial_posture": source_family["commercial_posture"],
        "attribution_required": source_family["attribution_required"],
        "share_alike_required": source_family["share_alike_required"],
        "tags": item.get("tags", []),
        "target_domains": domains,
    }
    row.update(owner_meta(item))
    return row


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunks(items: list[str], size: int) -> list[list[str]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def update_registry(registry_path: Path, run_entry: dict[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        return
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry.setdefault("acquisition_runs", []).append(run_entry)
    write_json(registry_path, registry)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pagesize", type=int, default=3, help="Questions per site/tag query.")
    parser.add_argument("--answer-pages", type=int, default=2, help="Answer pages per 100-question chunk.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    registry_path = PROJECT_ROOT / "data" / "source_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    source_family = next(s for s in registry["source_families"] if s["id"] == SOURCE_ID)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    raw_dir = PROJECT_ROOT / "data" / "raw" / SOURCE_ID / run_id
    attribution_path = PROJECT_ROOT / "data" / "attribution" / SOURCE_ID / f"{run_id}_attribution.jsonl"

    print(f"run_id={run_id}")
    print(f"source={SOURCE_ID}")
    print(f"queries={len(QUERY_SPECS)} pagesize={args.pagesize} dry_run={args.dry_run}")

    session = requests.Session()
    key = os.getenv("STACK_EXCHANGE_API_KEY")
    all_questions: dict[str, list[dict[str, Any]]] = {}
    question_domains: dict[int, list[str]] = {}
    question_meta: dict[int, dict[str, Any]] = {}
    raw_question_files: list[str] = []
    raw_answer_files: list[str] = []
    attribution_rows: list[dict[str, Any]] = []
    seen_questions: set[tuple[str, int]] = set()
    seen_answers: set[tuple[str, int]] = set()

    if args.dry_run:
        print(json.dumps(QUERY_SPECS, indent=2))
        return 0

    show_progress = not args.no_progress

    for spec in tqdm(QUERY_SPECS, desc="Stack Exchange question queries", unit="query", disable=not show_progress):
        params = {
            "site": spec["site"],
            "tagged": spec["tag"],
            "accepted": "true",
            "answers": 1,
            "sort": "votes",
            "order": "desc",
            "pagesize": args.pagesize,
            "filter": "withbody",
        }
        if key:
            params["key"] = key
        payload = request_json(session, "/search/advanced", params)
        filename = f"questions_{spec['site']}_{spec['tag']}.json".replace("/", "_")
        write_json(raw_dir / filename, {"query": spec, "request_params": params, "response": payload})
        raw_question_files.append(filename)
        questions = payload.get("items", [])
        all_questions.setdefault(spec["site"], []).extend(questions)
        for question in questions:
            if question.get("question_id"):
                question_id = int(question["question_id"])
                question_domains[question_id] = spec["domains"]
                question_meta[question_id] = question
                question_key = (spec["site"], question_id)
                if question_key not in seen_questions:
                    seen_questions.add(question_key)
                    attribution_rows.append(
                        attribution_row(
                            record_type="question",
                            site=spec["site"],
                            item=question,
                            source_family=source_family,
                            domains=spec["domains"],
                        )
                    )
            else:
                attribution_rows.append(
                    attribution_row(
                        record_type="question",
                        site=spec["site"],
                        item=question,
                        source_family=source_family,
                        domains=spec["domains"],
                    )
                )
        tqdm.write(f"{spec['site']} [{spec['tag']}]: questions={len(questions)}")

    for site, questions in tqdm(all_questions.items(), desc="Stack Exchange answer batches", unit="site", disable=not show_progress):
        ids = sorted({str(q["question_id"]) for q in questions if q.get("question_id")})
        if not ids:
            continue
        site_answer_count = 0
        for chunk_index, id_chunk in enumerate(chunks(ids, 100), start=1):
            for page in range(1, args.answer_pages + 1):
                params = {
                    "site": site,
                    "sort": "votes",
                    "order": "desc",
                    "page": page,
                    "pagesize": 100,
                    "filter": "withbody",
                }
                if key:
                    params["key"] = key
                payload = request_json(session, f"/questions/{';'.join(id_chunk)}/answers", params)
                filename = f"answers_{site}_chunk{chunk_index:02d}_page{page:02d}.json"
                write_json(raw_dir / filename, {"question_ids": id_chunk, "request_params": params, "response": payload})
                raw_answer_files.append(filename)
                for answer in payload.get("items", []):
                    if not answer.get("answer_id"):
                        continue
                    answer_key = (site, int(answer["answer_id"]))
                    if answer_key in seen_answers:
                        continue
                    seen_answers.add(answer_key)
                    question_id = int(answer.get("question_id", 0))
                    parent = question_meta.get(question_id, {})
                    answer_for_attribution = dict(answer)
                    answer_for_attribution.setdefault("title", parent.get("title"))
                    answer_for_attribution.setdefault("tags", parent.get("tags", []))
                    domains = question_domains.get(question_id, [])
                    row = attribution_row(
                        record_type="answer",
                        site=site,
                        item=answer_for_attribution,
                        source_family=source_family,
                        domains=domains,
                    )
                    row["parent_question_url"] = parent.get("link")
                    attribution_rows.append(row)
                    site_answer_count += 1
                if not payload.get("has_more"):
                    break
        tqdm.write(f"{site}: answers={site_answer_count}")

    write_jsonl(attribution_path, attribution_rows)

    manifest = {
        "run_id": run_id,
        "source_family_id": SOURCE_ID,
        "started_at_utc": run_id,
        "api_root": API_ROOT,
        "query_specs": QUERY_SPECS,
        "raw_dir": str(raw_dir.relative_to(PROJECT_ROOT)),
        "attribution_path": str(attribution_path.relative_to(PROJECT_ROOT)),
        "raw_question_files": raw_question_files,
        "raw_answer_files": raw_answer_files,
        "question_count": sum(len(items) for items in all_questions.values()),
        "attribution_record_count": len(attribution_rows),
        "license_note": source_family["license"],
        "commercial_posture": source_family["commercial_posture"],
    }
    write_json(raw_dir / "run_manifest.json", manifest)
    update_registry(
        registry_path,
        {
            "run_id": run_id,
            "source_family_id": SOURCE_ID,
            "status": "raw_acquired",
            "raw_dir": manifest["raw_dir"],
            "attribution_path": manifest["attribution_path"],
            "question_count": manifest["question_count"],
            "attribution_record_count": manifest["attribution_record_count"],
        },
        dry_run=args.dry_run,
    )
    print(f"raw_dir={manifest['raw_dir']}")
    print(f"attribution={manifest['attribution_path']}")
    print(f"questions={manifest['question_count']} attribution_rows={manifest['attribution_record_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
