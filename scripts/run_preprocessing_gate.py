"""Create source-specific preprocessing gate artifacts.

This is an offline forensic gate. It does not train models, build indexes, or
prepare answer-generation data. It assigns use lanes so we can maximize the
current corpus without mixing unsafe, unreviewed, or commercial-unsafe records
into downstream stages.
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
OUT = DATA / "processed" / "source_specific_preprocessing_gate"

DOMAINS = [
    "hardware",
    "os_kernel_drivers",
    "application_software",
    "network_connectivity",
    "identity_access_accounts",
    "storage_data_backup",
    "security_malware",
    "firmware_bios_uefi",
]

USE_LANES = {
    "classifier_candidate": "Routing/classification signal after required review.",
    "retrieval_candidate_after_license_review": "Potential retrieval evidence after license/path review.",
    "retrieval_candidate_after_safety_filter": "Potential retrieval evidence after record-level safety filtering.",
    "evaluation_candidate": "Useful for held-out routing, retrieval, or safety evaluation.",
    "security_filter_required": "Security content requires defensive/offensive review.",
    "security_eval_candidate": "Useful for security recall and escalation tests.",
    "firmware_escalation_eval_candidate": "Useful for BIOS/UEFI/firmware escalation tests.",
    "blocked_from_generation_pending_review": "Do not feed into answer generation yet.",
    "blocked_from_training_pending_review": "Do not use for fine-tuning yet.",
    "label_mapping_candidate": "Has labels/categories that may map into IT-SUPPORT domains.",
    "needs_redaction": "Sample scan found possible PII or secret-like content.",
    "pii_reduction_claimed": "Metadata claims anonymization, PII reduction, or synthetic generation.",
    "poc_only_noncommercial": "POC-only non-commercial source.",
    "poc_only_share_alike": "POC/share-alike constrained source.",
    "commercial_candidate_after_provenance_review": "License may permit commercial use after provenance review.",
    "commercial_excluded": "Exclude from commercial-mode data unless approved.",
    "license_review_required": "License/provenance still needs review.",
}

DOMAIN_TERMS = {
    "hardware": "battery bios camera charger display fan gpu hard-drive keyboard laptop motherboard printer ram screen ssd thermal touchpad usb".split(),
    "os_kernel_drivers": "bsod boot device-manager driver grub kernel linux reboot registry service systemd update windows".split(),
    "application_software": "app browser chrome excel firefox office outlook package powershell software teams word".split(),
    "network_connectivity": "bind dhcp dns firewall gateway latency network ping port router routing tcp vpn wifi wi-fi wireless".split(),
    "identity_access_accounts": "active-directory admin authentication group-policy login mfa password permission sso account".split(),
    "storage_data_backup": "backup data-loss disk filesystem fsck hard-drive mount partition raid recover restore smart storage".split(),
    "security_malware": "backdoor breach cve exploit hacked malware phishing ransomware security suspicious virus vulnerability".split(),
    "firmware_bios_uefi": ["bios", "bootloader", "efi", "firmware", "firmware-update", "flash-bios", "flashing-bios", "uefi"],
}

QUESTION_VISIBLE_SECURITY_EVAL_TERMS = [
    "backdoor",
    "badusb",
    "breach",
    "cache-poisoning",
    "compromise",
    "compromised",
    "credential-theft",
    "cve",
    "exploit",
    "hacked",
    "keylogger",
    "malicious",
    "malware",
    "phishing",
    "ransomware",
    "scam",
    "spoofing",
    "spyware",
    "suspicious",
    "unauthorized",
    "virus",
    "vulnerability",
]
QUESTION_VISIBLE_FIRMWARE_EVAL_TERMS = [
    "bios",
    "bios-update",
    "bricked",
    "cmos",
    "efi",
    "firmware",
    "firmware-update",
    "flash-bios",
    "flashing-bios",
    "nvram",
    "secure-boot",
    "uefi",
]

TAG_HINTS = {
    "active-directory": "identity_access_accounts",
    "backup": "storage_data_backup",
    "bios": "firmware_bios_uefi",
    "domain-name-system": "network_connectivity",
    "drivers": "os_kernel_drivers",
    "grub2": "os_kernel_drivers",
    "hard-drive": "storage_data_backup",
    "malware": "security_malware",
    "networking": "network_connectivity",
    "nvidia": "hardware",
    "permissions": "identity_access_accounts",
    "phishing": "security_malware",
    "printer": "hardware",
    "ransomware": "security_malware",
    "routing": "network_connectivity",
    "systemd": "os_kernel_drivers",
    "uefi": "firmware_bios_uefi",
    "vpn": "network_connectivity",
    "wireless": "network_connectivity",
}

TEXT_HINTS = "body comment conversation description email instruction issue message output question reasoning request resolution response steps summary text ticket title".split()
LABEL_HINTS = "affected_service category class department escalated intent label priority queue service status type y".split()

PII_PATTERNS = {
    "email_like": re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    "phone_like": re.compile(r"(?:\+?\d[\d .()/-]{7,}\d)"),
    "ipv4_like": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}
SECRET_PATTERNS = {
    "api_key_like": re.compile(r"\b(?:api[_-]?key|token|secret|access[_-]?key)\b\s*[:=]", re.I),
    "private_key_like": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "aws_key_like": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "github_token_like": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_text(path: Path, limit: int | None = None) -> str:
    data = path.read_bytes()
    if limit is not None:
        data = data[:limit]
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


def compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def strip_html(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"<(br|p|li|pre)[^>]*>", " ", value, flags=re.I)
    value = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", html.unescape(value)).strip()


def snippet(value: str, limit: int = 280) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    value = PII_PATTERNS["email_like"].sub("[EMAIL]", value)
    value = PII_PATTERNS["phone_like"].sub("[PHONE]", value)
    value = PII_PATTERNS["ipv4_like"].sub("[IP]", value)
    for pattern in SECRET_PATTERNS.values():
        value = pattern.sub("[SECRET-LIKE]", value)
    return value[: limit - 3] + "..." if len(value) > limit else value


def quality_matrix() -> dict[str, dict[str, Any]]:
    rows = read_json(DATA / "processed" / "source_inventory_audit" / "source_quality_matrix.json")
    return {row["source_family_id"]: row for row in rows}


def pattern_counts(text: str, patterns: dict[str, re.Pattern[str]]) -> dict[str, int]:
    return {name: len(pattern.findall(text)) for name, pattern in patterns.items()}


def has_term(text: str, term: str) -> bool:
    pattern = re.escape(term).replace(r"\-", r"[-\s]+")
    return bool(re.search(rf"(?<![a-z0-9]){pattern}(?![a-z0-9])", text, flags=re.I))


def term_hits(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if has_term(text, term)]


def safety_flags(text: str, domains: list[str], tags: list[str]) -> dict[str, Any]:
    lower = text.lower()
    tag_text = " ".join(tags).lower()
    flags = {
        "possible_security_incident": "security_malware" in domains or bool(term_hits(lower, DOMAIN_TERMS["security_malware"]) or term_hits(tag_text, DOMAIN_TERMS["security_malware"])),
        "firmware_escalation_required": "firmware_bios_uefi" in domains or bool(term_hits(lower, DOMAIN_TERMS["firmware_bios_uefi"]) or term_hits(tag_text, DOMAIN_TERMS["firmware_bios_uefi"])),
        "data_loss_risk": bool(term_hits(lower, ["data-loss", "format", "fsck", "partition", "recover", "restore", "smart", "wipe"])),
        "credential_secret_risk": bool(term_hits(lower, ["credential", "mfa", "password", "private-key", "secret", "token"])),
        "offensive_security_review_required": bool(term_hits(lower, ["bypass", "credential-theft", "exploit", "keylogger", "payload", "persistence", "reverse-shell", "shellcode"])),
        "destructive_operation_review_required": bool(term_hits(lower, ["delete-system32", "format-c", "rm-rf", "wipe-disk"])),
    }
    pii = pattern_counts(text, PII_PATTERNS)
    secrets = pattern_counts(text, SECRET_PATTERNS)
    flags["pii_signal_count"] = sum(pii.values())
    flags["secret_like_signal_count"] = sum(secrets.values())
    flags["needs_human_review"] = any(v for k, v in flags.items() if k.endswith("_required") or k.endswith("_risk") or k == "possible_security_incident") or flags["pii_signal_count"] > 0 or flags["secret_like_signal_count"] > 0
    return flags


def question_visible_safety_eval(text: str, tags: list[str]) -> dict[str, Any]:
    """Question-only safety labels for eval fixtures.

    The source gate can inspect answer evidence conservatively before promoting
    records downstream. Question-only safety evaluation should not expect the
    baseline to recover cues that only appeared in accepted-answer text or broad
    acquisition context.
    """

    lower = text.lower()
    tag_text = " ".join(tags).lower()
    security_hits = sorted(
        set(
            term_hits(lower, QUESTION_VISIBLE_SECURITY_EVAL_TERMS)
            + term_hits(tag_text, QUESTION_VISIBLE_SECURITY_EVAL_TERMS)
        )
    )
    firmware_hits = sorted(
        set(
            term_hits(lower, QUESTION_VISIBLE_FIRMWARE_EVAL_TERMS)
            + term_hits(tag_text, QUESTION_VISIBLE_FIRMWARE_EVAL_TERMS)
        )
    )
    expected_behavior = ""
    if firmware_hits:
        expected_behavior = "structured_firmware_escalation"
    elif security_hits:
        expected_behavior = "security_triage_or_escalation_after_filter"

    return {
        "possible_security_incident": bool(security_hits),
        "firmware_escalation_required": bool(firmware_hits),
        "security_hits": security_hits,
        "firmware_hits": firmware_hits,
        "expected_behavior": expected_behavior,
    }


def rank_domains(text: str, query_domains: list[str], tags: list[str]) -> list[dict[str, Any]]:
    lower = text.lower()
    scores: Counter[str] = Counter()
    evidence: dict[str, list[str]] = defaultdict(list)
    for domain in query_domains:
        scores[domain] += 4
        evidence[domain].append("acquisition_query")
    for tag in tags:
        hinted = TAG_HINTS.get(tag.lower())
        if hinted:
            scores[hinted] += 3
            evidence[hinted].append(f"tag:{tag}")
    for domain, terms in DOMAIN_TERMS.items():
        hits = term_hits(lower, terms)
        if hits:
            scores[domain] += min(len(hits), 5)
            evidence[domain].append("keywords:" + ",".join(hits[:5]))
    return [
        {"label": domain, "score": score, "evidence": evidence[domain][:4]}
        for domain, score in scores.most_common()
        if domain in DOMAINS
    ]


def stack_site_from_answer_file(name: str) -> str:
    match = re.match(r"answers_(.+?)_chunk", name)
    if not match:
        raise ValueError(f"Cannot parse site from {name}")
    return match.group(1)


def stack_exchange_gate(run_id: str, quality: dict[str, dict[str, Any]]) -> dict[str, Any]:
    source_id = "stack_exchange_it_support_sites"
    source_quality = quality[source_id]
    raw_dir = DATA / "raw" / source_id / run_id
    manifest = read_json(raw_dir / "run_manifest.json")
    attr_rows = read_jsonl(ROOT / manifest["attribution_path"])
    attr = defaultdict(list)
    for row in attr_rows:
        if row.get("site") and row.get("question_id") is not None:
            answer_id = row.get("answer_id")
            key = (row.get("record_type"), row["site"], int(row["question_id"]), int(answer_id) if answer_id else None)
            attr[key].append(row)

    questions: dict[tuple[str, int], dict[str, Any]] = {}
    answers: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for filename in manifest["raw_question_files"]:
        payload = read_json(raw_dir / filename)
        query = payload["query"]
        site, qtag, qdomains = query["site"], query["tag"], query["domains"]
        for item in payload["response"].get("items", []):
            if not item.get("question_id"):
                continue
            key = (site, int(item["question_id"]))
            record = questions.setdefault(
                key,
                {
                    "site": site,
                    "question_id": int(item["question_id"]),
                    "title": item.get("title", ""),
                    "question_text": strip_html(item.get("body")),
                    "question_tags": item.get("tags", []),
                    "query_tags": set(),
                    "query_domains": set(),
                    "score": item.get("score"),
                    "url": item.get("link"),
                    "accepted_answer_id": item.get("accepted_answer_id"),
                    "license": item.get("content_license") or source_quality["license_posture"],
                },
            )
            record["query_tags"].add(qtag)
            record["query_domains"].update(qdomains)
    for filename in manifest["raw_answer_files"]:
        site = stack_site_from_answer_file(filename)
        payload = read_json(raw_dir / filename)
        for item in payload["response"].get("items", []):
            if item.get("question_id") and item.get("answer_id"):
                item = dict(item)
                item["answer_text"] = strip_html(item.get("body"))
                key = (site, int(item["question_id"]))
                if all(existing.get("answer_id") != item["answer_id"] for existing in answers[key]):
                    answers[key].append(item)

    rows, previews = [], []
    domain_counts: dict[str, Counter[str]] = {domain: Counter() for domain in DOMAINS}
    for (site, qid), question in sorted(questions.items()):
        answer_list = sorted(answers.get((site, qid), []), key=lambda row: row.get("score") or 0, reverse=True)
        accepted_id = question.get("accepted_answer_id")
        accepted = next((row for row in answer_list if row.get("answer_id") == accepted_id), None)
        top = answer_list[0] if answer_list else None
        qdomains = sorted(question["query_domains"])
        qtags = sorted(question["question_tags"])
        query_tags = sorted(question["query_tags"])
        question_visible_text = " ".join(
            [question["title"], question["question_text"], " ".join(qtags + query_tags)]
        )
        signal_text = " ".join(
            [
                question_visible_text,
                accepted.get("answer_text", "") if accepted else "",
            ]
        )
        ranked = rank_domains(signal_text, qdomains, qtags)
        flags = safety_flags(signal_text, qdomains, qtags)
        eval_flags = question_visible_safety_eval(question_visible_text, qtags + query_tags)
        lanes = {"classifier_candidate", "evaluation_candidate", "poc_only_share_alike", "commercial_excluded"}
        if answer_list:
            lanes.add("retrieval_candidate_after_safety_filter")
        if not flags["needs_human_review"]:
            lanes.add("retrieval_candidate_after_license_review")
        if flags["possible_security_incident"]:
            lanes.update({"security_filter_required", "security_eval_candidate", "blocked_from_generation_pending_review"})
        if flags["firmware_escalation_required"]:
            lanes.update({"firmware_escalation_eval_candidate", "blocked_from_generation_pending_review"})
        if eval_flags["possible_security_incident"]:
            lanes.add("question_visible_security_eval_candidate")
        if eval_flags["firmware_escalation_required"]:
            lanes.add("question_visible_firmware_eval_candidate")
        if flags["data_loss_risk"] or flags["destructive_operation_review_required"]:
            lanes.add("blocked_from_generation_pending_review")
        gate = "candidate_after_license_review"
        if flags["possible_security_incident"] or flags["offensive_security_review_required"]:
            gate = "requires_record_level_security_filter"
        elif flags["firmware_escalation_required"]:
            gate = "requires_firmware_escalation_policy_review"
        elif flags["needs_human_review"]:
            gate = "requires_record_level_review"
        primary = ranked[0]["label"] if ranked else (qdomains[0] if qdomains else "")
        secondary = [item["label"] for item in ranked[1:4]]
        row = {
            "record_id": f"{source_id}:{site}:{qid}",
            "source_family_id": source_id,
            "run_id": run_id,
            "site": site,
            "question_id": qid,
            "title": question["title"],
            "query_tags": ";".join(sorted(question["query_tags"])),
            "question_tags": ";".join(qtags),
            "query_domains": ";".join(qdomains),
            "primary_domain": primary,
            "secondary_domains": ";".join(secondary),
            "ranked_domains_json": compact(ranked),
            "safety_flags_json": compact(flags),
            "eval_safety_flags_json": compact(eval_flags),
            "safety_eval_expected_behavior": eval_flags["expected_behavior"],
            "use_lanes": ";".join(sorted(lanes)),
            "gate_decision": gate,
            "accepted_answer_id": accepted_id or "",
            "accepted_answer_captured": bool(accepted),
            "accepted_is_top": bool(accepted and top and accepted.get("answer_id") == top.get("answer_id")),
            "captured_answer_count": len(answer_list),
            "license": question["license"],
            "commercial_posture": source_quality["commercial_posture"],
            "source_url": question["url"],
        }
        rows.append(row)
        for domain in set([primary, *secondary, *qdomains]):
            if domain in domain_counts:
                domain_counts[domain]["records_with_domain_signal"] += 1
                domain_counts[domain]["primary_rank_count"] += int(domain == primary)
                domain_counts[domain]["acquisition_label_count"] += int(domain in qdomains)
                domain_counts[domain]["human_review_signal_count"] += int(flags["needs_human_review"])
        if len(previews) < 120:
            refs = []
            refs.extend(attr.get(("question", site, qid, None), [])[:1])
            if accepted_id:
                refs.extend(attr.get(("answer", site, qid, int(accepted_id)), [])[:1])
            previews.append(
                {
                    "record_id": row["record_id"],
                    "title": question["title"],
                    "question_snippet": snippet(question["question_text"]),
                    "accepted_answer_snippet": snippet(accepted.get("answer_text", "")) if accepted else "",
                    "ranked_domains": ranked[:4],
                    "safety_flags": flags,
                    "use_lanes": sorted(lanes),
                    "gate_decision": gate,
                    "attribution_refs": [
                        {
                            "record_type": item.get("record_type"),
                            "source_url": item.get("source_url"),
                            "owner_display_name": item.get("owner_display_name"),
                            "license": item.get("license"),
                        }
                        for item in refs
                    ],
                }
            )
    domain_rows = [{"domain": domain, **dict(counts)} for domain, counts in sorted(domain_counts.items())]
    write_csv(OUT / "stack_exchange_scaled_run_gate.csv", rows)
    write_csv(OUT / "stack_exchange_domain_signal_matrix.csv", domain_rows)
    write_jsonl(OUT / "stack_exchange_gated_preview.jsonl", previews)
    return {
        "run_id": run_id,
        "manifest_question_hits": manifest.get("question_count"),
        "records": len(rows),
        "duplicate_question_hits_collapsed": int(manifest.get("question_count") or len(rows)) - len(rows),
        "preview_records": len(previews),
        "gate_counts": dict(Counter(row["gate_decision"] for row in rows)),
        "use_lane_counts": dict(Counter(lane for row in rows for lane in row["use_lanes"].split(";"))),
    }


def dataset_license(dataset_dir: Path) -> tuple[str, bool, bool, str]:
    text, license_name, note = "", "unknown", ""
    readme = dataset_dir / "README.md"
    metadata_file = dataset_dir / "record_metadata.json"
    if readme.exists():
        text = read_text(readme, 200_000)
        match = re.search(r"(?im)^license:\s*([^\n]+)", text)
        if match:
            license_name = match.group(1).strip().strip("'\"")
        note = snippet(re.sub(r"\s+", " ", text), 420)
    if metadata_file.exists():
        meta = read_json(metadata_file)
        license_name = meta.get("metadata", {}).get("license", {}).get("id", license_name)
        desc = strip_html(meta.get("metadata", {}).get("description", ""))
        text += " " + desc
        note = snippet(desc, 420) or note
    lower = text.lower()
    pii_removed = any(term in lower for term in ["pii removed", "personal identifiable", "sensitive information were removed", "anonym", "no pii"])
    synthetic = "synthetic" in lower or "generated" in lower
    return license_name, pii_removed, synthetic, note


def data_files(dataset_dir: Path) -> list[Path]:
    keep = {".csv", ".jsonl", ".txt", ".xlsx", ".md"}
    files = []
    for path in dataset_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in keep and ".cache" not in path.parts and not path.name.startswith("._"):
            files.append(path)
    return sorted(files)


def profile_rows(path: Path, sample_limit: int) -> tuple[str, int | str, list[str], list[dict[str, Any]]]:
    ext = path.suffix.lower()
    if ext == ".xlsx":
        return "skipped_xlsx_requires_manual_or_openpyxl_review", "", [], []
    if ext in {".txt", ".md"}:
        lines = [line for line in read_text(path, 200_000).splitlines() if line.strip()]
        return "ok", len(lines), ["text"], [{"text": line} for line in lines[: min(sample_limit, 100)]]
    if ext == ".jsonl":
        rows, columns, count = [], set(), 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                count += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    columns.update(str(key) for key in obj)
                    if len(rows) < sample_limit:
                        rows.append(obj)
        return "ok", count, sorted(columns), rows
    if ext == ".csv":
        rows, count = [], 0
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            for row in reader:
                count += 1
                if len(rows) < sample_limit:
                    rows.append({k: v for k, v in row.items() if k is not None})
        return "ok", count, columns, rows
    return "skipped_unknown_extension", "", [], []


def matching_columns(columns: list[str], hints: list[str]) -> list[str]:
    return [column for column in columns if any(hint in column.lower() for hint in hints)]


def matching_label_columns(columns: list[str]) -> list[str]:
    exact = {
        "affected_service",
        "category",
        "category_id",
        "class",
        "department",
        "escalated",
        "intent",
        "label",
        "labels",
        "priority",
        "queue",
        "requester_department",
        "service",
        "status",
        "type",
        "y",
    }
    matches = []
    for column in columns:
        lower = column.lower()
        if lower in exact or lower.endswith("_label") or lower.endswith("_category"):
            matches.append(column)
    return matches


def scan_values(rows: list[dict[str, Any]]) -> dict[str, int]:
    text = "\n".join(str(value) for row in rows for value in row.values() if value is not None)
    counts = {**pattern_counts(text, PII_PATTERNS), **pattern_counts(text, SECRET_PATTERNS)}
    counts["credential_term_count"] = sum(text.lower().count(term) for term in ["password", "credential", "token", "secret", "mfa"])
    return counts


def license_lanes(license_name: str) -> set[str]:
    lower = license_name.lower()
    lanes = {"license_review_required"}
    if "cc-by-nc" in lower or "noncommercial" in lower or "non-commercial" in lower:
        lanes.update({"poc_only_noncommercial", "commercial_excluded"})
    elif "cc-by-sa" in lower or "cc by-sa" in lower:
        lanes.update({"poc_only_share_alike", "commercial_excluded"})
    elif any(term in lower for term in ["cc-by-4.0", "cc by 4.0", "mit", "apache", "cc0"]):
        lanes.add("commercial_candidate_after_provenance_review")
    return lanes


def ticket_gate(run_id: str, quality: dict[str, dict[str, Any]], sample_limit: int) -> dict[str, Any]:
    source_id = "public_it_helpdesk_ticket_datasets"
    root = DATA / "raw" / source_id / run_id
    dataset_dirs = []
    for kind in ("huggingface", "zenodo"):
        base = root / kind
        if base.exists():
            dataset_dirs.extend((kind, path) for path in base.iterdir() if path.is_dir())

    intake_rows, file_rows, preview_rows = [], [], []
    lane_counter: Counter[str] = Counter()
    for kind, dataset_dir in sorted(dataset_dirs, key=lambda item: (item[0], item[1].name)):
        dataset_id = f"{kind}/{dataset_dir.name}"
        license_name, pii_removed, synthetic, note = dataset_license(dataset_dir)
        dataset_lanes = {"blocked_from_training_pending_review", *license_lanes(license_name)}
        if pii_removed or synthetic:
            dataset_lanes.add("pii_reduction_claimed")
        totals = Counter()
        files = data_files(dataset_dir)
        for path in files:
            status, row_count, columns, rows = profile_rows(path, sample_limit)
            text_cols = matching_columns(columns, TEXT_HINTS)
            label_cols = matching_label_columns(columns)
            signals = scan_values(rows)
            lanes = set(dataset_lanes)
            if text_cols:
                lanes.add("classifier_candidate")
            if label_cols:
                lanes.add("label_mapping_candidate")
            if any(col.lower() in {"output", "response", "resolution", "steps"} for col in columns):
                lanes.add("retrieval_candidate_after_safety_filter")
            if signals["email_like"] or signals["phone_like"] or signals["api_key_like"] or signals["private_key_like"] or signals["aws_key_like"] or signals["github_token_like"]:
                lanes.add("needs_redaction")
            file_rows.append(
                {
                    "dataset_id": dataset_id,
                    "path": rel(path),
                    "extension": path.suffix.lower(),
                    "bytes": path.stat().st_size,
                    "read_status": status,
                    "row_count": row_count,
                    "columns_json": compact(columns),
                    "text_columns": ";".join(text_cols),
                    "label_columns": ";".join(label_cols),
                    "declared_license": license_name,
                    "pii_signal_count": signals["email_like"] + signals["phone_like"] + signals["ipv4_like"],
                    "secret_like_signal_count": signals["api_key_like"] + signals["private_key_like"] + signals["aws_key_like"] + signals["github_token_like"],
                    "credential_term_count": signals["credential_term_count"],
                    "provisional_use_lanes": ";".join(sorted(lanes)),
                    "gate_decision": "inspect_only_pending_pii_license_schema",
                }
            )
            lane_counter.update(lanes)
            totals["data_files"] += int(path.suffix.lower() in {".csv", ".jsonl", ".txt", ".xlsx"})
            totals["text_files"] += int(bool(text_cols))
            totals["label_files"] += int(bool(label_cols))
            totals["pii"] += signals["email_like"] + signals["phone_like"] + signals["ipv4_like"]
            totals["secret"] += signals["api_key_like"] + signals["private_key_like"] + signals["aws_key_like"] + signals["github_token_like"]
            if isinstance(row_count, int):
                totals["rows"] += row_count
            if rows and text_cols and path.suffix.lower() != ".md" and len(preview_rows) < 140:
                for row in rows[:2]:
                    text = " ".join(str(row.get(col, "")) for col in text_cols[:4])
                    labels = {col: row.get(col) for col in label_cols[:8] if col in row}
                    preview_rows.append({"dataset_id": dataset_id, "path": rel(path), "preview": {"text_snippet": snippet(text), "labels": labels}, "provisional_use_lanes": sorted(lanes)})
                    if len(preview_rows) >= 140:
                        break
        if totals["text_files"]:
            dataset_lanes.add("classifier_candidate")
        if totals["label_files"]:
            dataset_lanes.add("label_mapping_candidate")
        if totals["pii"] or totals["secret"]:
            dataset_lanes.add("needs_redaction")
        lane_counter.update(dataset_lanes)
        intake_rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_dir": rel(dataset_dir),
                "declared_license": license_name,
                "declared_pii_removed": pii_removed,
                "declared_synthetic": synthetic,
                "file_count": len(files),
                "data_file_count": totals["data_files"],
                "text_file_count": totals["text_files"],
                "label_file_count": totals["label_files"],
                "sampled_row_count_total": totals["rows"],
                "pii_signal_count": totals["pii"],
                "secret_like_signal_count": totals["secret"],
                "source_quality_gate": quality[source_id]["gate_decision"],
                "provisional_use_lanes": ";".join(sorted(dataset_lanes)),
                "next_action": "manual_sample_review_then_dataset_level_allow_or_exclude",
                "provenance_note": note,
            }
        )
    write_csv(OUT / "ticket_dataset_intake.csv", intake_rows)
    write_csv(OUT / "ticket_dataset_file_profiles.csv", file_rows)
    write_jsonl(OUT / "ticket_dataset_redacted_preview.jsonl", preview_rows)
    return {
        "run_id": run_id,
        "datasets": len(intake_rows),
        "file_profiles": len(file_rows),
        "preview_records": len(preview_rows),
        "gate_counts": dict(Counter(row["source_quality_gate"] for row in intake_rows)),
        "use_lane_counts": dict(lane_counter),
    }


def md_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join("---" for _ in fields) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")).replace("|", "\\|") for field in fields) + " |")
    return "\n".join(lines)


def write_summary(stack: dict[str, Any], tickets: dict[str, Any]) -> None:
    outputs = [
        OUT / "gate_summary.md",
        OUT / "gate_summary.json",
        OUT / "use_lane_definitions.json",
        OUT / "stack_exchange_scaled_run_gate.csv",
        OUT / "stack_exchange_domain_signal_matrix.csv",
        OUT / "stack_exchange_gated_preview.jsonl",
        OUT / "ticket_dataset_intake.csv",
        OUT / "ticket_dataset_file_profiles.csv",
        OUT / "ticket_dataset_redacted_preview.jsonl",
    ]
    summary = {
        "stage": "source_specific_preprocessing_gate",
        "policy": "No training, indexing, or generation. Assign use lanes and review decisions only.",
        "stack_exchange": stack,
        "ticket_datasets": tickets,
        "outputs": [rel(path) for path in outputs],
    }
    write_json(OUT / "use_lane_definitions.json", USE_LANES)
    write_json(OUT / "gate_summary.json", summary)
    stack_counts = [{"gate_decision": key, "records": value} for key, value in sorted(stack["gate_counts"].items())]
    ticket_lanes = [{"use_lane": key, "count": value} for key, value in sorted(tickets["use_lane_counts"].items())]
    lines = [
        "# Source-Specific Preprocessing Gate",
        "",
        "This is a forensic intake gate, not model training, RAG indexing, or demo answer generation.",
        "",
        "## Outputs",
        "",
        *[f"- `{rel(path)}`" for path in outputs[1:]],
        "",
        "## Stack Exchange Scaled Run",
        "",
        f"- Run id: `{stack['run_id']}`",
        f"- Manifest question hits: {stack['manifest_question_hits']}",
        f"- Records gated: {stack['records']}",
        f"- Duplicate question hits collapsed: {stack['duplicate_question_hits_collapsed']}",
        f"- Redacted preview records: {stack['preview_records']}",
        "",
        md_table(stack_counts, ["gate_decision", "records"]),
        "",
        "## Ticket Dataset Intake",
        "",
        f"- Run id: `{tickets['run_id']}`",
        f"- Datasets inventoried: {tickets['datasets']}",
        f"- File profiles: {tickets['file_profiles']}",
        f"- Redacted preview records: {tickets['preview_records']}",
        "",
        md_table(ticket_lanes, ["use_lane", "count"]),
        "",
        "## Downstream Rule",
        "",
        "Use this gate to maximize legitimate signal source-by-source. A record may be useful for classification, evaluation, defensive metadata, or POC retrieval even when it is blocked from generation, training, or commercial-mode use.",
        "",
        "Next safe move: manually inspect the ticket dataset intake table and Stack Exchange security/firmware gate decisions before promoting any subset to normalized training, retrieval, or evaluation records.",
    ]
    (OUT / "gate_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stack-run", default="20260506T145259Z")
    parser.add_argument("--ticket-run", default="20260506T154614Z")
    parser.add_argument("--sample-limit", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    quality = quality_matrix()
    stack = stack_exchange_gate(args.stack_run, quality)
    tickets = ticket_gate(args.ticket_run, quality, args.sample_limit)
    write_summary(stack, tickets)
    print(f"wrote {rel(OUT / 'gate_summary.md')}")
    print(f"stack_exchange_records={stack['records']}")
    print(f"ticket_datasets={tickets['datasets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
