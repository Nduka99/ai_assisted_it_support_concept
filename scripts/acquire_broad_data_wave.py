"""Acquire a broad first wave of non-StackExchange IT-SUPPORT data.

This script intentionally stays simple: download raw source artifacts, capture
attribution, and write one run report. Normalization happens after inspection.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
import trafilatura
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_ID = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
USER_AGENT = "IT-SUPPORT-POC/0.1 data acquisition; contact: local research project"
SUCCESS_STATUSES = {"ok", "existing"}
LIVE_TOTAL = 0


EXTRA_SOURCE_FAMILIES = [
    {
        "id": "public_it_helpdesk_ticket_datasets",
        "name": "Public IT/helpdesk ticket datasets",
        "status": "candidate",
        "priority": 1,
        "source_type": "ticket_dataset",
        "acquisition_mode": "dataset_snapshot_or_record_api",
        "seed_urls": [
            "https://huggingface.co/datasets/benjaminmacklin/IT_Support",
            "https://zenodo.org/records/7384758",
        ],
        "license": "dataset-specific; capture license/provenance per dataset",
        "commercial_posture": "license_varies_capture_per_dataset",
        "attribution_required": True,
        "share_alike_required": False,
        "target_domains": [
            "hardware",
            "os_kernel_drivers",
            "application_software",
            "network_connectivity",
            "identity_access_accounts",
            "storage_data_backup",
            "security_malware",
        ],
        "notes": "Use for classifier/ticket-routing language only after PII/provenance inspection.",
    },
    {
        "id": "microsoftdocs_github_repos",
        "name": "MicrosoftDocs GitHub source repositories",
        "status": "candidate",
        "priority": 2,
        "source_type": "vendor_documentation_source_repo",
        "acquisition_mode": "github_archive_zip",
        "seed_urls": ["https://github.com/MicrosoftDocs/SupportArticles-docs"],
        "license": "repo-specific; many MicrosoftDocs repos use CC BY 4.0 for docs and MIT for code",
        "commercial_posture": "repo_license_review_required",
        "attribution_required": True,
        "share_alike_required": False,
        "target_domains": [
            "os_kernel_drivers",
            "application_software",
            "network_connectivity",
            "identity_access_accounts",
            "storage_data_backup",
            "security_malware",
            "firmware_bios_uefi",
        ],
        "notes": "Prefer source repos with explicit LICENSE files over blind scraping of Learn pages.",
    },
    {
        "id": "fedora_docs",
        "name": "Fedora Docs",
        "status": "candidate",
        "priority": 3,
        "source_type": "documentation",
        "acquisition_mode": "allowlisted_page_capture",
        "seed_urls": ["https://docs.fedoraproject.org/"],
        "license": "CC BY-SA 4.0 unless specifically noted otherwise",
        "commercial_posture": "commercially_constrained_share_alike",
        "attribution_required": True,
        "share_alike_required": True,
        "target_domains": [
            "os_kernel_drivers",
            "application_software",
            "network_connectivity",
            "storage_data_backup",
            "security_malware",
        ],
        "notes": "Linux admin and troubleshooting reference evidence.",
    },
    {
        "id": "archwiki_docs",
        "name": "ArchWiki documentation",
        "status": "candidate",
        "priority": 3,
        "source_type": "documentation",
        "acquisition_mode": "allowlisted_page_capture",
        "seed_urls": ["https://wiki.archlinux.org/"],
        "license": "GFDL 1.3 or later unless otherwise noted",
        "commercial_posture": "poc_only_until_gfdl_handling_is_explicit",
        "attribution_required": True,
        "share_alike_required": True,
        "target_domains": [
            "hardware",
            "os_kernel_drivers",
            "application_software",
            "network_connectivity",
            "identity_access_accounts",
            "storage_data_backup",
            "security_malware",
        ],
        "notes": "High-quality Linux troubleshooting; keep GFDL lineage separate.",
    },
    {
        "id": "nvd_security_feeds",
        "name": "NVD JSON 2.0 vulnerability and product feeds",
        "status": "candidate",
        "priority": 1,
        "source_type": "security_metadata",
        "acquisition_mode": "official_json_feed_download",
        "seed_urls": ["https://nvd.nist.gov/vuln/data-feeds"],
        "license": "NVD public-service data; API terms and attribution notice apply",
        "commercial_posture": "commercially_usable_with_attribution_notice_and_terms",
        "attribution_required": True,
        "share_alike_required": False,
        "target_domains": ["security_malware", "firmware_bios_uefi", "application_software", "hardware"],
        "notes": "Use for CVE/CPE/security escalation and firmware safety metadata.",
    },
    {
        "id": "cisa_kev_catalog",
        "name": "CISA Known Exploited Vulnerabilities catalog",
        "status": "candidate",
        "priority": 1,
        "source_type": "security_metadata",
        "acquisition_mode": "official_json_feed_download",
        "seed_urls": ["https://www.cisa.gov/known-exploited-vulnerabilities-catalog"],
        "license": "CISA.gov public domain; external references have their own terms",
        "commercial_posture": "commercially_usable_public_domain_with_attribution_caution",
        "attribution_required": True,
        "share_alike_required": False,
        "target_domains": ["security_malware", "firmware_bios_uefi", "network_connectivity"],
        "notes": "Use for exploited-in-the-wild safety and urgency signals.",
    },
    {
        "id": "github_advisory_database",
        "name": "GitHub Advisory Database",
        "status": "candidate",
        "priority": 1,
        "source_type": "security_metadata",
        "acquisition_mode": "github_archive_zip",
        "seed_urls": ["https://github.com/github/advisory-database"],
        "license": "CC BY 4.0",
        "commercial_posture": "commercially_usable_with_attribution",
        "attribution_required": True,
        "share_alike_required": False,
        "target_domains": ["security_malware", "application_software"],
        "notes": "Open-source vulnerability advisories in OSV format.",
    },
]


DOC_PAGES = [
    ("ubuntu_official_and_community_docs", "CC BY-SA 4.0 unless otherwise stated", [
        ("https://help.ubuntu.com/community/WifiDocs/WirelessTroubleShootingGuide", ["network_connectivity", "os_kernel_drivers"]),
        ("https://help.ubuntu.com/community/HardwareSupport", ["hardware", "os_kernel_drivers"]),
        ("https://help.ubuntu.com/community/BinaryDriverHowto/Nvidia", ["hardware", "os_kernel_drivers"]),
        ("https://help.ubuntu.com/community/FilePermissions", ["identity_access_accounts", "os_kernel_drivers"]),
        ("https://help.ubuntu.com/community/Boot-Repair", ["os_kernel_drivers", "storage_data_backup"]),
        ("https://help.ubuntu.com/community/Grub2/Troubleshooting", ["os_kernel_drivers", "storage_data_backup"]),
        ("https://help.ubuntu.com/community/DiskSpace", ["storage_data_backup"]),
        ("https://help.ubuntu.com/community/BackupYourSystem", ["storage_data_backup"]),
    ]),
    ("fedora_docs", "CC BY-SA 4.0 unless specifically noted otherwise", [
        ("https://docs.fedoraproject.org/en-US/quick-docs/", ["os_kernel_drivers", "application_software"]),
        ("https://docs.fedoraproject.org/en-US/fedora/latest/system-administrators-guide/", ["os_kernel_drivers", "network_connectivity"]),
        ("https://docs.fedoraproject.org/en-US/quick-docs/getting-started-guide/", ["application_software"]),
        ("https://docs.fedoraproject.org/en-US/quick-docs/upgrading-fedora-offline/", ["application_software", "os_kernel_drivers"]),
    ]),
    ("archwiki_docs", "GFDL 1.3 or later unless otherwise noted", [
        ("https://wiki.archlinux.org/title/General_troubleshooting", ["os_kernel_drivers", "application_software"]),
        ("https://wiki.archlinux.org/title/Network_configuration", ["network_connectivity"]),
        ("https://wiki.archlinux.org/title/Wireless_network_configuration", ["network_connectivity", "os_kernel_drivers"]),
        ("https://wiki.archlinux.org/title/Users_and_groups", ["identity_access_accounts"]),
        ("https://wiki.archlinux.org/title/File_permissions_and_attributes", ["identity_access_accounts", "os_kernel_drivers"]),
        ("https://wiki.archlinux.org/title/Kernel_modules", ["os_kernel_drivers"]),
        ("https://wiki.archlinux.org/title/GRUB", ["os_kernel_drivers", "storage_data_backup"]),
        ("https://wiki.archlinux.org/title/S.M.A.R.T.", ["hardware", "storage_data_backup"]),
        ("https://wiki.archlinux.org/title/Solid_state_drive", ["hardware", "storage_data_backup"]),
    ]),
    ("mdn_web_docs", "CC BY-SA 2.5 or later unless otherwise indicated", [
        ("https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Overview", ["application_software", "network_connectivity"]),
        ("https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Authentication", ["identity_access_accounts", "application_software"]),
        ("https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/CORS", ["application_software", "network_connectivity"]),
        ("https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Cookies", ["application_software", "security_malware"]),
        ("https://developer.mozilla.org/en-US/docs/Web/Security", ["application_software", "security_malware"]),
    ]),
    ("linux_man_pages", "page-specific free software/manual licenses", [
        ("https://man7.org/linux/man-pages/man8/ip.8.html", ["network_connectivity", "os_kernel_drivers"]),
        ("https://man7.org/linux/man-pages/man8/mount.8.html", ["storage_data_backup", "os_kernel_drivers"]),
        ("https://man7.org/linux/man-pages/man8/fsck.8.html", ["storage_data_backup"]),
        ("https://man7.org/linux/man-pages/man1/chmod.1.html", ["identity_access_accounts", "os_kernel_drivers"]),
        ("https://man7.org/linux/man-pages/man1/dmesg.1.html", ["os_kernel_drivers", "hardware"]),
    ]),
]


API_PAGES = [
    ("ifixit_guides", "CC BY-NC-SA 3.0", [
        ("https://www.ifixit.com/api/2.0/categories/Laptop", ["hardware"]),
        ("https://www.ifixit.com/api/2.0/categories/PC%20Laptop", ["hardware"]),
        ("https://www.ifixit.com/api/2.0/categories/Desktop%20PC", ["hardware"]),
        ("https://www.ifixit.com/api/2.0/categories/Hard%20Drive", ["hardware", "storage_data_backup"]),
    ]),
    ("vendor_firmware_security_advisories", "vendor terms vary", [
        ("https://support.hp.com/us-en/security-bulletins", ["hardware", "firmware_bios_uefi", "security_malware"]),
        ("https://support.hp.com/us-en/security-bulletin-rss", ["hardware", "firmware_bios_uefi", "security_malware"]),
        ("https://support.lenovo.com/us/en/product_security/home", ["hardware", "firmware_bios_uefi", "security_malware"]),
        ("https://www.dell.com/support/security/en-us", ["hardware", "firmware_bios_uefi", "security_malware"]),
        ("https://www.asus.com/content/asus-product-security-advisory/", ["hardware", "firmware_bios_uefi", "security_malware"]),
        ("https://www.acer.com/us-en/support/security-advisories", ["hardware", "firmware_bios_uefi", "security_malware"]),
    ]),
]


REPO_ZIPS = [
    ("github_advisory_database", "github_advisory_database_main.zip", "https://github.com/github/advisory-database/archive/refs/heads/main.zip"),
    ("microsoftdocs_github_repos", "MicrosoftDocs_SupportArticles-docs_main.zip", "https://github.com/MicrosoftDocs/SupportArticles-docs/archive/refs/heads/main.zip"),
    ("microsoftdocs_github_repos", "MicrosoftDocs_windowsserverdocs_main.zip", "https://github.com/MicrosoftDocs/windowsserverdocs/archive/refs/heads/main.zip"),
    ("microsoftdocs_github_repos", "MicrosoftDocs_windows-driver-docs_main.zip", "https://github.com/MicrosoftDocs/windows-driver-docs/archive/refs/heads/main.zip"),
    ("microsoftdocs_github_repos", "MicrosoftDocs_microsoft-365-docs_public.zip", "https://github.com/MicrosoftDocs/microsoft-365-docs/archive/refs/heads/public.zip"),
    ("microsoftdocs_github_repos", "MicrosoftDocs_PowerShell-Docs_main.zip", "https://github.com/MicrosoftDocs/PowerShell-Docs/archive/refs/heads/main.zip"),
    ("microsoftdocs_github_repos", "MicrosoftDocs_windows-itpro-docs_public.zip", "https://github.com/MicrosoftDocs/windows-itpro-docs/archive/refs/heads/public.zip"),
    ("mdn_web_docs", "mdn_content_main.zip", "https://github.com/mdn/content/archive/refs/heads/main.zip"),
]


HF_DATASETS = [
    "benjaminmacklin/IT_Support",
    "Tobi-Bueck/customer-support-tickets",
    "Noise144/ticket_classification_IT_EN",
    "ale-dp/german-english-email-ticket-classification",
    "mindweave/help-desk-tickets",
    "CJJones/ServiceNow_Search_NLP_2_JSON_LLM_Training_Sample",
]


ZENODO_RECORDS = ["7384758", "7648117"]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")[:160]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_registry_run(registry_path: Path, run_entry: dict[str, Any]) -> None:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    existing_runs = registry.setdefault("acquisition_runs", [])
    registry["acquisition_runs"] = [
        run
        for run in existing_runs
        if not (
            run.get("run_id") == run_entry["run_id"]
            and run.get("source_family_id") == run_entry["source_family_id"]
        )
    ]
    registry["acquisition_runs"].append(run_entry)
    write_json(registry_path, registry)


def progress_dir() -> Path:
    return PROJECT_ROOT / "data" / "processed" / "broad_data_wave" / RUN_ID


def write_current_download(payload: dict[str, Any]) -> None:
    path = progress_dir() / "current_download.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def count_expected_artifacts(args: argparse.Namespace) -> int:
    total = 0
    if not args.skip_doc_pages:
        total += sum(len(pages) for _, _, pages in DOC_PAGES + API_PAGES)
    if not args.skip_feeds:
        total += len(nvd_urls(args.include_large)) + 2
    if not args.skip_repo_zips:
        total += len(REPO_ZIPS)
    if not args.skip_hf:
        total += len(HF_DATASETS)
    if not args.skip_zenodo:
        total += len(ZENODO_RECORDS)
    return total


def update_live_status(report: list[dict[str, Any]]) -> None:
    done = len(report)
    ok = sum(item["status"] in SUCCESS_STATUSES for item in report)
    failed = sum(item["status"] not in SUCCESS_STATUSES for item in report)
    bytes_done = sum(int(item.get("bytes") or 0) for item in report if item["status"] in SUCCESS_STATUSES)
    last = report[-1] if report else {}
    lines = [
        "# Live Broad Acquisition Status",
        "",
        f"Run id: `{RUN_ID}`",
        f"Artifacts processed: {done}/{LIVE_TOTAL or '?'}",
        f"Successful/existing: {ok}",
        f"Failed/gated: {failed}",
        f"Bytes present: {bytes_done:,}",
    ]
    if last:
        lines.extend([
            "",
            "## Latest Artifact",
            "",
            f"- Source: `{last.get('source_id')}`",
            f"- Kind: `{last.get('kind')}`",
            f"- Status: `{last.get('status')}`",
            f"- URL: {last.get('url')}",
            f"- Path: `{last.get('path')}`",
        ])
        if last.get("error"):
            lines.append(f"- Error: `{last['error']}`")
    path = progress_dir() / "live_status.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def reset_live_progress_files() -> None:
    path = progress_dir()
    path.mkdir(parents=True, exist_ok=True)
    for name in ("progress.jsonl", "current_download.json", "live_status.md"):
        target = path / name
        if target.exists():
            target.unlink()


def record_result(
    rows: list[dict[str, Any]],
    report: list[dict[str, Any]],
    row: dict[str, Any],
    item: dict[str, Any],
    overall: tqdm | None,
) -> None:
    rows.append(row)
    report.append(item)
    progress_path = progress_dir() / "progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    update_live_status(report)
    if overall:
        overall.update(1)
        overall.set_postfix_str(f"{item['source_id']}:{item['status']}", refresh=False)


def ensure_source_families(registry_path: Path) -> None:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    known = {family["id"] for family in registry["source_families"]}
    for family in EXTRA_SOURCE_FAMILIES:
        if family["id"] not in known:
            registry["source_families"].append(family)
    write_json(registry_path, registry)


def download(
    session: requests.Session,
    url: str,
    dest: Path,
    *,
    force: bool = False,
    show_progress: bool = True,
) -> dict[str, Any]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if dest.exists() and dest.stat().st_size > 0 and not force:
        write_current_download(
            {
                "status": "existing",
                "url": url,
                "path": str(dest.relative_to(PROJECT_ROOT)),
                "bytes_done": dest.stat().st_size,
                "total_bytes": dest.stat().st_size,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        return {
            "status": "existing",
            "url": url,
            "path": str(dest.relative_to(PROJECT_ROOT)),
            "bytes": dest.stat().st_size,
            "seconds": 0,
        }
    if dest.exists() and dest.stat().st_size == 0:
        dest.unlink()
    temp_path = dest.with_name(f"{dest.name}.part")
    if temp_path.exists():
        temp_path.unlink()
    try:
        with session.get(url, timeout=60, stream=True) as response:
            status = response.status_code
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
            bytes_done = 0
            last_status_write = 0.0
            write_current_download(
                {
                    "status": "downloading",
                    "url": url,
                    "path": str(dest.relative_to(PROJECT_ROOT)),
                    "bytes_done": 0,
                    "total_bytes": total,
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                }
            )
            bar = tqdm(
                total=total or None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name[:48],
                leave=False,
                disable=not show_progress,
            )
            with temp_path.open("wb") as handle, bar:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        bar.update(len(chunk))
                        bytes_done += len(chunk)
                        now = time.time()
                        if now - last_status_write >= 0.5:
                            write_current_download(
                                {
                                    "status": "downloading",
                                    "url": url,
                                    "path": str(dest.relative_to(PROJECT_ROOT)),
                                    "bytes_done": bytes_done,
                                    "total_bytes": total,
                                    "updated_at_utc": datetime.now(UTC).isoformat(),
                                }
                            )
                            last_status_write = now
        temp_path.replace(dest)
        write_current_download(
            {
                "status": "complete",
                "url": url,
                "path": str(dest.relative_to(PROJECT_ROOT)),
                "bytes_done": dest.stat().st_size,
                "total_bytes": dest.stat().st_size,
                "updated_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        return {
            "status": "ok",
            "url": url,
            "path": str(dest.relative_to(PROJECT_ROOT)),
            "bytes": dest.stat().st_size,
            "seconds": round(time.time() - started, 2),
        }
    except Exception as exc:  # noqa: BLE001 - record acquisition failures, do not hide them
        write_current_download(
            {
                "status": "failed",
                "url": url,
                "path": str(dest.relative_to(PROJECT_ROOT)),
                "bytes_done": temp_path.stat().st_size if temp_path.exists() else 0,
                "total_bytes": 0,
                "error": repr(exc),
                "updated_at_utc": datetime.now(UTC).isoformat(),
            }
        )
        return {"status": "failed", "url": url, "path": str(dest.relative_to(PROJECT_ROOT)), "error": repr(exc)}


def source_dir(source_id: str) -> Path:
    return PROJECT_ROOT / "data" / "raw" / source_id / RUN_ID


def attr_row(source_id: str, url: str, license_name: str, domains: list[str], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_family_id": source_id,
        "record_type": "raw_source_artifact",
        "source_url": url,
        "license": license_name,
        "target_domains": domains,
        "acquired_at_utc": RUN_ID,
        "raw_path": result.get("path"),
        "bytes": result.get("bytes"),
        "status": result["status"],
        "error": result.get("error"),
    }


def acquire_doc_pages(
    session: requests.Session,
    rows: list[dict[str, Any]],
    report: list[dict[str, Any]],
    overall: tqdm | None,
    *,
    force: bool,
    show_progress: bool,
) -> None:
    for source_id, license_name, pages in [*DOC_PAGES, *API_PAGES]:
        for url, domains in pages:
            base = safe_name(url)
            result = download(session, url, source_dir(source_id) / f"{base}.raw", force=force, show_progress=show_progress)
            if result["status"] in SUCCESS_STATUSES:
                raw_path = PROJECT_ROOT / result["path"]
                text_path = raw_path.with_suffix(".txt")
                if not text_path.exists() or force:
                    text = trafilatura.extract(raw_path.read_bytes(), url=url) or ""
                else:
                    text = text_path.read_text(encoding="utf-8")
                if text and (not text_path.exists() or force):
                    text_path = raw_path.with_suffix(".txt")
                    text_path.write_text(text, encoding="utf-8")
                if text:
                    result["extracted_text_path"] = str(text_path.relative_to(PROJECT_ROOT))
                    result["text_chars"] = len(text)
            item = {"source_id": source_id, "kind": "page", **result}
            record_result(rows, report, attr_row(source_id, url, license_name, domains, result), item, overall)
            tqdm.write(f"{source_id}: {result['status']} {url}")


def nvd_urls(include_large: bool) -> list[tuple[str, str]]:
    years = list(range(2002, max(datetime.now(UTC).year, 2026) + 1))
    names = ["modified", "recent", *[str(year) for year in years]]
    urls: list[tuple[str, str]] = []
    for name in names:
        stem = f"https://nvd.nist.gov/feeds/json/cve/2.0/nvdcve-2.0-{name}"
        urls.extend([(f"nvdcve-2.0-{name}.meta", f"{stem}.meta"), (f"nvdcve-2.0-{name}.json.gz", f"{stem}.json.gz")])
    urls.extend([
        ("vendorcomments.meta", "https://nvd.nist.gov/feeds/xml/vendorcomments/vendorcomments.meta"),
        ("vendorcomments.xml.gz", "https://nvd.nist.gov/feeds/xml/vendorcomments/vendorcomments.xml.gz"),
        ("nvdcpe-2.0.meta", "https://nvd.nist.gov/feeds/json/cpe/2.0/nvdcpe-2.0.meta"),
        ("nvdcpe-2.0.tar.gz", "https://nvd.nist.gov/feeds/json/cpe/2.0/nvdcpe-2.0.tar.gz"),
    ])
    if include_large:
        urls.extend([
            ("nvdcpematch-2.0.meta", "https://nvd.nist.gov/feeds/json/cpematch/2.0/nvdcpematch-2.0.meta"),
            ("nvdcpematch-2.0.tar.gz", "https://nvd.nist.gov/feeds/json/cpematch/2.0/nvdcpematch-2.0.tar.gz"),
        ])
    return urls


def acquire_feed_files(
    session: requests.Session,
    rows: list[dict[str, Any]],
    report: list[dict[str, Any]],
    overall: tqdm | None,
    *,
    include_large: bool,
    force: bool,
    show_progress: bool,
) -> None:
    feeds = [
        ("cisa_kev_catalog", "CISA.gov public domain", ["security_malware", "firmware_bios_uefi"], "known_exploited_vulnerabilities.json", "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.json"),
        ("cisa_kev_catalog", "CISA.gov public domain", ["security_malware", "firmware_bios_uefi"], "known_exploited_vulnerabilities.csv", "https://www.cisa.gov/sites/default/files/feeds/known_exploited_vulnerabilities.csv"),
    ]
    feeds.extend(
        ("nvd_security_feeds", "NVD public-service data; API terms apply", ["security_malware", "firmware_bios_uefi"], name, url)
        for name, url in nvd_urls(include_large)
    )
    for source_id, license_name, domains, name, url in feeds:
        result = download(session, url, source_dir(source_id) / name, force=force, show_progress=show_progress)
        item = {"source_id": source_id, "kind": "feed", **result}
        record_result(rows, report, attr_row(source_id, url, license_name, domains, result), item, overall)
        tqdm.write(f"{source_id}: {result['status']} {name}")


def acquire_repo_zips(
    session: requests.Session,
    rows: list[dict[str, Any]],
    report: list[dict[str, Any]],
    overall: tqdm | None,
    *,
    force: bool,
    show_progress: bool,
) -> None:
    for source_id, filename, url in REPO_ZIPS:
        result = download(session, url, source_dir(source_id) / filename, force=force, show_progress=show_progress)
        item = {"source_id": source_id, "kind": "repo_zip", **result}
        record_result(rows, report, attr_row(source_id, url, "repo-specific; inspect LICENSE files", [], result), item, overall)
        tqdm.write(f"{source_id}: {result['status']} {filename}")


def acquire_hf_datasets(
    rows: list[dict[str, Any]],
    report: list[dict[str, Any]],
    overall: tqdm | None,
) -> None:
    token = os.getenv("HF_TOKEN") or None
    for repo_id in HF_DATASETS:
        target = source_dir("public_it_helpdesk_ticket_datasets") / "huggingface" / safe_name(repo_id)
        url = f"https://huggingface.co/datasets/{repo_id}"
        started = time.time()
        try:
            write_current_download(
                {
                    "status": "snapshot_downloading",
                    "url": url,
                    "path": str(target.relative_to(PROJECT_ROOT)),
                    "bytes_done": 0,
                    "total_bytes": 0,
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                }
            )
            snapshot_download(repo_id=repo_id, repo_type="dataset", local_dir=target, token=token)
            total_bytes = sum(path.stat().st_size for path in target.rglob("*") if path.is_file())
            write_current_download(
                {
                    "status": "complete",
                    "url": url,
                    "path": str(target.relative_to(PROJECT_ROOT)),
                    "bytes_done": total_bytes,
                    "total_bytes": total_bytes,
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                }
            )
            result = {
                "status": "ok",
                "url": url,
                "path": str(target.relative_to(PROJECT_ROOT)),
                "bytes": total_bytes,
                "seconds": round(time.time() - started, 2),
            }
        except Exception as exc:  # noqa: BLE001
            write_current_download(
                {
                    "status": "failed",
                    "url": url,
                    "path": str(target.relative_to(PROJECT_ROOT)),
                    "bytes_done": sum(path.stat().st_size for path in target.rglob("*") if path.is_file()) if target.exists() else 0,
                    "total_bytes": 0,
                    "error": repr(exc),
                    "updated_at_utc": datetime.now(UTC).isoformat(),
                }
            )
            result = {"status": "failed", "url": url, "path": str(target.relative_to(PROJECT_ROOT)), "error": repr(exc)}
        item = {"source_id": "public_it_helpdesk_ticket_datasets", "kind": "hf_dataset", **result}
        record_result(
            rows,
            report,
            attr_row("public_it_helpdesk_ticket_datasets", url, "dataset-specific; inspect dataset card", [], result),
            item,
            overall,
        )
        tqdm.write(f"public_it_helpdesk_ticket_datasets: {result['status']} {repo_id}")


def acquire_zenodo(
    session: requests.Session,
    rows: list[dict[str, Any]],
    report: list[dict[str, Any]],
    overall: tqdm | None,
    *,
    force: bool,
    show_progress: bool,
) -> None:
    for record_id in ZENODO_RECORDS:
        api_url = f"https://zenodo.org/api/records/{record_id}"
        meta_result = download(
            session,
            api_url,
            source_dir("public_it_helpdesk_ticket_datasets") / "zenodo" / record_id / "record_metadata.json",
            force=force,
            show_progress=show_progress,
        )
        meta_item = {"source_id": "public_it_helpdesk_ticket_datasets", "kind": "zenodo_metadata", **meta_result}
        record_result(
            rows,
            report,
            attr_row("public_it_helpdesk_ticket_datasets", api_url, "record-specific Zenodo license metadata", [], meta_result),
            meta_item,
            overall,
        )
        if meta_result["status"] not in SUCCESS_STATUSES:
            continue
        metadata = json.loads((PROJECT_ROOT / meta_result["path"]).read_text(encoding="utf-8"))
        for file_info in metadata.get("files", []):
            file_url = file_info.get("links", {}).get("self")
            if not file_url:
                continue
            filename = safe_name(file_info.get("key") or Path(file_url).name)
            if overall:
                overall.total = (overall.total or 0) + 1
                overall.refresh()
            result = download(
                session,
                file_url,
                source_dir("public_it_helpdesk_ticket_datasets") / "zenodo" / record_id / filename,
                force=force,
                show_progress=show_progress,
            )
            item = {"source_id": "public_it_helpdesk_ticket_datasets", "kind": "zenodo_file", **result}
            record_result(
                rows,
                report,
                attr_row(
                    "public_it_helpdesk_ticket_datasets",
                    file_url,
                    metadata.get("metadata", {}).get("license", {}).get("id", "record-specific"),
                    [],
                    result,
                ),
                item,
                overall,
            )
            tqdm.write(f"public_it_helpdesk_ticket_datasets: {result['status']} Zenodo {record_id} {filename}")


def write_report(report_path: Path, report: list[dict[str, Any]]) -> None:
    ok = [item for item in report if item["status"] in SUCCESS_STATUSES]
    failed = [item for item in report if item["status"] not in SUCCESS_STATUSES]
    by_source: dict[str, dict[str, int]] = {}
    for item in report:
        stats = by_source.setdefault(item["source_id"], {"successful_or_existing": 0, "failed": 0, "bytes": 0})
        key = "successful_or_existing" if item["status"] in SUCCESS_STATUSES else "failed"
        stats[key] = stats.get(key, 0) + 1
        stats["bytes"] += int(item.get("bytes") or 0)
    lines = [
        "# Broad Data Acquisition Wave Report",
        "",
        f"Run id: `{RUN_ID}`",
        "",
        f"- Successful/existing artifacts: {len(ok)}",
        f"- Failed/gated artifacts: {len(failed)}",
        f"- Total bytes downloaded: {sum(int(item.get('bytes') or 0) for item in ok):,}",
        "",
        "## By Source",
        "",
        "| Source | Successful/Existing | Failed | Bytes |",
        "|---|---:|---:|---:|",
    ]
    for source_id, stats in sorted(by_source.items()):
        lines.append(f"| `{source_id}` | {stats.get('successful_or_existing', 0)} | {stats.get('failed', 0)} | {stats['bytes']:,} |")
    if failed:
        lines.extend(["", "## Failed Or Gated", ""])
        for item in failed:
            lines.append(f"- `{item['source_id']}` {item.get('kind', '')}: {item['url']} -> {item.get('error', item['status'])}")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-large", action="store_true", help="Include large NVD CPE-match feed.")
    parser.add_argument("--skip-doc-pages", action="store_true")
    parser.add_argument("--skip-feeds", action="store_true")
    parser.add_argument("--skip-repo-zips", action="store_true")
    parser.add_argument("--skip-hf", action="store_true")
    parser.add_argument("--skip-zenodo", action="store_true")
    parser.add_argument("--run-id", help="Resume or continue a specific run id directory.")
    parser.add_argument("--force", action="store_true", help="Redownload artifacts even if non-empty files already exist.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global LIVE_TOTAL, RUN_ID
    if args.run_id:
        RUN_ID = args.run_id
    LIVE_TOTAL = count_expected_artifacts(args)
    load_dotenv(PROJECT_ROOT / ".env")
    registry_path = PROJECT_ROOT / "data" / "source_registry.json"
    ensure_source_families(registry_path)

    if args.dry_run:
        print(f"run_id={RUN_ID}")
        print(f"expected_artifacts={LIVE_TOTAL}")
        print(f"doc_pages={0 if args.skip_doc_pages else sum(len(pages) for _, _, pages in DOC_PAGES + API_PAGES)}")
        print(f"feed_files={0 if args.skip_feeds else len(nvd_urls(args.include_large)) + 2}")
        print(f"repo_zips={0 if args.skip_repo_zips else len(REPO_ZIPS)}")
        print(f"hf_datasets={0 if args.skip_hf else len(HF_DATASETS)}")
        print(f"zenodo_records={0 if args.skip_zenodo else len(ZENODO_RECORDS)}")
        return 0

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    rows: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    show_progress = not args.no_progress
    reset_live_progress_files()

    with tqdm(total=LIVE_TOTAL, desc="broad wave artifacts", unit="artifact", disable=not show_progress) as overall:
        if not args.skip_doc_pages:
            acquire_doc_pages(session, rows, report, overall, force=args.force, show_progress=show_progress)
        if not args.skip_feeds:
            acquire_feed_files(
                session,
                rows,
                report,
                overall,
                include_large=args.include_large,
                force=args.force,
                show_progress=show_progress,
            )
        if not args.skip_repo_zips:
            acquire_repo_zips(session, rows, report, overall, force=args.force, show_progress=show_progress)
        if not args.skip_hf:
            acquire_hf_datasets(rows, report, overall)
        if not args.skip_zenodo:
            acquire_zenodo(session, rows, report, overall, force=args.force, show_progress=show_progress)

    attribution_path = PROJECT_ROOT / "data" / "attribution" / "broad_data_wave" / f"{RUN_ID}_attribution.jsonl"
    manifest_path = PROJECT_ROOT / "data" / "raw" / "broad_data_wave" / RUN_ID / "run_manifest.json"
    report_path = PROJECT_ROOT / "data" / "processed" / "broad_data_wave" / RUN_ID / "acquisition_report.md"
    write_jsonl(attribution_path, rows)
    write_json(manifest_path, {"run_id": RUN_ID, "artifact_count": len(report), "artifacts": report})
    write_report(report_path, report)

    for source_id in sorted({row["source_family_id"] for row in rows}):
        source_rows = [row for row in rows if row["source_family_id"] == source_id]
        append_registry_run(
            registry_path,
            {
                "run_id": RUN_ID,
                "source_family_id": source_id,
                "status": "raw_acquired" if any(row["status"] in SUCCESS_STATUSES for row in source_rows) else "attempted_failed",
                "raw_dir": str((PROJECT_ROOT / "data" / "raw" / source_id / RUN_ID).relative_to(PROJECT_ROOT)),
                "attribution_path": str(attribution_path.relative_to(PROJECT_ROOT)),
                "artifact_count": len(source_rows),
                "successful_artifact_count": sum(row["status"] in SUCCESS_STATUSES for row in source_rows),
                "report_path": str(report_path.relative_to(PROJECT_ROOT)),
            },
        )

    empty_dirs = [path for path in (PROJECT_ROOT / "data" / "raw").rglob("*") if path.is_dir() and not any(path.iterdir())]
    for path in sorted(empty_dirs, key=lambda p: len(p.parts), reverse=True):
        shutil.rmtree(path, ignore_errors=True)

    print(f"run_id={RUN_ID}")
    print(f"manifest={manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"attribution={attribution_path.relative_to(PROJECT_ROOT)}")
    print(f"report={report_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
