"""Deterministic rule-based triage and safety baseline.

This is a fast, auditable baseline for the first routing/safety evaluation gate.
It intentionally uses transparent keyword/tag rules instead of models. The goal
is not to be clever; it is to create a reproducible floor for later classifiers.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from it_support.schemas import DOMAIN_LABELS


DOMAIN_PRIORITY = [
    "security_malware",
    "firmware_bios_uefi",
    "hardware",
    "os_kernel_drivers",
    "network_connectivity",
    "identity_access_accounts",
    "storage_data_backup",
    "application_software",
]


DOMAIN_KEYWORDS: dict[str, dict[str, float]] = {
    "hardware": {
        "printer": 4.0,
        "printing": 3.5,
        "nvidia": 3.0,
        "graphics card": 3.0,
        "gpu": 3.0,
        "display": 2.5,
        "screen": 2.0,
        "monitor": 2.5,
        "motherboard": 3.5,
        "keyboard": 2.5,
        "mouse": 2.0,
        "battery": 3.0,
        "fan": 2.0,
        "thermal": 3.0,
        "overheating": 3.0,
        "usb": 2.0,
        "bluetooth": 2.0,
        "realtek": 2.0,
        "webcam": 2.5,
        "scanner": 2.5,
    },
    "os_kernel_drivers": {
        "driver": 4.0,
        "drivers": 4.0,
        "kernel": 4.0,
        "systemd": 4.0,
        "grub": 4.0,
        "grub2": 4.0,
        "boot": 2.5,
        "bootloader": 2.5,
        "bsod": 4.0,
        "blue screen": 4.0,
        "windows update": 3.0,
        "windows 10": 2.5,
        "windows 11": 2.5,
        "windows server": 2.5,
        "linux": 2.0,
        "ubuntu": 2.0,
        "mount": 2.5,
        "service": 2.0,
        "daemon": 2.0,
        "registry": 3.0,
        "xorg": 2.5,
        "nouveau": 2.5,
    },
    "application_software": {
        "application": 3.0,
        "app": 2.0,
        "office": 3.0,
        "outlook": 4.0,
        "browser": 2.5,
        "chrome": 3.0,
        "firefox": 3.0,
        "edge": 2.5,
        "package": 2.0,
        "install": 1.5,
        "msi": 3.0,
        "wmi": 2.0,
        "shortcut": 2.0,
        "notification": 2.0,
        "widget": 2.0,
        "crash": 2.5,
        "hang": 2.0,
    },
    "network_connectivity": {
        "network": 4.0,
        "networking": 4.0,
        "wifi": 4.0,
        "wi-fi": 4.0,
        "wireless": 4.0,
        "vpn": 4.5,
        "dns": 4.0,
        "dhcp": 4.0,
        "routing": 4.0,
        "router": 3.0,
        "gateway": 3.0,
        "subnet": 3.0,
        "firewall": 2.5,
        "latency": 3.0,
        "packet": 2.5,
        "port": 2.0,
        "openvpn": 4.0,
        "ipsec": 4.0,
        "cisco": 2.5,
        "ethernet": 3.0,
        "internet": 2.5,
        "hotspot": 3.0,
        "access point": 3.0,
    },
    "identity_access_accounts": {
        "active directory": 4.5,
        "active-directory": 4.5,
        "domain controller": 4.0,
        "domain-controller": 4.0,
        "group policy": 4.0,
        "group-policy": 4.0,
        "permission": 3.5,
        "permissions": 3.5,
        "login": 3.0,
        "logon": 3.0,
        "account": 3.0,
        "user": 1.5,
        "password": 3.0,
        "mfa": 3.5,
        "sso": 3.0,
        "ssh key": 3.0,
        "access denied": 4.0,
        "certificate": 2.5,
        "pki": 2.5,
        "authentication": 3.0,
        "authorization": 3.0,
    },
    "storage_data_backup": {
        "backup": 4.5,
        "restore": 4.0,
        "disk": 3.5,
        "hard drive": 4.0,
        "hard-drive": 4.0,
        "ssd": 4.0,
        "smart": 4.0,
        "raid": 4.0,
        "partition": 3.0,
        "partitioning": 3.0,
        "filesystem": 3.0,
        "file system": 3.0,
        "fsck": 3.5,
        "mount": 2.5,
        "data loss": 4.0,
        "recover": 3.0,
        "recovery": 3.0,
        "bad sector": 4.0,
        "bad-sectors": 4.0,
        "sql backup": 3.0,
        "mysqldump": 3.0,
    },
    "security_malware": {
        "malware": 5.0,
        "ransomware": 5.0,
        "phishing": 5.0,
        "virus": 4.5,
        "trojan": 4.5,
        "rootkit": 4.5,
        "suspicious": 3.5,
        "compromise": 4.0,
        "hacked": 4.0,
        "exploit": 4.0,
        "antivirus": 3.5,
        "endpoint": 2.5,
        "credential": 3.0,
        "encrypt": 2.5,
        "encryption": 2.0,
        "firewall": 1.5,
        "security": 2.0,
        "password": 1.0,
        "open port": 2.0,
        "ports open": 2.0,
    },
    "firmware_bios_uefi": {
        "firmware": 5.0,
        "bios": 5.0,
        "uefi": 5.0,
        "efi": 3.0,
        "secure boot": 3.5,
        "flash bios": 5.0,
        "bios update": 5.0,
        "firmware update": 5.0,
        "bricked": 4.5,
        "rom": 2.5,
        "cmos": 3.0,
        "nvram": 2.5,
        "boot order": 2.5,
        "gpt": 2.0,
        "mbr": 2.0,
    },
}


TAG_ALIASES: dict[str, str] = {
    "domain-name-system": "dns",
    "wireless-access-point": "access point",
    "windows-firewall": "firewall",
    "boot-repair": "boot repair",
    "software-installation": "software installation",
    "hybrid-graphics": "graphics",
    "hard-drive": "hard drive",
    "bad-sectors": "bad sector",
}


SAFETY_KEYWORDS: dict[str, dict[str, float]] = {
    "possible_security_incident": {
        "malware": 5.0,
        "malicious": 4.0,
        "ransomware": 5.0,
        "phishing": 5.0,
        "virus": 4.0,
        "trojan": 4.0,
        "rootkit": 4.0,
        "hacked": 4.0,
        "compromised": 4.0,
        "suspicious": 3.5,
        "vulnerability": 4.0,
        "vulnerable": 4.0,
        "exploit": 4.0,
        "credential": 3.0,
        "password leak": 4.0,
        "open port": 2.0,
        "ports open": 2.0,
        "unauthorized": 3.0,
        "cache poisoning": 4.0,
        "cache-poisoning": 4.0,
    },
    "firmware_escalation_required": {
        "firmware": 5.0,
        "bios": 5.0,
        "uefi": 5.0,
        "efi": 5.0,
        "secure boot": 5.0,
        "flash bios": 5.0,
        "bios update": 5.0,
        "firmware update": 5.0,
        "bricked": 4.5,
        "cmos": 3.0,
        "nvram": 3.0,
        "eeprom": 4.0,
    },
    "data_loss_risk": {
        "data loss": 5.0,
        "backup": 3.0,
        "restore": 3.0,
        "recover": 3.0,
        "recovery": 3.0,
        "format": 3.5,
        "partition": 3.0,
        "disk failure": 4.0,
        "smart": 4.0,
        "bad sector": 4.0,
        "fsck": 2.5,
    },
    "credential_secret_risk": {
        "password": 4.0,
        "secret": 4.0,
        "token": 4.0,
        "api key": 4.0,
        "private key": 4.0,
        "credential": 4.0,
        "mfa": 3.5,
    },
}


@dataclass(frozen=True)
class RuleMatch:
    label: str
    score: float
    evidence: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TriagePrediction:
    domains: list[RuleMatch]
    safety_signals: dict[str, bool]
    safety_scores: dict[str, float]
    expected_behavior: str
    profile: str

    @property
    def primary_domain(self) -> str | None:
        return self.domains[0].label if self.domains else None


def normalize_text(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9+#./_-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def token_contains(text: str, phrase: str) -> bool:
    phrase = normalize_text(phrase)
    if " " in phrase:
        return phrase in text
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


def case_text(case: dict[str, Any], *, include_source_tags: bool) -> str:
    parts = [case.get("title", ""), case.get("question_text", "")]
    if include_source_tags:
        tags = list(case.get("question_tags") or []) + list(case.get("query_tags") or [])
        expanded_tags = [TAG_ALIASES.get(tag, tag) for tag in tags]
        parts.append(" ".join(tags + expanded_tags))
    return normalize_text(" ".join(str(part) for part in parts if part))


def score_rules(text: str, rules: dict[str, dict[str, float]]) -> dict[str, RuleMatch]:
    matches: dict[str, RuleMatch] = {}
    for label, keyword_weights in rules.items():
        score = 0.0
        evidence = []
        for keyword, weight in keyword_weights.items():
            if token_contains(text, keyword):
                score += weight
                evidence.append(keyword)
        if score:
            matches[label] = RuleMatch(label=label, score=score, evidence=evidence)
    return matches


def rank_domain_matches(matches: dict[str, RuleMatch]) -> list[RuleMatch]:
    priority = {label: index for index, label in enumerate(DOMAIN_PRIORITY)}
    return sorted(
        matches.values(),
        key=lambda item: (-item.score, priority.get(item.label, 99), item.label),
    )


def infer_domains(text: str, *, max_domains: int = 3) -> list[RuleMatch]:
    matches = score_rules(text, DOMAIN_KEYWORDS)
    if not matches:
        return [RuleMatch(label="application_software", score=0.1, evidence=["default"])]

    ranked = rank_domain_matches(matches)
    top_score = ranked[0].score
    threshold = max(2.0, top_score * 0.35)
    selected = [match for match in ranked if match.score >= threshold]
    return selected[:max_domains]


def infer_safety(text: str) -> tuple[dict[str, bool], dict[str, float]]:
    matches = score_rules(text, SAFETY_KEYWORDS)
    scores = {label: match.score for label, match in matches.items()}
    signals = {
        "possible_security_incident": scores.get("possible_security_incident", 0.0) >= 4.0,
        "firmware_escalation_required": scores.get("firmware_escalation_required", 0.0) >= 5.0,
        "data_loss_risk": scores.get("data_loss_risk", 0.0) >= 4.0,
        "credential_secret_risk": scores.get("credential_secret_risk", 0.0) >= 4.0,
    }
    signals["needs_human_review"] = (
        signals["possible_security_incident"]
        or signals["firmware_escalation_required"]
        or signals["data_loss_risk"]
        or signals["credential_secret_risk"]
    )
    return signals, scores


def expected_behavior_from_signals(signals: dict[str, bool]) -> str:
    if signals.get("firmware_escalation_required"):
        return "structured_firmware_escalation"
    if signals.get("possible_security_incident"):
        return "security_triage_or_escalation_after_filter"
    if signals.get("data_loss_risk"):
        return "data_preservation_review"
    if signals.get("credential_secret_risk"):
        return "credential_or_access_review"
    return "route_to_expected_domains"


def confidence_from_score(score: float) -> float:
    return round(1.0 - math.exp(-score / 10.0), 4)


def predict_case(
    case: dict[str, Any],
    *,
    include_source_tags: bool = False,
    max_domains: int = 3,
) -> TriagePrediction:
    """Predict domains and safety behavior for one question-only eval case."""

    profile = "metadata_assisted" if include_source_tags else "text_only"
    text = case_text(case, include_source_tags=include_source_tags)
    domains = infer_domains(text, max_domains=max_domains)
    signals, safety_scores = infer_safety(text)
    return TriagePrediction(
        domains=domains,
        safety_signals=signals,
        safety_scores=safety_scores,
        expected_behavior=expected_behavior_from_signals(signals),
        profile=profile,
    )


def prediction_to_record(case: dict[str, Any], prediction: TriagePrediction) -> dict[str, Any]:
    return {
        "case_id": case.get("case_id"),
        "split": case.get("split"),
        "profile": prediction.profile,
        "title": case.get("title", ""),
        "expected_primary_domain": case.get("expected_primary_domain"),
        "expected_domains": case.get("expected_domains", []),
        "predicted_primary_domain": prediction.primary_domain,
        "predicted_domains": [
            {
                "label": match.label,
                "score": match.score,
                "confidence": confidence_from_score(match.score),
                "evidence": match.evidence,
            }
            for match in prediction.domains
        ],
        "expected_behavior": case.get("expected_behavior"),
        "predicted_behavior": prediction.expected_behavior,
        "safety_signals": prediction.safety_signals,
        "safety_scores": prediction.safety_scores,
        "source_url": case.get("source_url"),
        "question_tags": case.get("question_tags", []),
        "query_tags": case.get("query_tags", []),
    }


def known_domain_labels() -> list[str]:
    return list(DOMAIN_LABELS)
