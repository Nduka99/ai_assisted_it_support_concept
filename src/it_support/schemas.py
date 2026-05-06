"""Shared data schemas for the IT-SUPPORT proof of concept."""

from __future__ import annotations

from dataclasses import dataclass, field


DOMAIN_LABELS = [
    "hardware",
    "os_kernel_drivers",
    "application_software",
    "network_connectivity",
    "identity_access_accounts",
    "storage_data_backup",
    "security_malware",
    "firmware_bios_uefi",
]


SAFETY_SIGNALS = [
    "no_fault_likely",
    "low_confidence",
    "needs_human",
    "needs_vendor",
    "possible_security_incident",
]


@dataclass
class SourceMetadata:
    source: str
    license: str
    commercial_status: str
    source_url: str | None = None
    attribution_required: bool = False
    share_alike_required: bool = False


@dataclass
class CorpusRecord:
    id: str
    title: str
    body_text: str
    metadata: SourceMetadata
    answer_text: str | None = None
    domain_labels: list[str] = field(default_factory=list)
    no_fault: bool = False
    quality_score: float = 0.0
    solution_score: float = 0.0
