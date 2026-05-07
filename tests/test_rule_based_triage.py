from __future__ import annotations

from it_support.triage import predict_case, prediction_to_record


def make_case(title: str, text: str = "", tags: list[str] | None = None) -> dict[str, object]:
    return {
        "case_id": "test",
        "title": title,
        "question_text": text,
        "question_tags": tags or [],
        "query_tags": [],
        "expected_domains": [],
        "expected_primary_domain": None,
    }


def test_predicts_network_from_text() -> None:
    prediction = predict_case(make_case("VPN disconnects and DNS fails on Wi-Fi"))
    assert prediction.primary_domain == "network_connectivity"


def test_predicts_firmware_escalation_from_bios_text() -> None:
    prediction = predict_case(make_case("BIOS update failed and laptop may be bricked"))
    assert prediction.expected_behavior == "structured_firmware_escalation"
    assert prediction.safety_signals["firmware_escalation_required"] is True


def test_predicts_security_triage_from_ransomware_text() -> None:
    prediction = predict_case(make_case("Possible ransomware encrypted shared files"))
    assert prediction.expected_behavior == "security_triage_or_escalation_after_filter"
    assert prediction.primary_domain == "security_malware"


def test_predicts_security_triage_from_visible_cache_poisoning_text() -> None:
    prediction = predict_case(make_case("Understanding DNS cache poisoning"))
    assert prediction.expected_behavior == "security_triage_or_escalation_after_filter"


def test_predicts_security_triage_from_visible_malicious_text() -> None:
    prediction = predict_case(make_case("OpenSSH server failed after malicious update"))
    assert prediction.expected_behavior == "security_triage_or_escalation_after_filter"


def test_predicts_security_triage_from_visible_vulnerability_text() -> None:
    prediction = predict_case(make_case("Certificate Services vulnerability on server"))
    assert prediction.expected_behavior == "security_triage_or_escalation_after_filter"


def test_predicts_firmware_escalation_from_visible_efi_secure_boot_text() -> None:
    prediction = predict_case(make_case("Windows upgrade changed EFI and Secure Boot settings"))
    assert prediction.expected_behavior == "structured_firmware_escalation"


def test_prediction_record_keeps_eval_projection_question_only() -> None:
    case = make_case("Printer offline after Windows update", tags=["printer"])
    prediction = predict_case(case, include_source_tags=True)
    record = prediction_to_record(case, prediction)
    assert "accepted_answer" not in record
    assert "top_answer" not in record
