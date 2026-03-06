import pytest
from prompts.quality_gates import check_quality

def test_quality_gate_passes():
    content = """
    Purpose: This SOP defines the procedure for onboarding new employees.
    Scope: Applies to all new employees joining the organization.
    Procedure: Step 1 - Fill the form. Step 2 - Submit documents. Step 3 - Attend orientation.
    Responsibilities: HR is responsible for execution and monitoring.
    """ * 5
    passed, reason = check_quality(content, "sop")
    assert passed, f"Should pass: {reason}"

def test_quality_gate_fails_short():
    content = "This is too short."
    passed, reason = check_quality(content, "sop")
    assert not passed
    assert "short" in reason.lower()

def test_quality_gate_fails_missing_sections():
    content = " ".join(["word"] * 200)
    passed, reason = check_quality(content, "policy")
    assert not passed
    assert "missing" in reason.lower()
