from __future__ import annotations

import json
import os
import sys

import pytest

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
sys.path.insert(0, SRC)

from compliance_report import (  # noqa: E402
    ALLOWED_LABELS,
    STABILITY_LIMITATION,
    append_records,
    make_audit_record,
    validate_record,
)


def _record(sample_id="sample-1"):
    return make_audit_record(
        sample_id=sample_id,
        typology_label="structuring",
        second_prototype="layering",
        raw_distance=0.5,
        second_raw_distance=1.5,
        motif="eşik-kenarı kümesi",
        model_version="quann_hybrid",
        predicted_class="structuring",
    )


def test_compliance_record_schema_is_bounded_and_controlled():
    record = _record()
    validate_record(record)
    assert record["rationale"].strip()
    assert 0.0 <= record["prototype"]["distance"] <= 1.0
    assert record["typology_label"] in ALLOWED_LABELS
    assert record["model_version"] == "quann_hybrid"
    with pytest.raises(ValueError, match="controlled vocabulary"):
        invalid = dict(record, typology_label="made_up_typology")
        validate_record(invalid)


def test_audit_trail_is_append_only_across_runs(tmp_path):
    path = tmp_path / "audit.jsonl"
    append_records([_record("first")], str(path))
    first_contents = path.read_text(encoding="utf-8")
    append_records([_record("second")], str(path))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert lines[0] == first_contents.rstrip("\n")
    assert [json.loads(line)["sample_id"] for line in lines] == ["first", "second"]


def test_report_uses_name_scope_and_discloses_stability_limit():
    record = _record()
    assert record["explanation_scope"] == "name-scope"
    assert record["stability_limitation"] == STABILITY_LIMITATION
    assert "coordinates" in record["stability_limitation"]
    assert "attribution magnitudes" in record["stability_limitation"]
