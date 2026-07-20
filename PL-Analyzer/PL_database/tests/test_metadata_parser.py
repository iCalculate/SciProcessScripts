from __future__ import annotations

from backend.services.metadata_parser import (
    infer_legacy_source_label_from_path,
    infer_spectrum_type_from_grating,
    infer_spectrum_type_with_context,
    normalize_source_path,
    parse_metadata_from_path,
)


def test_parse_metadata_from_path_uses_full_file_stem_for_source() -> None:
    metadata = parse_metadata_from_path(r"C:\data\20260127-GrayScale.wip")

    assert metadata["sample_id"] == "20260127"
    assert metadata["source"] == "20260127-GrayScale"


def test_legacy_source_label_helper_preserves_previous_tokenized_behavior() -> None:
    assert infer_legacy_source_label_from_path(r"C:\data\20260127-GrayScale.wip") == "GrayScale"


def test_infer_spectrum_type_from_grating_uses_requested_g_rules() -> None:
    assert infer_spectrum_type_from_grating("G1: 300 g/mm BLZ 500.00 nm") == "PL"
    assert infer_spectrum_type_from_grating("G2 custom") == "PL"
    assert infer_spectrum_type_from_grating("G3: 1800 g/mm BLZ 500.00 nm") == "Raman"
    assert infer_spectrum_type_from_grating("unknown grating") is None


def test_infer_spectrum_type_with_context_lets_grating_override_axis_heuristics() -> None:
    assert infer_spectrum_type_with_context([485.0, 486.0, 487.0], "nm", grating="G3: 1800 g/mm") == "Raman"
    assert infer_spectrum_type_with_context([100.0, 200.0, 300.0], "cm^-1", grating="G1: 300 g/mm") == "PL"


def test_normalize_source_path_strips_wrapped_quotes_for_unc_paths() -> None:
    assert normalize_source_path(r'"\\server\share\team\demo.wip"') == r"\\server\share\team\demo.wip"
