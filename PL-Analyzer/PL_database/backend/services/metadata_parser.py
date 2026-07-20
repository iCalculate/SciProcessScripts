from __future__ import annotations

import hashlib
import re
from pathlib import Path


FILENAME_PATTERNS = {
    "laser_wavelength": re.compile(r"(?P<value>\d{3,4})\s*nm", re.IGNORECASE),
    "laser_power": re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*mW", re.IGNORECASE),
    "integration_time": re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*(?:ms|s)", re.IGNORECASE),
    "grating": re.compile(r"(?P<value>\d{2,4})\s*g", re.IGNORECASE),
    "objective": re.compile(r"(?P<value>\d{1,3})\s*x", re.IGNORECASE),
}

COMMON_LASER_LINES_NM = (325, 355, 405, 442, 457, 473, 488, 514, 532, 561, 594, 633, 660, 785, 830, 1064)


def build_spectrum_id(source_wip: str, source_tree_path: str, index: int = 0) -> str:
    digest = hashlib.sha1(
        f"{Path(source_wip).as_posix()}|{source_tree_path}|{index}".encode("utf-8")
    ).hexdigest()
    return f"plspec-{digest[:16]}"


def build_media_id(source_wip: str, source_tree_path: str) -> str:
    digest = hashlib.sha1(
        f"{Path(source_wip).as_posix()}|{source_tree_path}".encode("utf-8")
    ).hexdigest()
    return f"plmedia-{digest[:16]}"


def normalize_dataset_tree_path(source_tree_path: str) -> str:
    text = str(source_tree_path).strip()
    return re.sub(r"/trace-\d+$", "", text)


def build_dataset_id(source_wip: str, source_tree_path: str) -> str:
    normalized_tree = normalize_dataset_tree_path(source_tree_path)
    digest = hashlib.sha1(
        f"{Path(source_wip).as_posix()}|{normalized_tree}".encode("utf-8")
    ).hexdigest()
    return f"plset-{digest[:16]}"


def infer_dataset_label_from_tree_path(source_tree_path: str) -> str | None:
    normalized_tree = normalize_dataset_tree_path(source_tree_path)
    parts = [part for part in normalized_tree.split("/") if part]
    if not parts:
        return None
    return parts[-1]


def is_mock_source(source_path: str) -> bool:
    text = strip_wrapped_quotes(str(source_path).strip())
    if not text:
        return False
    lowered = text.lower()
    if lowered.startswith("mock://"):
        return True
    if re.match(r"^[a-z]:[\\/]", text):
        return False
    return lowered.startswith("mock")


def normalize_source_path(source_path: str) -> str:
    text = strip_wrapped_quotes(str(source_path).strip())
    if not text or is_mock_source(text) or "://" in text:
        return text
    return str(Path(text).expanduser())


def strip_wrapped_quotes(value: str) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        return text[1:-1].strip()
    return text


def infer_source_label_from_path(source_path: str) -> str | None:
    text = Path(source_path).stem.strip()
    return text or None


def infer_legacy_source_label_from_path(source_path: str) -> str | None:
    text = Path(source_path).stem
    parts = [part for part in re.split(r"[_\-\s]+", text) if part]
    if len(parts) > 1:
        return parts[1]
    if parts:
        return parts[0]
    return None


def infer_belonging_from_path(source_path: str) -> str | None:
    text = strip_wrapped_quotes(str(source_path).strip())
    if not text:
        return None

    parts = [part.strip() for part in re.split(r"[\\/]+", text) if part.strip()]
    for index, part in enumerate(parts[:-1]):
        if part.lower() not in {"user", "nexstrom"}:
            continue
        belonging = parts[index + 1].strip()
        if belonging:
            return belonging
    return None


def parse_metadata_from_path(source_path: str) -> dict[str, str]:
    text = Path(source_path).stem
    lowered = text.lower()
    metadata: dict[str, str] = {}
    parts = [part for part in re.split(r"[_\-\s]+", text) if part]

    if parts:
        metadata["sample_id"] = parts[0]
    source_label = infer_source_label_from_path(source_path)
    if source_label:
        metadata["source"] = source_label
    belonging = infer_belonging_from_path(source_path)
    if belonging:
        metadata["belonging"] = belonging

    for key, pattern in FILENAME_PATTERNS.items():
        match = pattern.search(text)
        if match:
            metadata[key] = match.group("value")

    if "sio2" in lowered or "oxide" in lowered:
        metadata["substrate"] = "SiO2"
    elif "sapphire" in lowered:
        metadata["substrate"] = "sapphire"
    elif "glass" in lowered:
        metadata["substrate"] = "glass"

    if "device" in lowered:
        metadata["device_id"] = text

    return metadata


def infer_spectrum_type(source_wip: str, tree_path: str) -> str:
    combined = f"{source_wip} {tree_path}".lower()
    if "raman" in combined:
        return "Raman"
    if "photoluminescence" in combined or re.search(r"\bpl\b", combined):
        return "PL"
    return "unknown"


def infer_spectrum_type_from_grating(grating: object) -> str | None:
    text = str(grating or "").strip()
    if not text:
        return None
    if re.search(r"\bG3\b", text, re.IGNORECASE):
        return "Raman"
    if re.search(r"\bG(?:1|2)\b", text, re.IGNORECASE):
        return "PL"
    return None


def infer_spectrum_type_with_context(
    x_axis: list[float] | tuple[float, ...] | object,
    x_axis_unit: str | None = None,
    *,
    grating: object = None,
) -> str:
    spectrum_type_from_grating = infer_spectrum_type_from_grating(grating)
    if spectrum_type_from_grating is not None:
        return spectrum_type_from_grating
    return infer_spectrum_type_from_axis(x_axis, x_axis_unit)


def infer_spectrum_type_from_axis(
    x_axis: list[float] | tuple[float, ...] | object,
    x_axis_unit: str | None = None,
) -> str:
    try:
        values = [float(value) for value in x_axis]
    except Exception:
        return "unknown"
    if not values:
        return "unknown"

    unit = (x_axis_unit or "").strip().lower()
    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum

    if any(token in unit for token in ("cm^-1", "cm-1", "1/cm", "raman")):
        return "Raman"
    if any(token in unit for token in ("ev", "mev")):
        return "PL"
    if any(token in unit for token in ("nm", "nanometer", "nanometre")):
        return "Raman" if infer_raman_excitation_from_nm_axis(values) is not None else "PL"

    if 0 <= minimum and maximum <= 10:
        return "PL"
    if infer_raman_excitation_from_nm_axis(values) is not None:
        return "Raman"
    if 300 <= minimum and maximum <= 1200:
        return "PL"
    if 0 <= minimum and maximum <= 4500 and (minimum < 250 or maximum > 1400 or span > 1000):
        return "Raman"
    return "unknown"


def infer_x_axis_unit(x_axis: list[float] | tuple[float, ...] | object) -> str:
    try:
        values = list(float(value) for value in x_axis)
    except Exception:
        return "unknown"
    if not values:
        return "unknown"
    minimum = min(values)
    maximum = max(values)
    if 0 <= minimum and maximum <= 5:
        return "eV"
    if 200 <= minimum and maximum <= 2000:
        return "nm"
    if 10 <= minimum and maximum <= 5000:
        return "cm^-1"
    return "unknown"


def infer_raman_excitation_from_nm_axis(
    x_axis: list[float] | tuple[float, ...] | object,
) -> float | None:
    try:
        values = [float(value) for value in x_axis]
    except Exception:
        return None
    if not values:
        return None

    minimum = min(values)
    maximum = max(values)
    span = maximum - minimum
    if span <= 0 or span > 120:
        return None

    candidates = sorted(
        (line for line in COMMON_LASER_LINES_NM if minimum - 2 <= line <= minimum + 8),
        key=lambda line: abs(line - minimum),
    )
    for candidate in candidates:
        shifts = [_to_raman_shift(value, candidate) for value in values]
        minimum_shift = min(shifts)
        maximum_shift = max(shifts)
        positive_ratio = sum(1 for value in shifts if value >= -120) / len(shifts)
        if minimum_shift >= -250 and maximum_shift >= 120 and maximum_shift <= 4200 and positive_ratio >= 0.85:
            return float(candidate)
    return None


def _to_raman_shift(wavelength_nm: float, laser_wavelength_nm: float) -> float:
    return (1e7 / laser_wavelength_nm) - (1e7 / wavelength_nm)
