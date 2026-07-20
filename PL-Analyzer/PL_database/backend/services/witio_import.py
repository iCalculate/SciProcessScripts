from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .media_assets import PHOTO_IMAGE
from .metadata_parser import infer_spectrum_type_with_context


POINT_SPECTRUM = "point_spectrum"
LINE_SCAN = "line_scan"
AREA_MAP = "area_map"
SERIES_SCAN = "series_scan"

SUPPORTED_ACQUISITION_MODES = {
    POINT_SPECTRUM,
    LINE_SCAN,
    AREA_MAP,
    SERIES_SCAN,
}

_SKIP_CAPTION_KEYWORDS = ("mask",)
_MODE_SEGMENTS = {
    POINT_SPECTRUM: "point",
    LINE_SCAN: "line",
    AREA_MAP: "image",
    SERIES_SCAN: "series",
}
_PHOTO_ENTRY_CHANNELS = {
    "TDImage": 1,
    "TDBitmap": 3,
}

_TRACE_PARAM_GUID_FIELD_MAP = {
    "{50786913-FDEE-4C98-A3D2-69CEAD859087}": "system_id",
    "{1AFAD5D3-7519-4017-9402-169E14483ACB}": "configuration_name",
    "{3BD5B7C3-02B3-4226-8541-B4FA098F9FAA}": "duration_s",
    "{C59F0865-BCA6-42CC-970B-E78863813092}": "laser_wavelength_nm",
    "{E1353512-9D5D-4B42-85B0-563E1D2E0922}": "laser_power_in_fiber_mw",
    "{DE90D5DF-6355-4003-8C30-69F4ECA394D6}": "laser_power_mw",
    "{C7BF2E9E-4588-4FE7-A661-8BC5F9ABBEA9}": "integration_time_s",
    "{09449FFD-F53F-48FE-A687-D17A866CDB64}": "accumulations",
    "{16EDE678-FAF2-4B8D-B8E4-D7F8805438A0}": "objective_name",
    "{B4B5BA2D-BCF0-4DF3-AFE8-09808536C16B}": "objective_magnification",
    "{B9B7E353-ACAD-434A-B02E-65260A793A76}": "is_lambda_4_coupled",
    "{0894921D-94F8-4B6A-92E0-B1F8F9C875E7}": "sample_position_x_um",
    "{459B4B02-6301-4309-AB36-122A31ACF47B}": "sample_position_y_um",
    "{FD47BC57-3E4E-48B3-BF05-CD87CA53540E}": "sample_position_z_um",
    "{8DF0A965-92FB-478B-BA56-9F4BCFAB9EFE}": "spectrograph_name",
    "{D4FB7097-4A2B-48FB-8619-27FA848640C1}": "spectrograph_serial_number",
    "{70C15292-3849-485A-A19C-1C63BA2E737A}": "grating",
    "{4A1C85B9-F3E1-4D07-936D-D80C3E68F86F}": "center_wavelength_nm",
    "{75B3D31D-985E-4622-84E7-D07229299AA3}": "camera_name",
    "{41B6F20B-F543-4221-BCE0-FF70518D61A3}": "camera_serial_number",
    "{144EA40E-6F56-4321-9575-E08F11CD9DF8}": "camera_exposure_time_s",
    "{D3C1554D-27F0-414C-9E26-0A58EA8C3DB7}": "camera_cycle_time_s",
    "{5751D616-EEB4-4778-9BFC-21F7E98A916C}": "camera_readout_mode_guid",
    "{4BF5EC32-10B7-484F-8E8E-D1B6AF14182D}": "camera_single_track_range",
    "{F555CD8E-1549-4A19-9215-DFE1FAE4543A}": "camera_track_height_px",
    "{57946D40-7278-42C9-9F0F-96E3F2F04BA0}": "camera_vertical_shift_speed_us",
    "{979250FF-18FC-4E38-99AE-ED8B40F16A93}": "camera_horizontal_shift_speed_mhz",
    "{577AED5A-C75E-4DF0-A472-A8A9B4C00152}": "camera_pre_amplifier_gain",
    "{499DAAD1-793E-45DC-AE0E-5010C2839F66}": "camera_sensor_temperature_c",
    "{D5B07CC0-BD16-4827-9ABB-EC6D9CE841FF}": "line_start_x_um",
    "{3E30A781-526F-4B8F-ACCA-BEA2A3C47386}": "line_start_y_um",
    "{F98A0D8C-3666-4AD4-8B54-E679D570C83F}": "line_end_x_um",
    "{0A2B614F-E455-45D8-A27F-020F27D5435B}": "line_end_y_um",
    "{E0920DB3-7DC4-49B6-8D24-52913DA42DCF}": "scan_size_x_px",
    "{FCD082AF-F137-4D56-A3A5-E339ABDB2961}": "scan_size_y_px",
    "{9AB8AC43-E3DE-4494-9A7E-65AE9B5A185D}": "scan_span_x_um",
    "{55DDF72F-12EF-4AE2-9093-F92E4CDCA753}": "scan_span_y_um",
    "{E33819CD-8D98-453B-B7A0-4E2854AE3727}": "scan_center_x_um",
    "{BFC9F215-E47E-443A-92C8-FCFE847FECED}": "scan_center_y_um",
    "{57B1AA04-C0FC-45BB-8E23-738653046C53}": "scan_center_z_um",
    "{50613848-5B02-471C-BF75-96C4F03FFB3E}": "scan_rotation_deg",
    "{757E7375-9D48-4446-B1CA-567894E63996}": "scan_cycle_time_s",
    "{75F48F26-DEDB-482E-88B5-4A94438E95D6}": "scan_tilt_x_deg",
    "{3537D883-F44E-41CE-9E65-5A2D7A17AC9D}": "scan_tilt_y_deg",
    "{A1B7BA27-73D1-4BA9-9C4F-E8B48EE0A9BD}": "scan_pattern_guid",
    "{97AC98CA-00A2-4DB1-BE2A-E7542CB2438E}": "line_sampling_mode_guid",
    "{BFDB7906-8B4F-4165-ACC0-E3F02F5CCBB5}": "line_scan_pattern_guid",
}

_CAMERA_READOUT_MODE_LABELS = {
    "{A9769D28-E279-400B-9129-9473586C1A9F}": "Single Track",
}


@dataclass(frozen=True)
class TraceSample:
    file_name: str
    metadata: dict[str, object]
    x_axis: np.ndarray
    intensity: np.ndarray


@dataclass(frozen=True)
class DecodedTrace:
    metadata: dict[str, object]
    x_axis: np.ndarray
    intensity: np.ndarray


@dataclass(frozen=True)
class DatasetSample:
    summary: dict[str, object]
    traces: list[TraceSample]


def _read_tag_scalar(tag: Any) -> object | None:
    if tag is None:
        return None
    try:
        value = tag.scalar()
    except Exception:
        value = getattr(tag, "data", None)
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.item()
        else:
            value = value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _read_tree_text(tag: Any) -> str | None:
    value = _read_tag_scalar(tag)
    if isinstance(value, str) and value:
        return value
    if getattr(tag, "type", None) == 0 and len(getattr(tag, "children", []) or []) == 1:
        child = tag.children[0]
        if getattr(child, "type", None) == 0 and not getattr(child, "children", []):
            return str(child.name)
    return None


def _iter_element_children(tag: Any):
    if tag is None:
        return
    count = _read_tag_scalar(getattr(tag, "find", lambda _name: None)("NumElements"))
    if isinstance(count, (int, float)):
        for index in range(int(count)):
            child = tag.find(f"Element{index}")
            if child is not None:
                yield child
        return
    for child in getattr(tag, "children", []) or []:
        if str(getattr(child, "name", "")).startswith("Element"):
            yield child


def _read_trace_param_value(param_tag: Any) -> object | None:
    start_tag = param_tag.find("Start")
    if start_tag is not None:
        stop_tag = param_tag.find("Stop")
        return {
            "start": int(_read_tag_scalar(start_tag)),
            "stop": int(_read_tag_scalar(stop_tag)) if stop_tag is not None else None,
        }

    for field_name in ("StringValue", "DoubleValue", "IntValue", "BoolValue", "EnumValueGuid"):
        value_tag = param_tag.find(field_name)
        if value_tag is None:
            continue
        value = _read_tag_scalar(value_tag)
        if field_name == "BoolValue" and value is not None:
            return bool(value)
        return value
    return None


def extract_witio_project_metadata(project: Any) -> dict[str, object]:
    system_information = getattr(project, "root", None)
    if system_information is None:
        return {}
    system_information = system_information.find("SystemInformation")
    if system_information is None:
        return {}

    metadata: dict[str, object] = {}
    for field_name, key in (
        ("SystemID", "system_id"),
        ("ApplicationVersions", "application_version"),
        ("ServiceID", "service_id"),
        ("LicenseID", "license_id"),
    ):
        value = _read_tree_text(system_information.find(field_name))
        if value:
            metadata[key] = value
    return metadata


def extract_witio_trace_metadata(trace_element: Any) -> dict[str, object]:
    metadata: dict[str, object] = {}
    trace_guid = _read_tag_scalar(trace_element.find("TraceGuid"))
    if trace_guid:
        metadata["trace_guid"] = trace_guid
    trace_source_guid = _read_tag_scalar(trace_element.find("TraceSourceGuid"))
    if trace_source_guid:
        metadata["trace_source_guid"] = trace_source_guid
    trace_source_version = _read_tag_scalar(trace_element.find("TraceSourceVersion"))
    if trace_source_version is not None:
        metadata["trace_source_version"] = int(trace_source_version)
    creation_utc = _read_tag_scalar(trace_element.find("CreationUTCTime"))
    if creation_utc:
        metadata["trace_creation_time_utc"] = str(creation_utc)
    creation_local = _read_tag_scalar(trace_element.find("CreationLocalTime"))
    if creation_local:
        metadata["trace_creation_time_local"] = str(creation_local)
    user_name = _read_tag_scalar(trace_element.find("UserName"))
    if user_name:
        metadata["trace_user_name"] = str(user_name)

    paramsets = trace_element.find("ParamSets")
    if paramsets is not None:
        metadata["trace_param_set_count"] = sum(1 for _ in _iter_element_children(paramsets))
        for paramset in _iter_element_children(paramsets):
            params = paramset.find("Params")
            if params is None:
                continue
            for param in _iter_element_children(params):
                guid = _read_tag_scalar(param.find("ParamGuid"))
                if not guid:
                    continue
                value = _read_trace_param_value(param)
                if value is None:
                    continue
                field_name = _TRACE_PARAM_GUID_FIELD_MAP.get(str(guid))
                if field_name:
                    metadata[field_name] = value

    if metadata.get("camera_readout_mode_guid"):
        readout_mode_guid = str(metadata["camera_readout_mode_guid"])
        if readout_mode_guid in _CAMERA_READOUT_MODE_LABELS:
            metadata["camera_readout_mode"] = _CAMERA_READOUT_MODE_LABELS[readout_mode_guid]

    if {"sample_position_x_um", "sample_position_y_um", "sample_position_z_um"} <= metadata.keys():
        metadata["sample_position_um"] = {
            "x": float(metadata["sample_position_x_um"]),
            "y": float(metadata["sample_position_y_um"]),
            "z": float(metadata["sample_position_z_um"]),
        }

    if {"line_start_x_um", "line_start_y_um", "line_end_x_um", "line_end_y_um"} <= metadata.keys():
        metadata["line_start_um"] = {
            "x": float(metadata["line_start_x_um"]),
            "y": float(metadata["line_start_y_um"]),
        }
        metadata["line_end_um"] = {
            "x": float(metadata["line_end_x_um"]),
            "y": float(metadata["line_end_y_um"]),
        }

    if "scan_size_x_px" in metadata or "scan_size_y_px" in metadata:
        metadata["scan_size_px"] = {
            "x": int(metadata["scan_size_x_px"]) if metadata.get("scan_size_x_px") is not None else None,
            "y": int(metadata["scan_size_y_px"]) if metadata.get("scan_size_y_px") is not None else None,
        }

    if "scan_span_x_um" in metadata or "scan_span_y_um" in metadata:
        metadata["scan_span_um"] = {
            "x": float(metadata["scan_span_x_um"]) if metadata.get("scan_span_x_um") is not None else None,
            "y": float(metadata["scan_span_y_um"]) if metadata.get("scan_span_y_um") is not None else None,
        }

    if "scan_center_x_um" in metadata or "scan_center_y_um" in metadata or "scan_center_z_um" in metadata:
        metadata["scan_center_um"] = {
            "x": float(metadata["scan_center_x_um"]) if metadata.get("scan_center_x_um") is not None else None,
            "y": float(metadata["scan_center_y_um"]) if metadata.get("scan_center_y_um") is not None else None,
            "z": float(metadata["scan_center_z_um"]) if metadata.get("scan_center_z_um") is not None else None,
        }

    return metadata


def build_witio_trace_metadata_lookup(project: Any) -> dict[str, dict[str, object]]:
    trace_root = getattr(project, "root", None)
    if trace_root is None:
        return {}
    trace_root = trace_root.find("Trace")
    if trace_root is None:
        return {}

    lookup: dict[str, dict[str, object]] = {}
    for trace_element in _iter_element_children(trace_root):
        trace_metadata = extract_witio_trace_metadata(trace_element)
        outputs = trace_element.find("Outputs")
        if outputs is None:
            continue
        for output_element in _iter_element_children(outputs):
            data_guid = _read_tag_scalar(output_element.find("DataGuid"))
            if data_guid:
                lookup[str(data_guid)] = dict(trace_metadata)
    return lookup


def _entry_guid(entry: Any) -> str | None:
    tdata = getattr(entry, "tdata", None)
    if tdata is None:
        return None
    guid = _read_tag_scalar(tdata.find("GUID"))
    return str(guid) if guid else None


def extract_witio_entry_metadata(
    entry: Any,
    *,
    trace_metadata_lookup: dict[str, dict[str, object]] | None = None,
    project_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata = dict(project_metadata or {})
    entry_guid = _entry_guid(entry)
    if entry_guid and trace_metadata_lookup and entry_guid in trace_metadata_lookup:
        metadata.update(trace_metadata_lookup[entry_guid])

    payload = getattr(entry, "payload", None)
    x_interpretation_id = _read_tag_scalar(payload.find("XInterpretationID")) if payload is not None else None
    x_interpretation = entry.project.find_by_id(x_interpretation_id) if x_interpretation_id else None
    if x_interpretation is not None and getattr(x_interpretation, "class_name", None) == "TDSpectralInterpretation":
        interpretation_payload = getattr(x_interpretation, "payload", None)
        excitation = _read_tag_scalar(interpretation_payload.find("ExcitationWaveLength")) if interpretation_payload else None
        if excitation is not None:
            metadata.setdefault("laser_wavelength_nm", float(excitation))

    if metadata.get("trace_user_name") and "measurement_user" not in metadata:
        metadata["measurement_user"] = metadata["trace_user_name"]
    if "laser_wavelength_nm" in metadata:
        metadata.setdefault("laser_wavelength", float(metadata["laser_wavelength_nm"]))
    if "laser_power_mw" in metadata:
        metadata.setdefault("laser_power", float(metadata["laser_power_mw"]))
    if "integration_time_s" in metadata:
        metadata.setdefault("integration_time", float(metadata["integration_time_s"]))
    if metadata.get("objective_name"):
        metadata.setdefault("objective", str(metadata["objective_name"]))

    return metadata


def load_witio() -> Any:
    try:
        import witio
    except ImportError as exc:  # pragma: no cover - exercised in runtime only
        raise RuntimeError("witio is required for Python-native .wip import") from exc
    return witio


def infer_witio_acquisition_mode(
    *,
    size_x: int,
    size_y: int,
    size_graph: int,
    caption: str | None,
) -> str | None:
    lowered = (caption or "").strip().lower()
    if size_graph <= 1:
        return None
    if any(keyword in lowered for keyword in _SKIP_CAPTION_KEYWORDS):
        return None
    if any(token in lowered for token in ("series", "time", "power")) and (size_x > 1 or size_y > 1):
        return SERIES_SCAN
    if size_x == 1 and size_y == 1:
        return POINT_SPECTRUM
    if size_x > 1 and size_y > 1:
        return AREA_MAP
    if size_x > 1 or size_y > 1:
        return LINE_SCAN
    return None


def sanitize_witio_path_segment(value: str | None) -> str:
    text = (value or "").strip()
    if not text:
        return "unnamed"
    text = text.replace("/", "_").replace("\\", "_")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._")
    return text or "unnamed"


def build_witio_source_tree_root(acquisition_mode: str, entry_id: int | None, caption: str | None) -> str:
    segment = _MODE_SEGMENTS.get(acquisition_mode, "graph")
    entry_part = f"{int(entry_id):04d}" if entry_id is not None else "unknown"
    caption_part = sanitize_witio_path_segment(caption)
    return f"/WITioRaw/{segment}/{entry_part}/{caption_part}"


def build_witio_source_tree_path(source_tree_root: str, acquisition_mode: str, trace_index: int) -> str:
    if acquisition_mode == POINT_SPECTRUM:
        return source_tree_root
    return f"{source_tree_root}/trace-{trace_index:04d}"


def build_witio_photo_source_tree_path(entry_id: int | None, caption: str | None) -> str:
    entry_part = f"{int(entry_id):04d}" if entry_id is not None else "unknown"
    caption_part = sanitize_witio_path_segment(caption)
    return f"/WITioRaw/photo/{entry_part}/{caption_part}"


def iter_trace_indices(size_x: int, size_y: int, trace_limit: int | None = None):
    yielded = 0
    for grid_y in range(size_y):
        for grid_x in range(size_x):
            if trace_limit is not None and yielded >= trace_limit:
                return
            yield yielded, grid_x, grid_y
            yielded += 1


def load_witio_position_grid(entry: Any, unit: str | int | None = "um") -> tuple[np.ndarray, np.ndarray]:
    try:
        return entry.position_grid(unit)
    except NotImplementedError as exc:
        if "TDSpaceTransformation" not in str(exc):
            raise
    except ValueError:
        pass

    payload = getattr(entry, "payload", None)
    if payload is None:
        raise ValueError("WITec entry payload is missing")
    size_x = _required_scalar(payload, "SizeX")
    size_y = _required_scalar(payload, "SizeY")
    xi, yi = np.meshgrid(np.arange(size_x, dtype=float), np.arange(size_y, dtype=float), indexing="ij")
    return xi, yi


def describe_witio_graph(
    entry: Any,
    *,
    trace_metadata_lookup: dict[str, dict[str, object]] | None = None,
    project_metadata: dict[str, object] | None = None,
) -> dict[str, object] | None:
    if getattr(entry, "class_name", None) != "TDGraph":
        return None

    payload = getattr(entry, "payload", None)
    if payload is None:
        return None

    size_x = _required_scalar(payload, "SizeX")
    size_y = _required_scalar(payload, "SizeY")
    size_graph = _required_scalar(payload, "SizeGraph")
    caption = str(getattr(entry, "caption", None) or "")
    acquisition_mode = infer_witio_acquisition_mode(
        size_x=size_x,
        size_y=size_y,
        size_graph=size_graph,
        caption=caption,
    )
    if acquisition_mode is None:
        return None

    entry_metadata = extract_witio_entry_metadata(
        entry,
        trace_metadata_lookup=trace_metadata_lookup,
        project_metadata=project_metadata,
    )
    x_axis, x_axis_unit = entry.x_axis(None)
    spectrum_type = infer_spectrum_type_with_context(
        x_axis.tolist(),
        str(x_axis_unit),
        grating=entry_metadata.get("grating"),
    )
    if spectrum_type == "unknown":
        return None

    position_x, position_y = load_witio_position_grid(entry, "um")
    entry_id = _optional_entry_id(entry)
    source_tree_root = build_witio_source_tree_root(acquisition_mode, entry_id, caption)

    return {
        "entry_id": entry_id,
        "caption": caption,
        "acquisition_mode": acquisition_mode,
        "source_tree_root": source_tree_root,
        "spectrum_type": spectrum_type,
        "x_axis_unit": str(x_axis_unit),
        "size_x": size_x,
        "size_y": size_y,
        "size_graph": size_graph,
        "trace_count": size_x * size_y,
        "measurement_time": extract_measurement_time(entry),
        "position_bounds_um": {
            "x_min": float(np.nanmin(position_x)),
            "x_max": float(np.nanmax(position_x)),
            "y_min": float(np.nanmin(position_y)),
            "y_max": float(np.nanmax(position_y)),
        },
        "entry_metadata": entry_metadata,
        "estimated_raw_array_mb": round(
            _estimate_graph_array_nbytes(payload, size_x=size_x, size_y=size_y, size_graph=size_graph) / 1024 / 1024,
            2,
        ),
        "history_count": len(getattr(entry, "history", []) or []),
    }


def describe_witio_photo_entry(
    entry: Any,
    *,
    trace_metadata_lookup: dict[str, dict[str, object]] | None = None,
    project_metadata: dict[str, object] | None = None,
) -> dict[str, object] | None:
    class_name = str(getattr(entry, "class_name", "") or "")
    if class_name not in _PHOTO_ENTRY_CHANNELS:
        return None

    payload = getattr(entry, "payload", None)
    if payload is None:
        return None

    size_x = _required_scalar(payload, "SizeX")
    size_y = _required_scalar(payload, "SizeY")
    caption = str(getattr(entry, "caption", None) or "")
    entry_metadata = extract_witio_entry_metadata(
        entry,
        trace_metadata_lookup=trace_metadata_lookup,
        project_metadata=project_metadata,
    )
    entry_id = _optional_entry_id(entry)
    source_tree_path = build_witio_photo_source_tree_path(entry_id, caption)

    return {
        "entry_id": entry_id,
        "caption": caption,
        "entry_class": class_name,
        "media_kind": PHOTO_IMAGE,
        "source_tree_path": source_tree_path,
        "measurement_time": extract_measurement_time(entry),
        "entry_metadata": entry_metadata,
        "width_px": size_x,
        "height_px": size_y,
        "channel_count": int(_PHOTO_ENTRY_CHANNELS[class_name]),
        "estimated_raw_array_mb": round(
            _estimate_image_array_nbytes(payload, size_x=size_x, size_y=size_y, class_name=class_name) / 1024 / 1024,
            2,
        ),
        "history_count": len(getattr(entry, "history", []) or []),
    }


def should_include_descriptor(descriptor: dict[str, object], import_options: dict[str, object] | None = None) -> bool:
    if not import_options:
        return True
    mode = str(descriptor["acquisition_mode"])
    if mode == POINT_SPECTRUM:
        return bool(import_options.get("include_point_spectra", True))
    if mode == LINE_SCAN:
        return bool(import_options.get("include_line_scans", False))
    if mode == AREA_MAP:
        return bool(import_options.get("include_area_maps", False))
    if mode == SERIES_SCAN:
        return bool(import_options.get("include_series_scans", False))
    return False


def should_include_photo_descriptor(import_options: dict[str, object] | None = None) -> bool:
    if not import_options:
        return False
    return bool(import_options.get("include_photo_images", False))


def select_witio_datasets(
    descriptors: list[dict[str, object]],
    *,
    max_point_datasets: int = 2,
    max_line_datasets: int = 1,
    max_area_datasets: int = 1,
    max_series_datasets: int = 1,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    counters = {
        POINT_SPECTRUM: 0,
        LINE_SCAN: 0,
        AREA_MAP: 0,
        SERIES_SCAN: 0,
    }
    point_signatures: set[tuple[str, str]] = set()

    for descriptor in descriptors:
        mode = str(descriptor["acquisition_mode"])
        if mode == POINT_SPECTRUM:
            signature = (str(descriptor["spectrum_type"]), str(descriptor["x_axis_unit"]))
            if signature in point_signatures:
                continue
            if counters[POINT_SPECTRUM] >= max_point_datasets:
                continue
            point_signatures.add(signature)
            counters[POINT_SPECTRUM] += 1
            selected.append(descriptor)
            continue
        if mode == LINE_SCAN and counters[LINE_SCAN] < max_line_datasets:
            counters[LINE_SCAN] += 1
            selected.append(descriptor)
            continue
        if mode == AREA_MAP and counters[AREA_MAP] < max_area_datasets:
            counters[AREA_MAP] += 1
            selected.append(descriptor)
            continue
        if mode == SERIES_SCAN and counters[SERIES_SCAN] < max_series_datasets:
            counters[SERIES_SCAN] += 1
            selected.append(descriptor)

    return selected


def sample_witio_dataset(
    entry: Any,
    *,
    source_wip: str | Path,
    descriptor: dict[str, object],
    trace_limit: int | None = None,
) -> DatasetSample:
    source_path = Path(source_wip)
    dataset_slug = sanitize_witio_path_segment(str(descriptor["caption"]))
    dataset_summary, x_axis_values, graph_array, position_x, position_y = load_witio_dataset_arrays(
        entry,
        descriptor=descriptor,
    )

    traces: list[TraceSample] = []
    for decoded_trace in iter_witio_dataset_traces(
        descriptor=descriptor,
        source_wip=source_path,
        x_axis_values=x_axis_values,
        graph_array=graph_array,
        position_x=position_x,
        position_y=position_y,
        trace_limit=trace_limit,
    ):
        trace_index = int(decoded_trace.metadata["trace_index"])
        acquisition_mode = str(decoded_trace.metadata["acquisition_mode"])
        file_name = f"{acquisition_mode}__{dataset_slug}__trace-{trace_index:04d}.csv"
        traces.append(
            TraceSample(
                file_name=file_name,
                metadata=decoded_trace.metadata,
                x_axis=decoded_trace.x_axis,
                intensity=decoded_trace.intensity,
            )
        )

    sampled_trace_count = len(traces)
    summary = {
        **dataset_summary,
        "sampled_trace_count": sampled_trace_count,
        "sampled_fraction": (
            round(sampled_trace_count / int(descriptor["trace_count"]), 6)
            if int(descriptor["trace_count"]) > 0
            else 0.0
        ),
    }
    if acquisition_mode == LINE_SCAN and traces:
        last_trace = traces[-1]
        summary["sampled_line_length_um"] = round(float(last_trace.metadata["secondary_axis_value"]), 6)
    return DatasetSample(summary=summary, traces=traces)


def load_witio_dataset_arrays(
    entry: Any,
    *,
    descriptor: dict[str, object],
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    size_x = int(descriptor["size_x"])
    size_y = int(descriptor["size_y"])
    size_graph = int(descriptor["size_graph"])

    x_axis, _ = entry.x_axis(None)
    x_axis_values = np.asarray(x_axis, dtype=float)
    graph_array = entry.array()
    position_x, position_y = load_witio_position_grid(entry, "um")

    if graph_array.shape != (size_x, size_y, size_graph):
        raise ValueError(
            f"Unexpected graph shape for {descriptor['caption']!r}: "
            f"{graph_array.shape} != {(size_x, size_y, size_graph)}"
        )

    dataset_summary = {
        **descriptor,
        "array_dtype": str(graph_array.dtype),
        "actual_array_mb": round(graph_array.nbytes / 1024 / 1024, 2),
    }
    return dataset_summary, x_axis_values, graph_array, position_x, position_y


def iter_witio_dataset_traces(
    *,
    descriptor: dict[str, object],
    source_wip: str | Path,
    x_axis_values: np.ndarray,
    graph_array: np.ndarray,
    position_x: np.ndarray,
    position_y: np.ndarray,
    trace_limit: int | None = None,
):
    source_path = Path(source_wip)
    acquisition_mode = str(descriptor["acquisition_mode"])
    origin_x_um = float(position_x[0, 0])
    origin_y_um = float(position_y[0, 0])

    for trace_index, grid_x, grid_y in iter_trace_indices(int(descriptor["size_x"]), int(descriptor["size_y"]), trace_limit):
        metadata = build_witio_trace_metadata(
            descriptor=descriptor,
            source_wip=source_path,
            acquisition_mode=acquisition_mode,
            trace_index=trace_index,
            grid_x=grid_x,
            grid_y=grid_y,
            position_x_um=float(position_x[grid_x, grid_y]),
            position_y_um=float(position_y[grid_x, grid_y]),
            origin_x_um=origin_x_um,
            origin_y_um=origin_y_um,
        )
        yield DecodedTrace(
            metadata=metadata,
            x_axis=x_axis_values,
            intensity=np.asarray(graph_array[grid_x, grid_y, :], dtype=float),
        )


def build_witio_trace_metadata(
    *,
    descriptor: dict[str, object],
    source_wip: str | Path,
    acquisition_mode: str,
    trace_index: int,
    grid_x: int,
    grid_y: int,
    position_x_um: float,
    position_y_um: float,
    origin_x_um: float,
    origin_y_um: float,
) -> dict[str, object]:
    size_x = int(descriptor["size_x"])
    size_y = int(descriptor["size_y"])
    size_graph = int(descriptor["size_graph"])
    source_tree_root = str(descriptor["source_tree_root"])
    entry_metadata = descriptor.get("entry_metadata")
    if not isinstance(entry_metadata, dict):
        entry_metadata = {}

    measurement_config = {
        **entry_metadata,
        "entry_id": descriptor["entry_id"],
        "graph_caption": descriptor["caption"],
        "data_shape": [size_x, size_y, size_graph],
        "position_unit": "um",
        "position_x": position_x_um,
        "position_y": position_y_um,
        "scan_label": descriptor["caption"],
        "extraction_backend": "witio",
        "estimated_raw_array_mb": descriptor["estimated_raw_array_mb"],
    }
    metadata: dict[str, object] = {
        **entry_metadata,
        "source_wip": str(Path(source_wip)),
        "source_tree_path": build_witio_source_tree_path(source_tree_root, acquisition_mode, trace_index),
        "spectrum_type": descriptor["spectrum_type"],
        "acquisition_mode": acquisition_mode,
        "x_axis_unit": descriptor["x_axis_unit"],
        "measurement_time": descriptor["measurement_time"],
        "trace_index": trace_index,
        "trace_count": int(descriptor["trace_count"]),
        "scan_size_x": size_x,
        "scan_size_y": size_y,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "measurement_config": measurement_config,
    }
    if acquisition_mode in {LINE_SCAN, SERIES_SCAN}:
        metadata["secondary_axis_kind"] = "position" if acquisition_mode == LINE_SCAN else "trace_index"
        metadata["secondary_axis_unit"] = "um" if acquisition_mode == LINE_SCAN else "index"
        metadata["secondary_axis_value"] = (
            math.hypot(position_x_um - origin_x_um, position_y_um - origin_y_um)
            if acquisition_mode == LINE_SCAN
            else float(trace_index)
        )
    return metadata


def probe_witio_file(
    source_wip: str | Path,
    *,
    max_point_datasets: int = 2,
    max_line_datasets: int = 1,
    max_area_datasets: int = 1,
    max_series_datasets: int = 1,
    max_traces_per_point: int = 1,
    max_traces_per_line: int = 5,
    max_traces_per_area: int = 4,
    max_traces_per_series: int = 5,
) -> tuple[dict[str, object], list[DatasetSample]]:
    witio = load_witio()
    source_path = Path(source_wip)
    project = witio.read(source_path)
    trace_metadata_lookup = build_witio_trace_metadata_lookup(project)
    project_metadata = extract_witio_project_metadata(project)

    descriptors: list[dict[str, object]] = []
    photo_descriptors: list[dict[str, object]] = []
    class_counts: dict[str, int] = {}
    graph_entries: dict[int | None, Any] = {}
    for entry in project.data:
        class_name = str(getattr(entry, "class_name", "unknown"))
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        descriptor = describe_witio_graph(
            entry,
            trace_metadata_lookup=trace_metadata_lookup,
            project_metadata=project_metadata,
        )
        if descriptor is None:
            photo_descriptor = describe_witio_photo_entry(
                entry,
                trace_metadata_lookup=trace_metadata_lookup,
                project_metadata=project_metadata,
            )
            if photo_descriptor is not None:
                photo_descriptors.append(photo_descriptor)
            continue
        descriptors.append(descriptor)
        graph_entries[descriptor["entry_id"]] = entry

    descriptors.sort(key=lambda item: ((item["entry_id"] is None), item["entry_id"], item["caption"]))
    selected = select_witio_datasets(
        descriptors,
        max_point_datasets=max_point_datasets,
        max_line_datasets=max_line_datasets,
        max_area_datasets=max_area_datasets,
        max_series_datasets=max_series_datasets,
    )

    dataset_samples: list[DatasetSample] = []
    for descriptor in selected:
        entry = graph_entries.get(descriptor["entry_id"])
        if entry is None:
            continue
        mode = str(descriptor["acquisition_mode"])
        if mode == POINT_SPECTRUM:
            trace_limit = max_traces_per_point
        elif mode == LINE_SCAN:
            trace_limit = max_traces_per_line
        elif mode == AREA_MAP:
            trace_limit = max_traces_per_area
        else:
            trace_limit = max_traces_per_series
        dataset_samples.append(
            sample_witio_dataset(
                entry,
                source_wip=source_path,
                descriptor=descriptor,
                trace_limit=trace_limit,
            )
        )

    inventory_by_mode: dict[str, int] = {}
    traces_by_mode: dict[str, int] = {}
    for descriptor in descriptors:
        mode = str(descriptor["acquisition_mode"])
        inventory_by_mode[mode] = inventory_by_mode.get(mode, 0) + 1
        traces_by_mode[mode] = traces_by_mode.get(mode, 0) + int(descriptor["trace_count"])

    media_inventory: dict[str, int] = {}
    for descriptor in photo_descriptors:
        media_kind = str(descriptor["media_kind"])
        media_inventory[media_kind] = media_inventory.get(media_kind, 0) + 1

    report = {
        "file_path": str(source_path),
        "project_version": project.version,
        "data_count": len(project.data),
        "class_counts": class_counts,
        "project_metadata": project_metadata,
        "inventory_by_mode": inventory_by_mode,
        "traces_by_mode": traces_by_mode,
        "media_inventory": media_inventory,
        "selected_datasets": [sample.summary for sample in dataset_samples],
        "selection_limits": {
            "max_point_datasets": max_point_datasets,
            "max_line_datasets": max_line_datasets,
            "max_area_datasets": max_area_datasets,
            "max_series_datasets": max_series_datasets,
            "max_traces_per_point": max_traces_per_point,
            "max_traces_per_line": max_traces_per_line,
            "max_traces_per_area": max_traces_per_area,
            "max_traces_per_series": max_traces_per_series,
        },
        "notes": [
            "source_tree_path values are generated from acquisition mode, WITec entry id, and caption.",
            "Large maps are sampled by trace count in this probe to keep disk and memory usage bounded.",
            "estimated_raw_array_mb is based on the stored WITec DataType; actual_array_mb reflects the in-memory numpy array after witio decoding.",
        ],
    }
    return report, dataset_samples


def extract_measurement_time(entry: Any) -> str | None:
    tdata = getattr(entry, "tdata", None)
    if tdata is None:
        return None
    for name in ("CreationOrChangeUTCTime", "CreationOrChangeLocalTime"):
        tag = tdata.find(name)
        if tag is not None:
            value = tag.scalar()
            if value not in (None, ""):
                return _normalize_measurement_time_value(value)
    return None


def _normalize_measurement_time_value(value: object) -> str:
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:
            pass
    return str(value)


def _estimate_graph_array_nbytes(payload: Any, *, size_x: int, size_y: int, size_graph: int) -> int:
    graph_data = payload.path("GraphData", "DataType")
    data_type = int(graph_data.scalar()) if graph_data is not None else 0
    bytes_per_value = {
        1: 8,
        2: 4,
        3: 2,
        4: 1,
        5: 4,
        6: 2,
        7: 1,
        8: 1,
        9: 4,
        10: 8,
    }.get(data_type, 0)
    return size_x * size_y * size_graph * bytes_per_value


def _estimate_image_array_nbytes(payload: Any, *, size_x: int, size_y: int, class_name: str) -> int:
    if class_name == "TDImage":
        image_data = payload.path("ImageData", "DataType")
        data_type = int(image_data.scalar()) if image_data is not None else 0
        bytes_per_value = {
            1: 8,
            2: 4,
            3: 2,
            4: 1,
            5: 4,
            6: 2,
            7: 1,
            8: 1,
            9: 4,
            10: 8,
        }.get(data_type, 0)
        return size_x * size_y * bytes_per_value

    # TDBitmap uses packed RGB(A) data internally.
    return size_x * size_y * 4


def _required_scalar(payload: Any, name: str) -> int:
    tag = payload.find(name)
    if tag is None:
        raise ValueError(f"Required WITec payload tag missing: {name}")
    return int(tag.scalar())


def _optional_entry_id(entry: Any) -> int | None:
    try:
        value = getattr(entry, "id", None)
    except Exception:  # pragma: no cover - defensive
        return None
    if value in (None, ""):
        return None
    return int(value)
