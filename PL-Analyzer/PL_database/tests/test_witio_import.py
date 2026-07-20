from __future__ import annotations

from types import SimpleNamespace

from backend.services.witio_import import extract_witio_project_metadata, extract_witio_trace_metadata


class FakeTag:
    def __init__(self, name: str, data=None, children: list["FakeTag"] | None = None, *, tag_type: int | None = None):
        self.name = name
        self.data = data
        self.children = children or []
        self.type = tag_type if tag_type is not None else (0 if self.children else 9)

    def find(self, name: str):
        for child in self.children:
            if child.name == name:
                return child
        return None

    def scalar(self):
        if self.children:
            raise ValueError("tree tag")
        return self.data


def leaf(name: str, data) -> FakeTag:
    return FakeTag(name=name, data=data)


def branch(name: str, *children: FakeTag) -> FakeTag:
    return FakeTag(name=name, children=list(children))


def tree_name(name: str) -> FakeTag:
    return FakeTag(name=name, children=[], tag_type=0)


def build_param(index: int, guid: str, value, *, field_name: str, content_kind: int) -> FakeTag:
    children = [
        leaf("ParamGuid", guid),
        leaf("ContentKind", content_kind),
    ]
    if field_name == "Range":
        start, stop = value
        children.extend([leaf("Start", start), leaf("Stop", stop)])
    else:
        children.append(leaf(field_name, value))
    return branch(f"Element{index}", *children)


def build_params(name: str, params: list[FakeTag]) -> FakeTag:
    return branch(
        name,
        leaf("NumElements", len(params)),
        *params,
    )


def test_extract_witio_project_metadata_reads_tree_encoded_strings() -> None:
    project = SimpleNamespace(
        root=branch(
            "WITec Project",
            branch(
                "SystemInformation",
                branch("SystemID", tree_name("WS-2882")),
                branch("ApplicationVersions", tree_name("WITec Control 7.0.5.153 (Plus Version)")),
            ),
        )
    )

    metadata = extract_witio_project_metadata(project)

    assert metadata["system_id"] == "WS-2882"
    assert metadata["application_version"] == "WITec Control 7.0.5.153 (Plus Version)"


def test_extract_witio_trace_metadata_maps_known_measurement_parameters() -> None:
    trace_element = branch(
        "Element11",
        leaf("TraceGuid", "{trace-guid}"),
        leaf("TraceSourceGuid", "{trace-source-guid}"),
        leaf("TraceSourceVersion", 1),
        leaf("CreationUTCTime", "2026-01-26T17:08:38.832Z"),
        leaf("CreationLocalTime", "27.01.2026 01:08:38"),
        leaf("UserName", r"Z790-P\Witec"),
        branch(
            "ParamSets",
            leaf("NumElements", 3),
            branch(
                "Element0",
                build_params(
                    "Params",
                    [
                        build_param(0, "{50786913-FDEE-4C98-A3D2-69CEAD859087}", "WS-2882", field_name="StringValue", content_kind=1),
                        build_param(1, "{1AFAD5D3-7519-4017-9402-169E14483ACB}", "Raman CCD1", field_name="StringValue", content_kind=1),
                        build_param(2, "{3BD5B7C3-02B3-4226-8541-B4FA098F9FAA}", 3.2530004857107997, field_name="DoubleValue", content_kind=2),
                        build_param(3, "{C59F0865-BCA6-42CC-970B-E78863813092}", 488.0050048828125, field_name="DoubleValue", content_kind=2),
                        build_param(4, "{E1353512-9D5D-4B42-85B0-563E1D2E0922}", 0.5378113389015198, field_name="DoubleValue", content_kind=2),
                        build_param(5, "{DE90D5DF-6355-4003-8C30-69F4ECA394D6}", 0.5071023106575012, field_name="DoubleValue", content_kind=2),
                        build_param(6, "{C7BF2E9E-4588-4FE7-A661-8BC5F9ABBEA9}", 0.02, field_name="DoubleValue", content_kind=2),
                        build_param(7, "{09449FFD-F53F-48FE-A687-D17A866CDB64}", 1, field_name="IntValue", content_kind=0),
                        build_param(8, "{16EDE678-FAF2-4B8D-B8E4-D7F8805438A0}", "Zeiss EC Epiplan 10x / 0.25", field_name="StringValue", content_kind=1),
                        build_param(9, "{B4B5BA2D-BCF0-4DF3-AFE8-09808536C16B}", 10.0, field_name="DoubleValue", content_kind=2),
                        build_param(10, "{B9B7E353-ACAD-434A-B02E-65260A793A76}", False, field_name="BoolValue", content_kind=5),
                        build_param(11, "{0894921D-94F8-4B6A-92E0-B1F8F9C875E7}", -25104.725, field_name="DoubleValue", content_kind=2),
                        build_param(12, "{459B4B02-6301-4309-AB36-122A31ACF47B}", 14303.475, field_name="DoubleValue", content_kind=2),
                        build_param(13, "{FD47BC57-3E4E-48B3-BF05-CD87CA53540E}", 0.0, field_name="DoubleValue", content_kind=2),
                    ],
                ),
            ),
            branch(
                "Element1",
                build_params(
                    "Params",
                    [
                        build_param(0, "{8DF0A965-92FB-478B-BA56-9F4BCFAB9EFE}", "UHTS300S_VIS", field_name="StringValue", content_kind=1),
                        build_param(1, "{D4FB7097-4A2B-48FB-8619-27FA848640C1}", "UHTSWSN:0XV6", field_name="StringValue", content_kind=1),
                        build_param(2, "{70C15292-3849-485A-A19C-1C63BA2E737A}", "G1: 300 g/mm BLZ 500.00 nm", field_name="StringValue", content_kind=1),
                        build_param(3, "{4A1C85B9-F3E1-4D07-936D-D80C3E68F86F}", 669.9895629882812, field_name="DoubleValue", content_kind=2),
                    ],
                ),
            ),
            branch(
                "Element2",
                build_params(
                    "Params",
                    [
                        build_param(0, "{75B3D31D-985E-4622-84E7-D07229299AA3}", "DR316B_LD,DD", field_name="StringValue", content_kind=1),
                        build_param(1, "{41B6F20B-F543-4221-BCE0-FF70518D61A3}", "11339", field_name="StringValue", content_kind=1),
                        build_param(2, "{144EA40E-6F56-4321-9575-E08F11CD9DF8}", 0.02, field_name="DoubleValue", content_kind=2),
                        build_param(3, "{D3C1554D-27F0-414C-9E26-0A58EA8C3DB7}", 0.10207, field_name="DoubleValue", content_kind=2),
                        build_param(4, "{5751D616-EEB4-4778-9BFC-21F7E98A916C}", "{A9769D28-E279-400B-9129-9473586C1A9F}", field_name="EnumValueGuid", content_kind=6),
                        build_param(5, "{4BF5EC32-10B7-484F-8E8E-D1B6AF14182D}", (1, 20), field_name="Range", content_kind=4),
                        build_param(6, "{57946D40-7278-42C9-9F0F-96E3F2F04BA0}", 32.13, field_name="DoubleValue", content_kind=2),
                        build_param(7, "{979250FF-18FC-4E38-99AE-ED8B40F16A93}", 0.13, field_name="DoubleValue", content_kind=2),
                        build_param(8, "{577AED5A-C75E-4DF0-A472-A8A9B4C00152}", 4.0, field_name="DoubleValue", content_kind=2),
                        build_param(9, "{499DAAD1-793E-45DC-AE0E-5010C2839F66}", -59.0, field_name="DoubleValue", content_kind=2),
                    ],
                ),
            ),
        ),
    )

    metadata = extract_witio_trace_metadata(trace_element)

    assert metadata["system_id"] == "WS-2882"
    assert metadata["configuration_name"] == "Raman CCD1"
    assert metadata["duration_s"] == 3.2530004857107997
    assert metadata["laser_wavelength_nm"] == 488.0050048828125
    assert metadata["laser_power_in_fiber_mw"] == 0.5378113389015198
    assert metadata["laser_power_mw"] == 0.5071023106575012
    assert metadata["integration_time_s"] == 0.02
    assert metadata["accumulations"] == 1
    assert metadata["objective_name"] == "Zeiss EC Epiplan 10x / 0.25"
    assert metadata["sample_position_um"] == {"x": -25104.725, "y": 14303.475, "z": 0.0}
    assert metadata["spectrograph_name"] == "UHTS300S_VIS"
    assert metadata["grating"] == "G1: 300 g/mm BLZ 500.00 nm"
    assert metadata["camera_name"] == "DR316B_LD,DD"
    assert metadata["camera_readout_mode"] == "Single Track"
    assert metadata["camera_single_track_range"] == {"start": 1, "stop": 20}
    assert metadata["trace_param_set_count"] == 3
