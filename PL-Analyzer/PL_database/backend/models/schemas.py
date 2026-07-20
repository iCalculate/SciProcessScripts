from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ImportOptions(BaseModel):
    include_point_spectra: bool = True
    include_line_scans: bool = False
    include_area_maps: bool = False
    include_series_scans: bool = False
    include_photo_images: bool = False


class ImportRequest(BaseModel):
    input_path: str = Field(min_length=1)
    recursive: bool = True
    force_reimport: bool = False
    options: ImportOptions = Field(default_factory=ImportOptions)


class MetadataUpdateRequest(BaseModel):
    spectrum_ids: list[str] = Field(default_factory=list)
    apply_mode: Literal["selected", "source_file", "folder", "all"] = "selected"
    scope_value: str | None = None
    metadata: dict[str, str | None] = Field(default_factory=dict)


class AnalysisOptions(BaseModel):
    baseline_order: int = Field(default=3, ge=1, le=7)
    baseline_quantile: float = Field(default=0.25, gt=0.0, lt=1.0)
    smoothing_window: int = Field(default=11, ge=3, le=301)
    smoothing_polyorder: int = Field(default=3, ge=1, le=7)
    normalization: Literal["none", "max", "area", "zscore"] = "max"
    prominence: float = Field(default=0.05, ge=0.0)
    height: float | None = Field(default=None, ge=0.0)
    distance: int | None = Field(default=None, ge=1)
    fit_model: Literal["auto", "gaussian", "lorentzian", "pseudo_voigt"] = "gaussian"
    max_peaks: int = Field(default=4, ge=1, le=12)
    spectrum_family: Literal["auto", "PL", "Raman"] = "auto"
    material_hint: str | None = None
    method_version: str = "material-aware-v2"
    min_material_confidence: float = Field(default=0.28, ge=0.0, le=1.0)


class AnalysisRequest(BaseModel):
    spectrum_ids: list[str] = Field(default_factory=list, min_length=1)
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)
    save_results: bool = True


class BatchAnalysisRequest(BaseModel):
    spectrum_ids: list[str] = Field(default_factory=list, min_length=1)
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)
    save_results: bool = True


class MaterialAnalysisStartRequest(BaseModel):
    spectrum_ids: list[str] = Field(default_factory=list)
    filters: dict[str, str | None] = Field(default_factory=dict)
    search: str | None = None
    include_mock: bool = False
    options: AnalysisOptions = Field(default_factory=AnalysisOptions)
    save_results: bool = True
    update_entries: bool = True
