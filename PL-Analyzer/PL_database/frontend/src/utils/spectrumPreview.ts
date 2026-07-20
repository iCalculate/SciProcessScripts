interface SpectrumPreviewSource {
  x_axis: number[];
  x_axis_unit: string | null;
  laser_wavelength: string | null;
  spectrum_type: string | null;
  grating: string | null;
}

export interface PreviewAxis {
  values: number[];
  axisTitle: string;
  displayUnit: string;
  mode: "raw" | "raman_shift";
  rawUnit: string;
  laserWavelengthNm: number | null;
  laserSource: "metadata" | "inferred" | null;
}

const COMMON_LASER_LINES_NM = [325, 355, 405, 442, 457, 473, 488, 514, 532, 561, 594, 633, 660, 785, 830, 1064];

export function buildPreviewAxis(source: SpectrumPreviewSource): PreviewAxis {
  const rawUnit = String(source.x_axis_unit ?? "").trim();
  const normalizedUnit = rawUnit.toLowerCase();
  const spectrumFamily = inferSpectrumFamily(source);

  if (isRamanUnit(normalizedUnit) && spectrumFamily !== "PL") {
    return {
      values: source.x_axis,
      axisTitle: "Raman shift (cm^-1)",
      displayUnit: "cm^-1",
      mode: "raw",
      rawUnit: rawUnit || "cm^-1",
      laserWavelengthNm: parseLaserWavelength(source.laser_wavelength),
      laserSource: null
    };
  }

  if (!isNanometerUnit(normalizedUnit)) {
    return {
      values: source.x_axis,
      axisTitle: rawUnit || "x_axis",
      displayUnit: rawUnit || "x_axis",
      mode: "raw",
      rawUnit: rawUnit || "x_axis",
      laserWavelengthNm: parseLaserWavelength(source.laser_wavelength),
      laserSource: null
    };
  }

  const metadataLaser = parseLaserWavelength(source.laser_wavelength);
  const inferredLaser = metadataLaser == null ? inferLaserWavelength(source.x_axis, spectrumFamily === "Raman") : null;
  const laserWavelengthNm = metadataLaser ?? inferredLaser;
  const laserSource = metadataLaser != null ? "metadata" : inferredLaser != null ? "inferred" : null;

  if (laserWavelengthNm == null || !shouldRenderAsRamanShift(source, spectrumFamily, laserWavelengthNm, laserSource)) {
    return {
      values: source.x_axis,
      axisTitle: rawUnit || "Wavelength (nm)",
      displayUnit: "nm",
      mode: "raw",
      rawUnit: rawUnit || "nm",
      laserWavelengthNm: null,
      laserSource: null
    };
  }

  return {
    values: source.x_axis.map((value) => toRamanShift(value, laserWavelengthNm)),
    axisTitle: "Raman shift (cm^-1)",
    displayUnit: "cm^-1",
    mode: "raman_shift",
    rawUnit: rawUnit || "nm",
    laserWavelengthNm,
    laserSource
  };
}

export function summarizePreviewAxes(previews: PreviewAxis[]): {
  axisTitle: string;
  note: string | null;
} {
  if (previews.length === 0) {
    return {
      axisTitle: "x_axis",
      note: null
    };
  }

  const displayUnits = new Set(previews.map((item) => item.displayUnit));
  const converted = previews.filter((item) => item.mode === "raman_shift");

  if (displayUnits.size === 1) {
    return {
      axisTitle: previews[0].axisTitle,
      note: buildPreviewNote(converted)
    };
  }

  return {
    axisTitle: "Preview axis (mixed nm / cm^-1)",
    note: "Mixed PL and Raman-like selections are shown with their most informative preview axes."
  };
}

export function formatPreviewAxisSummary(preview: PreviewAxis): string {
  if (preview.displayUnit !== "cm^-1") {
    return preview.rawUnit || "nm";
  }
  if (preview.mode !== "raman_shift" || preview.laserWavelengthNm == null) {
    return "Raman shift (cm^-1)";
  }
  const laserLabel =
    preview.laserSource === "metadata"
      ? `${preview.laserWavelengthNm} nm laser`
      : `inferred ${preview.laserWavelengthNm} nm laser`;
  return `Raman shift (cm^-1, ${laserLabel})`;
}

function buildPreviewNote(converted: PreviewAxis[]): string | null {
  if (converted.length === 0) {
    return null;
  }
  const inferred = converted.filter((item) => item.laserSource === "inferred");
  if (inferred.length === 0) {
    return null;
  }

  const laserLabels = Array.from(new Set(inferred.map((item) => `${item.laserWavelengthNm ?? "?"} nm`)));
  return `Raman-like spectra are previewed in cm^-1 using inferred excitation ${laserLabels.join(", ")}.`;
}

function inferSpectrumFamily(source: SpectrumPreviewSource): "Raman" | "PL" | null {
  const fromGrating = inferSpectrumFamilyFromGrating(source.grating);
  if (fromGrating != null) {
    return fromGrating;
  }
  if (isNamedRaman(source.spectrum_type)) {
    return "Raman";
  }
  if (isNamedPl(source.spectrum_type)) {
    return "PL";
  }
  return null;
}

function inferSpectrumFamilyFromGrating(grating: string | null): "Raman" | "PL" | null {
  const text = String(grating ?? "").trim();
  if (!text) {
    return null;
  }
  if (/\bG3\b/i.test(text)) {
    return "Raman";
  }
  if (/\bG(?:1|2)\b/i.test(text)) {
    return "PL";
  }
  return null;
}

function isNamedRaman(spectrumType: string | null): boolean {
  return /raman/i.test(String(spectrumType ?? ""));
}

function isNamedPl(spectrumType: string | null): boolean {
  return /\bpl\b|photoluminescence/i.test(String(spectrumType ?? ""));
}

function isRamanUnit(unit: string): boolean {
  return ["cm^-1", "cm-1", "1/cm", "raman"].some((token) => unit.includes(token));
}

function isNanometerUnit(unit: string): boolean {
  return ["nm", "nanometer", "nanometre"].some((token) => unit.includes(token));
}

function parseLaserWavelength(value: string | null): number | null {
  if (!value) {
    return null;
  }
  const match = String(value).match(/(\d+(?:\.\d+)?)/);
  if (!match) {
    return null;
  }
  const parsed = Number(match[1]);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : null;
}

function inferLaserWavelength(xAxis: number[], preferWideRange: boolean): number | null {
  if (xAxis.length === 0) {
    return null;
  }

  const minimum = Math.min(...xAxis);
  const maximum = Math.max(...xAxis);
  const span = maximum - minimum;
  const maxSpan = preferWideRange ? 220 : 120;

  if (span <= 0 || span > maxSpan) {
    return null;
  }

  const candidates = COMMON_LASER_LINES_NM
    .filter((line) => line >= minimum - 10 && line <= minimum + 20)
    .sort((left, right) => Math.abs(left - minimum) - Math.abs(right - minimum));

  for (const candidate of candidates) {
    const shifts = xAxis.map((value) => toRamanShift(value, candidate));
    const minimumShift = Math.min(...shifts);
    const maximumShift = Math.max(...shifts);
    const positiveRatio = shifts.filter((value) => value >= -120).length / shifts.length;

    if (minimumShift >= -250 && maximumShift >= 120 && maximumShift <= 4200 && positiveRatio >= 0.85) {
      return candidate;
    }
  }

  return null;
}

function shouldRenderAsRamanShift(
  source: SpectrumPreviewSource,
  spectrumFamily: "Raman" | "PL" | null,
  laserWavelengthNm: number,
  laserSource: "metadata" | "inferred" | null
): boolean {
  if (source.x_axis.length === 0) {
    return false;
  }

  const shifts = source.x_axis.map((value) => toRamanShift(value, laserWavelengthNm));
  const minimumShift = Math.min(...shifts);
  const maximumShift = Math.max(...shifts);
  const span = Math.max(...source.x_axis) - Math.min(...source.x_axis);
  const isLabeledRaman = spectrumFamily === "Raman";
  const isInferred = laserSource === "inferred";

  if (minimumShift < -300 || maximumShift < 120 || maximumShift > 4200) {
    return false;
  }

  if (isLabeledRaman) {
    return true;
  }

  if (isInferred) {
    return span <= 120;
  }

  return span <= 90;
}

function toRamanShift(wavelengthNm: number, laserWavelengthNm: number): number {
  return (1e7 / laserWavelengthNm) - (1e7 / wavelengthNm);
}
