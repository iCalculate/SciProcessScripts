import type {
  GeneratedCandidate,
  GenerationCondition,
  InspectionResponse
} from "./types";

export function scientific(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "-";
  }
  return value.toExponential(digits);
}

export function fixed(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "-";
  }
  return value.toFixed(digits);
}

export function candidateFilename(candidate: GeneratedCandidate): string {
  return `devicecurvegen-candidate-${candidate.candidate_id}-seed-${candidate.seed}.csv`;
}

export function candidateExportHref(
  candidate: GeneratedCandidate,
  condition: GenerationCondition
): string {
  const encoded = encodeURIComponent(JSON.stringify(condition));
  return `/api/export?candidate_id=${candidate.candidate_id}&seed=${candidate.seed}&condition=${encoded}`;
}

export function downloadInspection(inspection: InspectionResponse): void {
  const rows = ["segment,direction,Vg,Id"];
  inspection.segments.forEach((segment, segmentIndex) => {
    segment.voltage.forEach((voltage, index) => {
      rows.push(
        [
          segmentIndex + 1,
          segment.direction,
          voltage,
          segment.current[index]
        ].join(",")
      );
    });
  });
  const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `${inspection.filename.replace(/\.[^.]+$/, "")}-cleaned.csv`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 0);
}
