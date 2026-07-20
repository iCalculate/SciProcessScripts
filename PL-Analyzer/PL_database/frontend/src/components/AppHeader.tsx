import type { DashboardSummary, HealthPayload, ImportJob } from "../types";

interface AppHeaderProps {
  activePage: "dashboard" | "import" | "database" | "analysis";
  dashboard: DashboardSummary | null;
  health: HealthPayload | null;
  jobs: ImportJob[];
  selectedCount: number;
  spectraTotal: number;
}

export function AppHeader(props: AppHeaderProps) {
  const backendLabel = props.health?.import_backend === "witio" ? "Python witio importer" : "Importer";
  const apiLabel = props.health?.status === "ok" ? backendLabel : "API loading";
  const activeJobs = props.jobs.filter((job) => job.status === "pending" || job.status === "running").length;
  const latestJob = props.jobs[0] ?? props.dashboard?.latest_job ?? null;
  const failedImports = props.dashboard?.failed_imports ?? props.jobs.filter((job) => job.status === "failed").length;
  const segments = buildStatusSegments({
    activeJobs,
    apiLabel,
    backendReady: props.health?.status === "ok",
    dashboard: props.dashboard,
    failedImports,
    latestJob,
    page: props.activePage,
    selectedCount: props.selectedCount,
    spectraTotal: props.spectraTotal
  });

  return (
    <div className="app-header">
      <div className={`status-bubble ${props.health?.mock_mode ? "status-bubble-warn" : ""}`}>
        {segments.map((segment) => (
          <span className="status-segment" key={segment.label}>
            <span>{segment.label}</span>
            <strong>{segment.value}</strong>
          </span>
        ))}
      </div>
    </div>
  );
}

function buildStatusSegments(context: {
  activeJobs: number;
  apiLabel: string;
  backendReady: boolean;
  dashboard: DashboardSummary | null;
  failedImports: number;
  latestJob: ImportJob | null;
  page: AppHeaderProps["activePage"];
  selectedCount: number;
  spectraTotal: number;
}) {
  const apiSegment = {
    label: "API",
    value: context.backendReady ? context.apiLabel : "loading"
  };
  const indexedSpectra = context.dashboard ? formatCount(context.dashboard.spectra_count) : "-";
  const indexedFiles = context.dashboard ? formatCount(context.dashboard.imported_files) : "-";

  switch (context.page) {
    case "import":
      return [
        apiSegment,
        { label: "Active", value: String(context.activeJobs) },
        { label: "Latest", value: formatJobStatus(context.latestJob?.status) },
        { label: "Failed", value: String(context.failedImports) }
      ];
    case "database":
      return [
        apiSegment,
        { label: "Indexed", value: indexedSpectra },
        { label: "Visible", value: formatCount(context.spectraTotal) },
        { label: "Selected", value: String(context.selectedCount) }
      ];
    case "analysis":
      return [
        apiSegment,
        { label: "Selected", value: String(context.selectedCount) },
        { label: "Visible", value: formatCount(context.spectraTotal) },
        { label: "Sources", value: indexedFiles }
      ];
    default:
      return [
        apiSegment,
        { label: "Spectra", value: indexedSpectra },
        { label: "WIP files", value: indexedFiles },
        { label: "Failed", value: String(context.failedImports) }
      ];
  }
}

function formatCount(value: number): string {
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 }).format(value);
}

function formatJobStatus(status: string | null | undefined): string {
  if (!status) {
    return "-";
  }
  return status.replace(/_/g, " ");
}
