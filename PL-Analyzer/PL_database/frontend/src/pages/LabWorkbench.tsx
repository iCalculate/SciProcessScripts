import { startTransition, useEffect, useState } from "react";
import {
  applyMetadata,
  fetchDashboard,
  fetchFilterOptions,
  fetchHealth,
  fetchImportedUploadHistory,
  fetchImportJobs,
  fetchSpectra,
  fetchSpectrum,
  probeUploadedFile,
  runBatchAnalysis,
  startUploadedImport,
  startImport,
  stopImportJob,
  type UploadedImportBatchResult
} from "../api";
import { AnalysisPanel } from "../components/AnalysisPanel";
import { AppHeader } from "../components/AppHeader";
import { DashboardPanel } from "../components/DashboardPanel";
import { DatabasePanel } from "../components/DatabasePanel";
import { ImportPanel } from "../components/ImportPanel";
import { MaterialAnalysisPanel } from "../components/MaterialAnalysisPanel";
import type {
  AnalysisOptions,
  BatchAnalysisResponse,
  DashboardSummary,
  FilterOptions,
  HealthPayload,
  ImportedUploadHistoryItem,
  ImportJob,
  ImportOptions,
  ImportProbeReport,
  SpectraResponse,
  SpectrumDetail,
  SpectrumFilters,
  UploadTransferProgress
} from "../types";

const EMPTY_FILTER_OPTIONS: FilterOptions = {
  spectrum_type: [],
  source: [],
  belonging: [],
  acquisition_mode: [],
  substrate: [],
  x_axis_unit: [],
  sample_id: [],
  analysis_material: [],
  analysis_family: [],
  analysis_status: []
};

const EMPTY_SPECTRA: SpectraResponse = {
  total: 0,
  items: []
};

type WorkbenchPage = "dashboard" | "import" | "database" | "analysis";

export function LabWorkbench() {
  const [health, setHealth] = useState<HealthPayload | null>(null);
  const [dashboard, setDashboard] = useState<DashboardSummary | null>(null);
  const [jobs, setJobs] = useState<ImportJob[]>([]);
  const [filterOptions, setFilterOptions] = useState<FilterOptions>(EMPTY_FILTER_OPTIONS);
  const [filters, setFilters] = useState<SpectrumFilters>({});
  const [spectra, setSpectra] = useState<SpectraResponse>(EMPTY_SPECTRA);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [activeSpectrumId, setActiveSpectrumId] = useState<string | null>(null);
  const [selectedSpectra, setSelectedSpectra] = useState<SpectrumDetail[]>([]);
  const [selectedSpectraLoading, setSelectedSpectraLoading] = useState(false);
  const [analysisResponse, setAnalysisResponse] = useState<BatchAnalysisResponse | null>(null);
  const [banner, setBanner] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [activePage, setActivePage] = useState<WorkbenchPage>("dashboard");

  async function refreshOverview() {
    const healthPayload = await fetchHealth();
    setHealth(healthPayload);
    const dashboardPayload = await fetchDashboard();
    setDashboard(dashboardPayload);
  }

  async function refreshJobs() {
    setJobs(await fetchImportJobs());
  }

  async function refreshFilterOptions() {
    setFilterOptions(await fetchFilterOptions());
  }

  async function refreshSpectra() {
    setSpectra(await fetchSpectra(filters));
  }

  useEffect(() => {
    void fetchHealth()
      .then(setHealth)
      .catch((caught) => setError(caught instanceof Error ? caught.message : String(caught)));
    void fetchDashboard()
      .then(setDashboard)
      .catch((caught) => setError(caught instanceof Error ? caught.message : String(caught)));
    void refreshJobs().catch((caught) => setError(caught instanceof Error ? caught.message : String(caught)));
  }, []);

  useEffect(() => {
    if (activePage !== "database" && activePage !== "analysis") {
      return;
    }
    void (async () => {
      try {
        await Promise.all([refreshFilterOptions(), refreshSpectra()]);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : String(caught));
      }
    })();
  }, [activePage]);

  useEffect(() => {
    if (activePage !== "database" && activePage !== "analysis") {
      return;
    }
    startTransition(() => {
      void refreshSpectra().catch((caught) => setError(caught instanceof Error ? caught.message : String(caught)));
    });
  }, [filters, activePage]);

  useEffect(() => {
    if (selectedIds.length === 0) {
      setSelectedSpectraLoading(false);
      setSelectedSpectra([]);
      return;
    }
    setSelectedSpectraLoading(true);
    void Promise.allSettled(selectedIds.slice(0, 5).map((spectrumId) => fetchSpectrum(spectrumId)))
      .then((results) => {
        const fulfilled = results
          .filter((result): result is PromiseFulfilledResult<SpectrumDetail> => result.status === "fulfilled")
          .map((result) => result.value);
        const rejected = results.filter((result): result is PromiseRejectedResult => result.status === "rejected");
        setSelectedSpectra(fulfilled);
        if (fulfilled.length > 0) {
          if (rejected.length > 0) {
            setError("Some selected rows could not be loaded, but the available spectra were kept.");
          }
          return;
        }
        if (rejected.length > 0) {
          const firstReason = rejected[0].reason;
          setError(firstReason instanceof Error ? firstReason.message : String(firstReason));
        }
      })
      .catch((caught) => setError(caught instanceof Error ? caught.message : String(caught)));
    return () => {
      setSelectedSpectraLoading(false);
    };
  }, [selectedIds]);

  useEffect(() => {
    if (selectedIds.length === 0) {
      return;
    }
    if (selectedSpectra.length > 0) {
      setSelectedSpectraLoading(false);
    }
  }, [selectedIds, selectedSpectra]);

  useEffect(() => {
    const timer = window.setInterval(() => {
      void Promise.all([refreshOverview(), refreshJobs()]).catch(() => undefined);
    }, 3000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (activePage !== "database") {
      return;
    }
    const timer = window.setInterval(() => {
      void refreshSpectra().catch(() => undefined);
    }, 8000);
    return () => window.clearInterval(timer);
  }, [activePage, filters]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      window.dispatchEvent(new Event("resize"));
    }, 60);
    return () => window.clearTimeout(timer);
  }, [activePage]);

  async function handleStartImport(
    inputPath: string,
    recursive: boolean,
    forceReimport: boolean,
    options: ImportOptions
  ) {
    setError("");
    setBanner("");
    try {
      await startImport(inputPath, recursive, forceReimport, options);
      await Promise.all([refreshJobs(), refreshOverview()]);
      setBanner("Import job queued.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  async function handleStartUploadImport(
    rootName: string,
    files: File[],
    sourceKind: "file_upload" | "folder_upload",
    forceReimport: boolean,
    options: ImportOptions,
    onProgress?: (progress: UploadTransferProgress) => void,
    onJobQueued?: (job: ImportJob) => void | Promise<void>,
    shouldStop?: () => boolean,
    registerAbort?: (abortCurrentUpload: (() => void) | null) => void
  ): Promise<UploadedImportBatchResult> {
    setError("");
    setBanner("");
    try {
      const result = await startUploadedImport(
        rootName,
        files,
        sourceKind,
        forceReimport,
        options,
        onProgress,
        async (job) => {
          onJobQueued?.(job);
          await Promise.all([refreshJobs(), refreshOverview()]);
        },
        shouldStop,
        registerAbort
      );
      await Promise.all([refreshJobs(), refreshOverview()]);
      if (sourceKind === "file_upload") {
        setBanner("Selected file queued and started importing.");
      } else if (result.failedUploads.length > 0) {
        setBanner(
          `Folder upload queued ${result.jobs.length} files. ${result.failedUploads.length} files failed before queueing and were skipped.`
        );
      } else if (result.jobs.some((job) => job.status === "failed")) {
        const failedJobCount = result.jobs.filter((job) => job.status === "failed").length;
        setBanner(
          `Folder upload recorded ${result.jobs.length} files. ${failedJobCount} failed during staging and were marked without stopping the batch.`
        );
      } else {
        setBanner(
          `Folder upload is streaming ${result.jobs.length} files. Each file starts importing as soon as its upload finishes.`
        );
      }
      return result;
    } catch (caught) {
      if (caught instanceof Error && caught.message === "UPLOAD_ABORTED") {
        setBanner("Upload stopped. Already completed imports were kept.");
        return {
          jobs: [],
          failedUploads: []
        };
      }
      setError(caught instanceof Error ? caught.message : String(caught));
      return {
        jobs: [],
        failedUploads: []
      };
    }
  }

  async function handleFetchUploadHistory(rootName?: string): Promise<ImportedUploadHistoryItem[]> {
    return (await fetchImportedUploadHistory(rootName)).items;
  }

  async function handleStopImportJobs(jobIds: string[]) {
    if (jobIds.length === 0) {
      return;
    }
    await Promise.all(
      jobIds.map(async (jobId) => {
        try {
          await stopImportJob(jobId);
        } catch {
          return;
        }
      })
    );
    await Promise.all([refreshJobs(), refreshOverview()]);
  }

  async function handleProbeFile(file: File): Promise<ImportProbeReport> {
    setError("");
    return await probeUploadedFile(file);
  }

  async function handleApplyMetadata(
    metadata: Record<string, string | null>,
    applyMode: "selected" | "source_file" | "folder" | "all",
    scopeValue: string | null
  ) {
    setError("");
    setBanner("");
    try {
      const response = await applyMetadata(selectedIds, applyMode, scopeValue, metadata);
      await Promise.all([refreshSpectra(), refreshFilterOptions()]);
      setBanner(`Updated ${response.updated_rows} rows.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  async function handleRunAnalysis(options: AnalysisOptions, saveResults: boolean) {
    setError("");
    setBanner("");
    try {
      const response = await runBatchAnalysis(selectedIds, options, saveResults);
      setAnalysisResponse(response);
      setBanner(`Analysis completed for ${response.summary.length} spectra.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    }
  }

  async function handleMaterialAnalysisCompleted() {
    await Promise.all([refreshOverview(), refreshSpectra(), refreshFilterOptions()]);
  }

  function toggleSelection(spectrumId: string) {
    setActiveSpectrumId(spectrumId);
    setSelectedIds((current) =>
      current.includes(spectrumId) ? current.filter((item) => item !== spectrumId) : [...current, spectrumId]
    );
  }

  function clearSelection() {
    setSelectedIds([]);
  }

  return (
    <main className="app-shell">
      <nav className="workspace-nav" aria-label="Workbench pages">
        <div className="workspace-tab-group">
          <button
            className={`page-tab ${activePage === "dashboard" ? "page-tab-active" : ""}`}
            onClick={() => setActivePage("dashboard")}
            type="button"
          >
            Dashboard
          </button>
          <button
            className={`page-tab ${activePage === "import" ? "page-tab-active" : ""}`}
            onClick={() => setActivePage("import")}
            type="button"
          >
            Data import
          </button>
          <button
            className={`page-tab ${activePage === "database" ? "page-tab-active" : ""}`}
            onClick={() => setActivePage("database")}
            type="button"
          >
            Database
          </button>
          <button
            className={`page-tab ${activePage === "analysis" ? "page-tab-active" : ""}`}
            onClick={() => setActivePage("analysis")}
            type="button"
          >
            Spectral analysis
          </button>
        </div>
        <AppHeader
          activePage={activePage}
          dashboard={dashboard}
          health={health}
          jobs={jobs}
          selectedCount={selectedIds.length}
          spectraTotal={spectra.total}
        />
      </nav>

      {banner ? <div className="banner banner-good">{banner}</div> : null}
      {error ? <div className="banner banner-error">{error}</div> : null}

      <div className="workspace-stage">
        <section className={`workspace-page ${activePage === "dashboard" ? "workspace-page-active" : ""}`}>
          <DashboardPanel dashboard={dashboard} health={health} />
        </section>

        <section className={`workspace-page ${activePage === "import" ? "workspace-page-active" : ""}`}>
          <ImportPanel
            jobs={jobs}
            onFetchUploadHistory={handleFetchUploadHistory}
            onProbeFile={handleProbeFile}
            onStart={handleStartImport}
            onStartUpload={handleStartUploadImport}
            onStopJobs={handleStopImportJobs}
          />
        </section>

        <section className={`workspace-page ${activePage === "database" ? "workspace-page-active" : ""}`}>
          <DatabasePanel
            activeSpectrumId={activeSpectrumId}
            filterOptions={filterOptions}
            filters={filters}
            selectedIds={selectedIds}
            selectedSpectraLoading={selectedSpectraLoading}
            selectedSpectra={selectedSpectra}
            spectra={spectra}
            onClearSelection={clearSelection}
            onFilterChange={setFilters}
            onToggleSelect={toggleSelection}
          />
        </section>

        <section className={`workspace-page ${activePage === "analysis" ? "workspace-page-active" : ""}`}>
          <MaterialAnalysisPanel
            filters={filters}
            matchedCount={spectra.total}
            selectedIds={selectedIds}
            selectedSpectra={selectedSpectra}
            visibleRows={spectra.items}
            onCompleted={handleMaterialAnalysisCompleted}
          />
        </section>
      </div>
    </main>
  );
}
