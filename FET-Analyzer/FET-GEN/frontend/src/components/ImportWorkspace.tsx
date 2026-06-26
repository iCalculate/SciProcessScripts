import { Database, FolderOpen, FolderSync, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { getDatabaseStatus, importDatabaseFolder, importDatabaseSource } from "../api";
import type {
  DatabaseImportRequest,
  DatabaseImportSummary,
  DatabaseStatus
} from "../types";

const ALL_SUFFIXES = [".csv", ".txt", ".tsv", ".dat", ".ztr", ".xtr", ".xml"];

const INITIAL_REQUEST: DatabaseImportRequest = {
  source_path: "",
  suffixes: [...ALL_SUFFIXES],
  max_xml_mb: 128,
  hash_files: false,
  replace: false
};

function StatCard({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="import-stat-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function ImportWorkspace() {
  const [request, setRequest] = useState<DatabaseImportRequest>(INITIAL_REQUEST);
  const [selectedFolderFiles, setSelectedFolderFiles] = useState<File[]>([]);
  const [summary, setSummary] = useState<DatabaseImportSummary | null>(null);
  const [status, setStatus] = useState<DatabaseStatus | null>(null);
  const [loadingStatus, setLoadingStatus] = useState(true);
  const [runningImport, setRunningImport] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  async function refreshStatus() {
    setLoadingStatus(true);
    try {
      const nextStatus = await getDatabaseStatus();
      setStatus(nextStatus);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not read database status");
    } finally {
      setLoadingStatus(false);
    }
  }

  useEffect(() => {
    void refreshStatus();
  }, []);

  const suffixLabel = useMemo(() => {
    if (request.suffixes.length === ALL_SUFFIXES.length) return "All supported suffixes";
    if (request.suffixes.length === 0) return "No suffix selected";
    return request.suffixes.join(", ");
  }, [request.suffixes]);

  function toggleSuffix(suffix: string) {
    setRequest((current) => {
      const selected = current.suffixes.includes(suffix)
        ? current.suffixes.filter((item) => item !== suffix)
        : [...current.suffixes, suffix];
      return { ...current, suffixes: selected };
    });
  }

  async function runImport() {
    if (!request.source_path.trim()) {
      setError("Source path is required.");
      return;
    }
    if (request.suffixes.length === 0) {
      setError("Select at least one source suffix.");
      return;
    }
    setRunningImport(true);
    setError(null);
    setSummary(null);
    try {
      const result = await importDatabaseSource({
        ...request,
        source_path: request.source_path.trim()
      });
      setSummary(result);
      const nextStatus = await getDatabaseStatus();
      setStatus(nextStatus);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Database import failed");
    } finally {
      setRunningImport(false);
    }
  }

  async function runFolderImport() {
    if (selectedFolderFiles.length === 0) {
      setError("Choose a folder first.");
      return;
    }
    if (request.suffixes.length === 0) {
      setError("Select at least one source suffix.");
      return;
    }
    setRunningImport(true);
    setError(null);
    setSummary(null);
    try {
      const result = await importDatabaseFolder(selectedFolderFiles, {
        suffixes: request.suffixes,
        max_xml_mb: request.max_xml_mb,
        hash_files: request.hash_files,
        replace: request.replace
      });
      setSummary(result);
      const nextStatus = await getDatabaseStatus();
      setStatus(nextStatus);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Folder import failed");
    } finally {
      setRunningImport(false);
    }
  }

  return (
    <main className="secondary-workspace import-workspace">
      <section className="workspace-heading">
        <div>
          <h1>Incremental database import</h1>
          <p>
            Import a raw measurement source directory into the DeviceCurveGen database.
            Unchanged files are skipped automatically, while new or modified files are
            parsed, aligned, and written into the database.
          </p>
        </div>
        <div className="import-header-actions">
          <label className="button secondary upload-button">
            <FolderOpen size={16} />
            Choose folder
            <input
              ref={folderInputRef}
              type="file"
              multiple
              webkitdirectory=""
              directory=""
              onChange={(event) => {
                const nextFiles = Array.from(event.currentTarget.files ?? []);
                setSelectedFolderFiles(nextFiles);
                if (!request.source_path && nextFiles.length > 0) {
                  const first = (nextFiles[0] as File & { webkitRelativePath?: string }).webkitRelativePath;
                  const rootName = first?.split("/")[0] ?? nextFiles[0].name;
                  setRequest((current) => ({ ...current, source_path: rootName }));
                }
              }}
            />
          </label>
          <button
            className="button secondary"
            disabled={loadingStatus || runningImport}
            onClick={() => void refreshStatus()}
          >
            <RefreshCw size={16} className={loadingStatus ? "spin" : ""} />
            Refresh status
          </button>
          <button
            className="button primary"
            disabled={runningImport}
            onClick={() => void runImport()}
          >
            <FolderSync size={16} />
            {runningImport ? "Importing..." : "Import into database"}
          </button>
          <button
            className="button primary"
            disabled={runningImport || selectedFolderFiles.length === 0}
            onClick={() => void runFolderImport()}
          >
            <FolderOpen size={16} />
            {runningImport ? "Importing..." : `Import selected folder (${selectedFolderFiles.length})`}
          </button>
        </div>
      </section>

      {error ? <div className="error-banner" role="alert">{error}</div> : null}

      <section className="import-status-strip">
        <StatCard label="Configured" value={status?.configured ? "Yes" : "No"} />
        <StatCard label="Source files" value={status?.source_files ?? "—"} />
        <StatCard label="Curves" value={status?.curves ?? "—"} />
        <StatCard label="Curves with Ig" value={status?.curves_with_ig ?? "—"} />
        <StatCard label="Rejected entries" value={status?.rejected_entries ?? "—"} />
      </section>

      <div className="import-grid">
        <section className="import-panel">
          <h2>Source directory</h2>
          <label className="import-field">
            <span>Root folder to scan</span>
            <input
              type="text"
              placeholder="D:\B1500_exports or use Choose folder"
              value={request.source_path}
              onChange={(event) =>
                setRequest((current) => ({ ...current, source_path: event.currentTarget.value }))
              }
            />
          </label>

          {selectedFolderFiles.length > 0 ? (
            <div className="import-summary-block">
              <strong>Selected folder payload</strong>
              <code>{selectedFolderFiles.length} file(s) selected from a local folder picker</code>
            </div>
          ) : null}

          <label className="import-field">
            <span>Max XML/XTR file size (MB)</span>
            <input
              type="number"
              min={1}
              max={4096}
              value={request.max_xml_mb}
              onChange={(event) =>
                setRequest((current) => ({
                  ...current,
                  max_xml_mb: Number(event.currentTarget.value) || 128
                }))
              }
            />
          </label>

          <div className="suffix-panel">
            <div className="suffix-panel-heading">
              <strong>Included suffixes</strong>
              <span>{suffixLabel}</span>
            </div>
            <div className="suffix-grid">
              {ALL_SUFFIXES.map((suffix) => (
                <label key={suffix} className="suffix-chip">
                  <input
                    type="checkbox"
                    checked={request.suffixes.includes(suffix)}
                    onChange={() => toggleSuffix(suffix)}
                  />
                  <span>{suffix}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="import-option-list">
            <label className="import-check">
              <input
                type="checkbox"
                checked={request.hash_files}
                onChange={(event) =>
                  setRequest((current) => ({ ...current, hash_files: event.currentTarget.checked }))
                }
              />
              <span>Use SHA1 to detect changed files more strictly</span>
            </label>
            <label className="import-check warning">
              <input
                type="checkbox"
                checked={request.replace}
                onChange={(event) =>
                  setRequest((current) => ({ ...current, replace: event.currentTarget.checked }))
                }
              />
              <span>Replace the whole database instead of importing incrementally</span>
            </label>
          </div>
        </section>

        <section className="import-panel">
          <h2>What this panel does</h2>
          <div className="import-note-list">
            <div>
              <Database size={18} />
              <p>Skips files whose relative path, size, and modified time still match the database.</p>
            </div>
            <div>
              <Database size={18} />
              <p>Reimports files that changed and adds files that are newly discovered under the same root.</p>
            </div>
            <div>
              <Database size={18} />
              <p>Writes raw points, aligned points, curve summaries, and gate-current data into the MySQL database.</p>
            </div>
            <div>
              <Database size={18} />
              <p>Leaves the existing database intact unless you explicitly enable full replace mode.</p>
            </div>
          </div>
        </section>
      </div>

      <section className="import-results">
        <h2>Last import summary</h2>
        {!summary ? (
          <div className="empty-state compact">
            <FolderSync size={30} />
            <h2>No import has been run yet</h2>
            <p>Run an incremental import to see how many files were discovered, skipped, added, and updated.</p>
          </div>
        ) : (
          <>
            <div className="import-status-strip summary">
              <StatCard label="Discovered" value={summary.files_discovered} />
              <StatCard label="Imported" value={summary.files_imported} />
              <StatCard label="Updated" value={summary.files_updated} />
              <StatCard label="Skipped" value={summary.files_skipped} />
              <StatCard label="Accepted segments" value={summary.accepted_transfer_segments} />
              <StatCard label="Rejected" value={summary.rejected_entries} />
            </div>
            <div className="import-summary-block">
              <strong>Imported source root</strong>
              <code>{summary.source}</code>
            </div>
          </>
        )}
      </section>
    </main>
  );
}
