import { useEffect, useState } from "react";
import type { SpectrumDetail } from "../types";

interface MetadataPanelProps {
  selectedSpectra: SpectrumDetail[];
  onApply: (
    metadata: Record<string, string | null>,
    applyMode: "selected" | "source_file" | "folder" | "all",
    scopeValue: string | null
  ) => Promise<void>;
}

const EMPTY_FORM: Record<string, string> = {
  sample_id: "",
  source: "",
  substrate: "",
  device_id: "",
  laser_wavelength: "",
  laser_power: "",
  integration_time: "",
  grating: "",
  objective: "",
  measurement_time: "",
  notes: ""
};

export function MetadataPanel(props: MetadataPanelProps) {
  const [form, setForm] = useState<Record<string, string>>(EMPTY_FORM);
  const [applyMode, setApplyMode] = useState<"selected" | "source_file" | "folder" | "all">("selected");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const first = props.selectedSpectra[0];
    if (!first) {
      setForm(EMPTY_FORM);
      return;
    }
    setForm({
      sample_id: first.sample_id ?? "",
      source: first.source ?? "",
      substrate: first.substrate ?? "",
      device_id: first.device_id ?? "",
      laser_wavelength: first.laser_wavelength ?? "",
      laser_power: first.laser_power ?? "",
      integration_time: first.integration_time ?? "",
      grating: first.grating ?? "",
      objective: first.objective ?? "",
      measurement_time: first.measurement_time ?? "",
      notes: first.notes ?? ""
    });
  }, [props.selectedSpectra]);

  const first = props.selectedSpectra[0] ?? null;
  const scopeValue =
    applyMode === "source_file"
      ? first?.file_path ?? null
      : applyMode === "folder"
        ? first?.folder_path ?? null
        : null;

  async function handleApply() {
    setSaving(true);
    try {
      await props.onApply(
        Object.fromEntries(Object.entries(form).map(([key, value]) => [key, value || null])),
        applyMode,
        scopeValue
      );
    } finally {
      setSaving(false);
    }
  }

  return (
    <section className="panel-grid">
      <div className="card card-span-2">
        <div className="card-head">
          <div>
            <p className="eyebrow">Metadata editor</p>
            <h2>Apply structured notes to selected spectra</h2>
          </div>
        </div>
        {props.selectedSpectra.length > 0 ? (
          <>
            <div className="form-grid">
              {Object.keys(form).map((field) => (
                <label key={field} className={field === "notes" ? "field field-wide" : "field"}>
                  <span>{field}</span>
                  {field === "notes" ? (
                    <textarea
                      rows={4}
                      value={form[field]}
                      onChange={(event) => setForm({ ...form, [field]: event.target.value })}
                    />
                  ) : (
                    <input value={form[field]} onChange={(event) => setForm({ ...form, [field]: event.target.value })} />
                  )}
                </label>
              ))}
            </div>
            <div className="action-row">
              <label className="field inline-field">
                <span>Apply scope</span>
                <select value={applyMode} onChange={(event) => setApplyMode(event.target.value as typeof applyMode)}>
                  <option value="selected">Selected spectra</option>
                  <option value="source_file">Same source .wip file</option>
                  <option value="folder">Same folder</option>
                  <option value="all">All indexed spectra</option>
                </select>
              </label>
              <button className="primary-button" disabled={saving} onClick={handleApply} type="button">
                {saving ? "Saving..." : "Apply metadata"}
              </button>
            </div>
          </>
        ) : (
          <p className="empty-state">Select spectra in the database view before editing metadata.</p>
        )}
      </div>
      <div className="card card-span-1">
        <div className="card-head">
          <div>
            <p className="eyebrow">Selection</p>
            <h2>Scope preview</h2>
          </div>
        </div>
        {first ? (
          <div className="summary-block">
            <div className="summary-row">
              <span>Selected spectra</span>
              <strong>{props.selectedSpectra.length}</strong>
            </div>
            <div className="summary-row">
              <span>Lead sample</span>
              <strong>{first.sample_id ?? "-"}</strong>
            </div>
            <div className="summary-row">
              <span>Source label</span>
              <code>{first.source ?? "-"}</code>
            </div>
            <div className="summary-row">
              <span>Source file</span>
              <code>{first.file_path}</code>
            </div>
            <div className="summary-row">
              <span>Folder</span>
              <code>{first.folder_path ?? "-"}</code>
            </div>
          </div>
        ) : (
          <p className="empty-state">No spectra selected.</p>
        )}
      </div>
    </section>
  );
}
