import {
  Activity,
  CalendarDays,
  ChevronLeft,
  ChevronRight,
  Database,
  List,
  RefreshCw,
  Search,
  SlidersHorizontal
} from "lucide-react";
import { lazy, Suspense, useDeferredValue, useEffect, useMemo, useState } from "react";
import {
  getDatabaseCurve,
  getDatabaseOptions,
  getDatabaseStatus,
  listDatabaseCalendar,
  listDatabaseCurves
} from "../api";
import { fixed, scientific } from "../format";
import type {
  CalendarCurve,
  CalendarCurveResponse,
  CurveDetail,
  CurveFilters,
  CurveListResponse,
  DatabaseOptions,
  DatabaseSelectionState,
  DatabaseStatus
} from "../types";
import { Logo } from "./Logo";

const AnalysisWorkspace = lazy(() =>
  import("./AnalysisWorkspace").then((module) => ({
    default: module.AnalysisWorkspace
  }))
);

type BrowseView = "list" | "calendar";
type CalendarMode = "month" | "week" | "day";

const EMPTY_OPTIONS: DatabaseOptions = {
  source_kinds: [],
  polarities: [],
  directions: []
};

function dateKey(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function startOfWeek(date: Date): Date {
  const next = new Date(date);
  const day = (next.getDay() + 6) % 7;
  next.setDate(next.getDate() - day);
  next.setHours(0, 0, 0, 0);
  return next;
}

function visibleDates(anchor: Date, mode: CalendarMode): Date[] {
  if (mode === "day") return [new Date(anchor)];
  const start = mode === "week"
    ? startOfWeek(anchor)
    : startOfWeek(new Date(anchor.getFullYear(), anchor.getMonth(), 1));
  const count = mode === "week" ? 7 : 42;
  return Array.from({ length: count }, (_, index) => {
    const date = new Date(start);
    date.setDate(start.getDate() + index);
    return date;
  });
}

function shiftAnchor(anchor: Date, mode: CalendarMode, delta: number): Date {
  const next = new Date(anchor);
  if (mode === "month") next.setMonth(next.getMonth() + delta);
  else next.setDate(next.getDate() + delta * (mode === "week" ? 7 : 1));
  return next;
}

function patchFilter(filters: CurveFilters, key: keyof CurveFilters, value: string) {
  const next = { ...filters };
  if (value.trim()) next[key] = value;
  else delete next[key];
  return next;
}

function curveTone(curve: Pick<CalendarCurve, "polarity" | "direction">): string {
  if (curve.direction === "single") return "single";
  return curve.polarity === "p-type" ? "ptype" : "ntype";
}

function CurvePreview({ detail }: { detail: CurveDetail }) {
  const path = useMemo(() => {
    const points = detail.raw_points
      .map((point) => ({
        x: point.voltage_v,
        y: Math.log10(Math.max(Math.abs(point.current_a), Number.MIN_VALUE))
      }))
      .filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
    if (points.length < 2) return "";
    const minX = Math.min(...points.map((point) => point.x));
    const maxX = Math.max(...points.map((point) => point.x));
    const minY = Math.min(...points.map((point) => point.y));
    const maxY = Math.max(...points.map((point) => point.y));
    return points.map((point, index) => {
      const x = 30 + ((point.x - minX) / Math.max(maxX - minX, 1e-12)) * 300;
      const y = 18 + (1 - (point.y - minY) / Math.max(maxY - minY, 1e-12)) * 166;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(" ");
  }, [detail]);
  return (
    <svg className="lab-curve-preview" viewBox="0 0 360 220" role="img" aria-label="Transfer curve preview">
      <line x1="30" y1="184" x2="334" y2="184" />
      <line x1="30" y1="18" x2="30" y2="184" />
      <path d={path} />
      <text x="184" y="210">Gate voltage Vg (V)</text>
      <text x="12" y="104" transform="rotate(-90 12 104)">log10 |Id|</text>
    </svg>
  );
}

function FilterRail({
  filters,
  options,
  total,
  onChange,
  onReset
}: {
  filters: CurveFilters;
  options: DatabaseOptions;
  total: number;
  onChange: (filters: CurveFilters) => void;
  onReset: () => void;
}) {
  return (
    <aside className="lab-filter-rail">
      <div className="lab-rail-title">
        <span><SlidersHorizontal size={15} /> Filters</span>
        <button onClick={onReset}>Reset</button>
      </div>
      <label>
        Search
        <span className="lab-search">
          <Search size={14} />
          <input
            value={filters.source_search ?? ""}
            placeholder="Source, setup, primitive"
            onChange={(event) => onChange(patchFilter(filters, "source_search", event.target.value))}
          />
        </span>
      </label>
      <div className="lab-filter-pair">
        <label>
          From
          <input type="date" value={filters.date_from ?? ""} onChange={(event) => onChange(patchFilter(filters, "date_from", event.target.value))} />
        </label>
        <label>
          To
          <input type="date" value={filters.date_to ?? ""} onChange={(event) => onChange(patchFilter(filters, "date_to", event.target.value))} />
        </label>
      </div>
      <label>
        Polarity
        <select value={filters.polarity ?? ""} onChange={(event) => onChange(patchFilter(filters, "polarity", event.target.value))}>
          <option value="">All polarities</option>
          {options.polarities.map((value) => <option key={value}>{value}</option>)}
        </select>
      </label>
      <label>
        Direction
        <select value={filters.direction ?? ""} onChange={(event) => onChange(patchFilter(filters, "direction", event.target.value))}>
          <option value="">All directions</option>
          {options.directions.map((value) => <option key={value}>{value}</option>)}
        </select>
      </label>
      <label>
        Source kind
        <select value={filters.source_kind ?? ""} onChange={(event) => onChange(patchFilter(filters, "source_kind", event.target.value))}>
          <option value="">All sources</option>
          {options.source_kinds.map((value) => <option key={value}>{value}</option>)}
        </select>
      </label>
      <label>
        Hysteresis
        <select value={filters.hysteresis_available ?? ""} onChange={(event) => onChange(patchFilter(filters, "hysteresis_available", event.target.value))}>
          <option value="">All scans</option>
          <option value="true">Paired forward/reverse</option>
          <option value="false">One-way (NA)</option>
        </select>
      </label>
      <label>
        Gate current
        <select value={filters.has_gate_current ?? ""} onChange={(event) => onChange(patchFilter(filters, "has_gate_current", event.target.value))}>
          <option value="">All</option>
          <option value="true">With Ig</option>
          <option value="false">Without Ig</option>
        </select>
      </label>
      <div className="lab-result-total">
        <span>Results after filters</span>
        <strong>{total.toLocaleString()}</strong>
        <small>measurement curves</small>
      </div>
    </aside>
  );
}

function CalendarView({
  response,
  anchor,
  mode,
  selectedId,
  onSelect
}: {
  response: CalendarCurveResponse | null;
  anchor: Date;
  mode: CalendarMode;
  selectedId: string | null;
  onSelect: (curve: CalendarCurve) => void;
}) {
  const dates = useMemo(() => visibleDates(anchor, mode), [anchor, mode]);
  const grouped = useMemo(() => {
    const result = new Map<string, CalendarCurve[]>();
    (response?.items ?? []).forEach((curve) => {
      const key = curve.test_time.slice(0, 10);
      result.set(key, [...(result.get(key) ?? []), curve]);
    });
    return result;
  }, [response]);
  const maxRugs = mode === "month" ? 9 : mode === "week" ? 30 : 80;
  return (
    <div className={`lab-calendar-grid ${mode}`}>
      {dates.map((date) => {
        const key = dateKey(date);
        const curves = grouped.get(key) ?? [];
        const outside = mode === "month" && date.getMonth() !== anchor.getMonth();
        return (
          <section className={`lab-calendar-cell${outside ? " outside" : ""}`} key={key}>
            <header>
              <strong>{mode === "day" ? date.toLocaleDateString(undefined, { weekday: "long", month: "long", day: "numeric" }) : date.getDate()}</strong>
              <span>{curves.length || ""}</span>
            </header>
            <div className="lab-rugs">
              {curves.slice(0, maxRugs).map((curve) => (
                <button
                  key={curve.curve_id}
                  className={`lab-rug ${curveTone(curve)}${selectedId === curve.curve_id ? " selected" : ""}`}
                  title={`${curve.curve_id} · ${curve.direction} · hysteresis ${curve.hysteresis_v === null ? "NA" : `${fixed(curve.hysteresis_v, 3)} V`}`}
                  onClick={() => onSelect(curve)}
                >
                  <span />
                  {mode !== "month" ? <b>{curve.curve_id}</b> : null}
                </button>
              ))}
              {curves.length > maxRugs ? <small>+{curves.length - maxRugs} more</small> : null}
            </div>
          </section>
        );
      })}
    </div>
  );
}

export function DatabaseLabApp() {
  const [section, setSection] = useState<"browse" | "analysis">("browse");
  const [browseView, setBrowseView] = useState<BrowseView>("calendar");
  const [calendarMode, setCalendarMode] = useState<CalendarMode>("month");
  const [anchor, setAnchor] = useState(() => new Date());
  const [filters, setFilters] = useState<CurveFilters>({});
  const deferredFilters = useDeferredValue(filters);
  const [options, setOptions] = useState<DatabaseOptions>(EMPTY_OPTIONS);
  const [status, setStatus] = useState<DatabaseStatus | null>(null);
  const [list, setList] = useState<CurveListResponse | null>(null);
  const [calendar, setCalendar] = useState<CalendarCurveResponse | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<CurveDetail | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const calendarRange = useMemo(() => visibleDates(anchor, calendarMode), [anchor, calendarMode]);
  const calendarFilters = useMemo<CurveFilters>(() => ({
    ...deferredFilters,
    date_from: deferredFilters.date_from ?? dateKey(calendarRange[0]),
    date_to: deferredFilters.date_to ?? dateKey(calendarRange.at(-1) ?? calendarRange[0])
  }), [calendarRange, deferredFilters]);
  const total = list?.total ?? status?.curves ?? 0;
  const selection = useMemo<DatabaseSelectionState>(() => ({
    selectedIds: [],
    allFiltered: true,
    filters: deferredFilters,
    total
  }), [deferredFilters, total]);

  useEffect(() => {
    void Promise.all([getDatabaseOptions(), getDatabaseStatus()])
      .then(([nextOptions, nextStatus]) => {
        setOptions(nextOptions);
        setStatus(nextStatus);
      })
      .catch((caught) => setError(caught instanceof Error ? caught.message : "Database unavailable"));
  }, []);

  useEffect(() => {
    if (section !== "browse") return undefined;
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    const request = browseView === "calendar"
      ? listDatabaseCalendar(calendarFilters, controller.signal).then(setCalendar)
      : listDatabaseCurves(deferredFilters, 250, 0, "modified_at_desc", controller.signal).then(setList);
    void request
      .catch((caught) => {
        if (caught instanceof Error && caught.name === "AbortError") return;
        setError(caught instanceof Error ? caught.message : "Could not load database");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [browseView, calendarFilters, deferredFilters, section]);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return undefined;
    }
    const controller = new AbortController();
    void getDatabaseCurve(selectedId, controller.signal).then(setDetail).catch(() => undefined);
    return () => controller.abort();
  }, [selectedId]);

  function selectCalendarCurve(curve: CalendarCurve) {
    setSelectedId(curve.curve_id);
  }

  const title = calendarMode === "month"
    ? anchor.toLocaleDateString(undefined, { month: "long", year: "numeric" })
    : calendarMode === "week"
      ? `Week of ${startOfWeek(anchor).toLocaleDateString()}`
      : anchor.toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" });

  return (
    <div className="database-lab">
      <header className="lab-header">
        <a href="/" className="lab-brand"><Logo /></a>
        <nav>
          <button className={section === "browse" ? "active" : ""} onClick={() => setSection("browse")}>Browse</button>
          <button className={section === "analysis" ? "active" : ""} onClick={() => setSection("analysis")}>Analysis</button>
        </nav>
        <div className="lab-db-state">
          <Database size={15} />
          {status?.curves.toLocaleString() ?? "—"} curves
        </div>
      </header>

      {section === "analysis" ? (
        <Suspense fallback={<div className="analysis-loading">Loading analysis workspace</div>}>
          <AnalysisWorkspace selection={selection} />
        </Suspense>
      ) : (
        <div className="lab-browse-layout">
          <FilterRail filters={filters} options={options} total={total} onChange={setFilters} onReset={() => setFilters({})} />
          <main className="lab-browser">
            <div className="lab-toolbar">
              <button className="button compact" onClick={() => setFilters({ ...filters })}><RefreshCw size={14} /> Refresh</button>
              <span className="lab-toolbar-spacer" />
              <div className="lab-view-toggle">
                <button className={browseView === "list" ? "active" : ""} onClick={() => setBrowseView("list")}><List size={14} /> List</button>
                <button className={browseView === "calendar" ? "active" : ""} onClick={() => setBrowseView("calendar")}><CalendarDays size={14} /> Calendar</button>
              </div>
            </div>
            {error ? <div className="error-banner">{error}</div> : null}
            {browseView === "calendar" ? (
              <>
                <div className="lab-calendar-toolbar">
                  <button onClick={() => setAnchor(shiftAnchor(anchor, calendarMode, -1))}><ChevronLeft size={16} /></button>
                  <button onClick={() => setAnchor(new Date())}>Today</button>
                  <button onClick={() => setAnchor(shiftAnchor(anchor, calendarMode, 1))}><ChevronRight size={16} /></button>
                  <h1>{title}</h1>
                  <div className="lab-mode-toggle">
                    {(["month", "week", "day"] as CalendarMode[]).map((mode) => (
                      <button key={mode} className={calendarMode === mode ? "active" : ""} onClick={() => setCalendarMode(mode)}>{mode}</button>
                    ))}
                  </div>
                </div>
                {loading && !calendar ? <div className="lab-loading"><RefreshCw className="spin" /> Loading calendar</div> : null}
                <CalendarView response={calendar} anchor={anchor} mode={calendarMode} selectedId={selectedId} onSelect={selectCalendarCurve} />
                <div className="lab-calendar-legend">
                  <span className="ntype">n-type / forward</span>
                  <span className="ptype">p-type / reverse</span>
                  <span className="single">one-way / hysteresis NA</span>
                  {calendar?.truncated ? <strong>Visible range truncated at {calendar.limit.toLocaleString()} records</strong> : null}
                </div>
              </>
            ) : (
              <div className="lab-list-wrap">
                <table className="lab-list-table">
                  <thead><tr><th>Test time</th><th>Curve</th><th>Polarity</th><th>Direction</th><th>Hysteresis</th><th>Ion</th><th>Ioff</th><th>Vth</th><th>Source</th></tr></thead>
                  <tbody>
                    {(list?.items ?? []).map((curve) => (
                      <tr key={curve.curve_id} className={selectedId === curve.curve_id ? "selected" : ""} onClick={() => setSelectedId(curve.curve_id)}>
                        <td>{curve.test_time?.replace("T", " ").slice(0, 19) ?? "—"}</td>
                        <td>{curve.curve_id}</td>
                        <td>{curve.polarity}</td>
                        <td>{curve.direction}</td>
                        <td>{curve.hysteresis_v === null ? "NA" : fixed(curve.hysteresis_v, 3)}</td>
                        <td>{scientific(curve.ion)}</td>
                        <td>{scientific(curve.ioff)}</td>
                        <td>{fixed(curve.vth, 3)}</td>
                        <td>{curve.source_path}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </main>
          <aside className="lab-detail">
            <header>
              <span>Selected record</span>
              <strong>{detail?.curve_id ?? "No selection"}</strong>
            </header>
            {detail ? (
              <>
                <dl>
                  <div><dt>Test time</dt><dd>{detail.test_time?.replace("T", " ").slice(0, 19) ?? "—"}</dd></div>
                  <div><dt>Polarity</dt><dd>{detail.polarity}</dd></div>
                  <div><dt>Direction</dt><dd>{detail.direction}</dd></div>
                  <div><dt>Source</dt><dd>{detail.source_kind}</dd></div>
                  <div><dt>Hysteresis</dt><dd>{detail.hysteresis_v === null ? "NA" : `${fixed(detail.hysteresis_v, 3)} V`}</dd></div>
                </dl>
                <h2>Transfer curve preview</h2>
                <CurvePreview detail={detail} />
                <div className="lab-metrics">
                  <div><span>Vth</span><strong>{fixed(detail.vth, 3)} V</strong></div>
                  <div><span>SS</span><strong>{fixed(detail.ss_mv_dec, 1)} mV/dec</strong></div>
                  <div><span>Ion</span><strong>{scientific(detail.ion)} A</strong></div>
                  <div><span>Ioff</span><strong>{scientific(detail.ioff)} A</strong></div>
                </div>
              </>
            ) : <div className="lab-empty"><Activity size={22} /> Select a rug or table row</div>}
          </aside>
        </div>
      )}
    </div>
  );
}
