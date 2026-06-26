import { CalendarDays, CheckSquare, ChevronLeft, ChevronRight, MoveHorizontal, Square, ZoomIn } from "lucide-react";
import { useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent, WheelEvent as ReactWheelEvent } from "react";
import { listDatabaseCalendar } from "../api";
import { fixed, scientific } from "../format";
import type { CalendarCurve, CalendarCurveResponse, CurveFilters } from "../types";

type CalendarLevel = "year" | "month" | "week" | "day";

type TimelineCluster = {
  key: string;
  dateKey: string;
  startMinute: number;
  endMinute: number;
  items: CalendarCurve[];
};

const MINUTES_PER_DAY = 24 * 60;

function startOfDay(date: Date): Date {
  const next = new Date(date);
  next.setHours(0, 0, 0, 0);
  return next;
}

function endOfDay(date: Date): Date {
  const next = startOfDay(date);
  next.setHours(23, 59, 59, 999);
  return next;
}

function startOfWeek(date: Date): Date {
  const next = startOfDay(date);
  const weekday = (next.getDay() + 6) % 7;
  next.setDate(next.getDate() - weekday);
  return next;
}

function endOfWeek(date: Date): Date {
  const next = startOfWeek(date);
  next.setDate(next.getDate() + 6);
  return endOfDay(next);
}

function startOfMonth(date: Date): Date {
  return new Date(date.getFullYear(), date.getMonth(), 1);
}

function startOfCalendarMonth(date: Date): Date {
  return startOfWeek(startOfMonth(date));
}

function endOfCalendarMonth(date: Date): Date {
  const next = startOfCalendarMonth(date);
  next.setDate(next.getDate() + 41);
  return endOfDay(next);
}

function startOfYear(date: Date): Date {
  return new Date(date.getFullYear(), 0, 1);
}

function endOfYear(date: Date): Date {
  return new Date(date.getFullYear(), 11, 31, 23, 59, 59, 999);
}

function formatDateKey(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function parseDateKey(value: string): Date {
  return new Date(`${value}T00:00:00`);
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function minuteOfDay(iso: string): number {
  const time = new Date(iso);
  return time.getHours() * 60 + time.getMinutes() + time.getSeconds() / 60;
}

function curveTone(curve: Pick<CalendarCurve, "polarity" | "direction">): string {
  if (curve.direction === "single") return "single";
  return curve.polarity === "p-type" ? "ptype" : "ntype";
}

function formatRangeTitle(level: CalendarLevel, anchor: Date): string {
  if (level === "year") return String(anchor.getFullYear());
  if (level === "month") {
    return anchor.toLocaleDateString(undefined, { month: "long", year: "numeric" });
  }
  if (level === "week") {
    const start = startOfWeek(anchor);
    const end = new Date(start);
    end.setDate(end.getDate() + 6);
    return `${start.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric"
    })} - ${end.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year: "numeric"
    })}`;
  }
  return anchor.toLocaleDateString(undefined, {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric"
  });
}

function shiftAnchor(anchor: Date, level: CalendarLevel, delta: number): Date {
  const next = new Date(anchor);
  if (level === "year") next.setFullYear(next.getFullYear() + delta);
  else if (level === "month") next.setMonth(next.getMonth() + delta);
  else if (level === "week") next.setDate(next.getDate() + delta * 7);
  else next.setDate(next.getDate() + delta);
  return next;
}

function visibleMonthDates(anchor: Date): Date[] {
  const start = startOfCalendarMonth(anchor);
  return Array.from({ length: 42 }, (_, index) => {
    const next = new Date(start);
    next.setDate(start.getDate() + index);
    return next;
  });
}

function visibleWeekDates(anchor: Date): Date[] {
  const start = startOfWeek(anchor);
  return Array.from({ length: 7 }, (_, index) => {
    const next = new Date(start);
    next.setDate(start.getDate() + index);
    return next;
  });
}

function intersectDateRange(
  baseStart: Date,
  baseEnd: Date,
  filters: CurveFilters
): { start: Date; end: Date } | null {
  let start = baseStart;
  let end = baseEnd;
  if (filters.date_from) {
    const nextStart = startOfDay(parseDateKey(filters.date_from));
    if (nextStart > start) start = nextStart;
  }
  if (filters.date_to) {
    const nextEnd = endOfDay(parseDateKey(filters.date_to));
    if (nextEnd < end) end = nextEnd;
  }
  if (start > end) return null;
  return { start, end };
}

function dayItemsMap(response: CalendarCurveResponse | null) {
  const grouped = new Map<string, CalendarCurve[]>();
  for (const item of response?.items ?? []) {
    const key = item.test_time.slice(0, 10);
    grouped.set(key, [...(grouped.get(key) ?? []), item]);
  }
  return grouped;
}

function monthSummaries(response: CalendarCurveResponse | null, anchor: Date) {
  const counts = Array.from({ length: 12 }, () => 0);
  for (const [dayKey, count] of Object.entries(response?.day_counts ?? {})) {
    if (!dayKey.startsWith(`${anchor.getFullYear()}-`)) continue;
    const month = Number(dayKey.slice(5, 7)) - 1;
    counts[month] += count;
  }
  return counts.map((count, month) => ({
    month,
    count,
    date: new Date(anchor.getFullYear(), month, 1)
  }));
}

function visibleHoursFromZoom(zoom: number): number {
  return 24 - (zoom / 100) * 21;
}

function clusterGapMinutes(visibleHours: number): number {
  return Math.max(6, Math.round(visibleHours * 1.9));
}

function buildClusters(items: CalendarCurve[], mergeGapMinutes: number): TimelineCluster[] {
  const sorted = [...items].sort((left, right) => left.test_time.localeCompare(right.test_time));
  const gapMinutes = mergeGapMinutes;
  const clusters: TimelineCluster[] = [];
  for (const item of sorted) {
    const currentMinute = minuteOfDay(item.test_time);
    const dateKey = item.test_time.slice(0, 10);
    const last = clusters.at(-1);
    if (!last || last.dateKey !== dateKey || currentMinute - last.endMinute > gapMinutes) {
      clusters.push({
        key: `${dateKey}-${item.curve_id}`,
        dateKey,
        startMinute: currentMinute,
        endMinute: currentMinute,
        items: [item]
      });
      continue;
    }
    last.endMinute = currentMinute;
    last.items.push(item);
  }
  return clusters;
}

function ticksForWindow(startMinute: number, visibleHours: number): number[] {
  const stepHours = visibleHours <= 4 ? 1 : visibleHours <= 10 ? 2 : 4;
  const stepMinutes = stepHours * 60;
  const ticks: number[] = [];
  const roundedStart = Math.ceil(startMinute / stepMinutes) * stepMinutes;
  ticks.push(startMinute);
  for (let minute = roundedStart; minute < startMinute + visibleHours * 60; minute += stepMinutes) {
    ticks.push(minute);
  }
  ticks.push(startMinute + visibleHours * 60);
  return [...new Set(ticks.map((value) => Math.round(value)))];
}

function formatMinuteLabel(value: number): string {
  const hour = Math.floor(value / 60);
  const minute = Math.round(value % 60);
  return `${String(hour).padStart(2, "0")}:${String(minute).padStart(2, "0")}`;
}

function clusterSummary(cluster: TimelineCluster): string {
  return String(cluster.items.length);
}

function clusterSecondary(cluster: TimelineCluster): string {
  if (cluster.items.length === 1) {
    return `${cluster.items[0].test_time.slice(11, 19)}  ${cluster.items[0].source_kind}`;
  }
  return `${formatMinuteLabel(cluster.startMinute)} - ${formatMinuteLabel(cluster.endMinute)}`;
}

function monthTrackClusters(items: CalendarCurve[]): TimelineCluster[] {
  return buildClusters(items, 40);
}

function toneForCluster(cluster: TimelineCluster): string {
  return curveTone(cluster.items[0]);
}

function positionPercent(minute: number, startMinute: number, visibleMinutes: number): number {
  return ((minute - startMinute) / Math.max(visibleMinutes, 1)) * 100;
}

function SelectionList({
  title,
  clusters,
  selectedIds,
  selectedCurveId,
  onSelectAll,
  onSelectCurve,
  onToggleSelection
}: {
  title: string;
  clusters: TimelineCluster[];
  selectedIds: Set<string>;
  selectedCurveId: string | null;
  onSelectAll: () => void;
  onSelectCurve: (curveId: string) => void;
  onToggleSelection: (curveId: string) => void;
}) {
  const items = clusters.flatMap((cluster) => cluster.items);
  return (
    <section className="database-selection-sheet">
      <header>
        <div className="database-selection-sheet-meta">
          <strong>{title}</strong>
          <span>{items.length.toLocaleString()} curves</span>
        </div>
        <button type="button" className="button secondary compact" onClick={onSelectAll} disabled={items.length === 0}>
          <CheckSquare size={15} />
          Select all
        </button>
      </header>
      <div className="database-selection-sheet-list">
        {items.map((curve) => (
          <div
            key={curve.curve_id}
            className={`database-selection-item${selectedCurveId === curve.curve_id ? " active" : ""}`}
          >
            <button type="button" className="database-selection-main" onClick={() => onSelectCurve(curve.curve_id)}>
              <span className={`database-tone-dot ${curveTone(curve)}`} />
              <strong>{curve.test_time.slice(11, 19)}</strong>
              <b>{curve.curve_id}</b>
              <small>{curve.source_kind}</small>
              <em>{scientific(curve.ion)}</em>
            </button>
            <button
              type="button"
              className="database-selection-toggle"
              onClick={() => onToggleSelection(curve.curve_id)}
              title={selectedIds.has(curve.curve_id) ? "Unselect curve" : "Select curve"}
            >
              {selectedIds.has(curve.curve_id) ? <CheckSquare size={15} /> : <Square size={15} />}
            </button>
          </div>
        ))}
      </div>
    </section>
  );
}

export function DatabaseCalendarPanel({
  filters,
  selectedCurveId,
  selectedIds,
  onSelectCurve,
  onSelectCurveIds,
  onToggleSelection
}: {
  filters: CurveFilters;
  selectedCurveId: string | null;
  selectedIds: Set<string>;
  onSelectCurve: (curveId: string) => void;
  onSelectCurveIds: (curveIds: string[]) => void;
  onToggleSelection: (curveId: string) => void;
}) {
  const deferredFilters = useDeferredValue(filters);
  const [level, setLevel] = useState<CalendarLevel>("year");
  const [anchor, setAnchor] = useState(() => new Date());
  const [zoom, setZoom] = useState(42);
  const [panMinutes, setPanMinutes] = useState(0);
  const [response, setResponse] = useState<CalendarCurveResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedClusterKeys, setSelectedClusterKeys] = useState<string[]>([]);
  const dragStateRef = useRef<{ startX: number; startPan: number } | null>(null);

  const requestedRange = useMemo(() => {
    if (level === "year") return { start: startOfYear(anchor), end: endOfYear(anchor) };
    if (level === "month") return { start: startOfCalendarMonth(anchor), end: endOfCalendarMonth(anchor) };
    if (level === "week") return { start: startOfWeek(anchor), end: endOfWeek(anchor) };
    return { start: startOfDay(anchor), end: endOfDay(anchor) };
  }, [anchor, level]);

  const effectiveRange = useMemo(
    () => intersectDateRange(requestedRange.start, requestedRange.end, deferredFilters),
    [deferredFilters, requestedRange]
  );

  useEffect(() => {
    if (!effectiveRange) {
      setResponse({ items: [], day_counts: {}, truncated: false, limit: 10000 });
      return undefined;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    void listDatabaseCalendar(
      {
        ...deferredFilters,
        date_from: formatDateKey(effectiveRange.start),
        date_to: formatDateKey(effectiveRange.end)
      },
      controller.signal
    )
      .then(setResponse)
      .catch((caught) => {
        if (caught instanceof Error && caught.name === "AbortError") return;
        setError(caught instanceof Error ? caught.message : "Could not load calendar");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [deferredFilters, effectiveRange]);

  useEffect(() => {
    setSelectedClusterKeys([]);
  }, [anchor, level, deferredFilters]);

  const months = useMemo(() => monthSummaries(response, anchor), [anchor, response]);
  const groupedByDay = useMemo(() => dayItemsMap(response), [response]);
  const monthDates = useMemo(() => visibleMonthDates(anchor), [anchor]);
  const weekDates = useMemo(() => visibleWeekDates(anchor), [anchor]);
  const visibleHours = visibleHoursFromZoom(zoom);
  const visibleMinutes = visibleHours * 60;
  const maxPanMinutes = Math.max(0, MINUTES_PER_DAY - visibleMinutes);
  const clampedPanMinutes = clampNumber(panMinutes, 0, maxPanMinutes);
  const tickMinutes = useMemo(() => ticksForWindow(clampedPanMinutes, visibleHours), [clampedPanMinutes, visibleHours]);
  const mergeGapMinutes = clusterGapMinutes(visibleHours);

  const dayClusters = useMemo(() => {
    const key = formatDateKey(anchor);
    return buildClusters(groupedByDay.get(key) ?? [], mergeGapMinutes)
      .filter((cluster) => cluster.endMinute >= clampedPanMinutes && cluster.startMinute <= clampedPanMinutes + visibleMinutes);
  }, [anchor, clampedPanMinutes, groupedByDay, mergeGapMinutes, visibleMinutes]);

  const weekClustersByDay = useMemo(() => {
    return new Map(
      weekDates.map((date) => {
        const key = formatDateKey(date);
        const clusters = buildClusters(groupedByDay.get(key) ?? [], mergeGapMinutes)
          .filter((cluster) => cluster.endMinute >= clampedPanMinutes && cluster.startMinute <= clampedPanMinutes + visibleMinutes);
        return [key, clusters] as const;
      })
    );
  }, [weekDates, groupedByDay, mergeGapMinutes, clampedPanMinutes, visibleMinutes]);

  const selectedClusters = useMemo(() => {
    const allClusters = level === "day"
      ? dayClusters
      : weekDates.flatMap((date) => weekClustersByDay.get(formatDateKey(date)) ?? []);
    return allClusters.filter((cluster) => selectedClusterKeys.includes(cluster.key));
  }, [dayClusters, level, selectedClusterKeys, weekClustersByDay, weekDates]);
  const activeClusters = selectedClusters;

  function jumpTo(nextLevel: CalendarLevel, date: Date) {
    setAnchor(date);
    setLevel(nextLevel);
  }

  function updateZoomWithWheel(deltaY: number) {
    setZoom((current) => clampNumber(current + Math.sign(deltaY) * 6, 0, 100));
  }

  function attachInteractionProps(): {
    onWheel: (event: ReactWheelEvent<HTMLDivElement>) => void;
    onPointerDown: (event: ReactPointerEvent<HTMLDivElement>) => void;
    onPointerMove: (event: ReactPointerEvent<HTMLDivElement>) => void;
    onPointerUp: () => void;
    onPointerCancel: () => void;
  } {
    return {
      onWheel(event: ReactWheelEvent<HTMLDivElement>) {
        event.preventDefault();
        updateZoomWithWheel(event.deltaY);
      },
      onPointerDown(event: ReactPointerEvent<HTMLDivElement>) {
        const target = event.target;
        if (target instanceof HTMLElement && target.closest(".database-range-pill")) {
          return;
        }
        dragStateRef.current = { startX: event.clientX, startPan: clampedPanMinutes };
        event.currentTarget.setPointerCapture(event.pointerId);
      },
      onPointerMove(event: ReactPointerEvent<HTMLDivElement>) {
        if (!dragStateRef.current) return;
        const pixelsPerMinute = Math.max(0.2, event.currentTarget.clientWidth / Math.max(visibleMinutes, 1));
        const deltaMinutes = (dragStateRef.current.startX - event.clientX) / pixelsPerMinute;
        setPanMinutes(clampNumber(dragStateRef.current.startPan + deltaMinutes, 0, maxPanMinutes));
      },
      onPointerUp() {
        dragStateRef.current = null;
      },
      onPointerCancel() {
        dragStateRef.current = null;
      }
    };
  }

  function toggleCluster(clusterKey: string, curveId: string) {
    setSelectedClusterKeys((current) => {
      if (current.includes(clusterKey)) {
        return current.filter((entry) => entry !== clusterKey);
      }
      return [clusterKey];
    });
    onSelectCurve(curveId);
  }

  function selectAllActiveClusterCurves() {
    const curveIds = activeClusters.flatMap((cluster) => cluster.items.map((curve) => curve.curve_id));
    onSelectCurveIds(curveIds);
  }

  function renderClusterPill(
    cluster: TimelineCluster,
    startMinute: number,
    widthMinutes: number,
    variant: "month" | "week" | "day"
  ) {
    const left = positionPercent(cluster.startMinute, startMinute, widthMinutes);
    const width = Math.max(
      variant === "day" ? 1.4 : variant === "week" ? 1.8 : 2.2,
      positionPercent(cluster.endMinute + 8, cluster.startMinute, widthMinutes)
    );
    const selected = selectedClusterKeys.includes(cluster.key);
    const widthClass =
      variant === "month"
        ? "compact"
        : width < 2.2
          ? "tiny"
          : width < 4.8
            ? "narrow"
            : "regular";
    return (
      <button
        key={cluster.key}
        type="button"
        className={`database-range-pill ${toneForCluster(cluster)} ${variant} ${widthClass}${selected ? " selected" : ""}`}
        style={{ left: `${left}%`, width: `${width}%` }}
        onPointerDown={(event) => {
          event.stopPropagation();
        }}
        onClick={() => toggleCluster(cluster.key, cluster.items[0].curve_id)}
        title={
          cluster.items.length === 1
            ? `${cluster.items[0].test_time.replace("T", " ").slice(0, 19)} | ${cluster.items[0].curve_id}`
            : `${cluster.items.length} curves between ${formatMinuteLabel(cluster.startMinute)} and ${formatMinuteLabel(cluster.endMinute)}`
        }
      >
        <span>{clusterSummary(cluster)}</span>
      </button>
    );
  }

  function renderTimelineTicks() {
    return (
      <div className="database-time-ticks">
        {tickMinutes.map((minute) => {
          const left = positionPercent(minute, clampedPanMinutes, visibleMinutes);
          return (
            <div key={minute} className="database-time-tick" style={{ left: `${left}%` }}>
              <span>{formatMinuteLabel(minute)}</span>
            </div>
          );
        })}
      </div>
    );
  }

  function renderYearView() {
    return (
      <div className="database-calendar-year-grid">
        {months.map((entry) => (
          <button
            key={entry.month}
            type="button"
            className="database-calendar-year-card"
            onClick={() => jumpTo("month", entry.date)}
          >
            <span>{entry.date.toLocaleDateString(undefined, { month: "short" })}</span>
            <strong>{entry.count.toLocaleString()}</strong>
            <small>{entry.count === 1 ? "measurement" : "measurements"}</small>
          </button>
        ))}
      </div>
    );
  }

  function renderMonthView() {
    return (
      <div className="database-month-grid">
        {monthDates.map((date) => {
          const key = formatDateKey(date);
          const items = groupedByDay.get(key) ?? [];
          const clusters = monthTrackClusters(items);
          const outside = date.getMonth() !== anchor.getMonth();
          return (
            <section key={key} className={`database-month-cell${outside ? " outside" : ""}`}>
              <button type="button" className="database-month-head" onClick={() => jumpTo("week", date)}>
                <strong>{date.getDate()}</strong>
                <span>{date.toLocaleDateString(undefined, { weekday: "short" })}</span>
                <b>{items.length.toLocaleString()}</b>
              </button>
              <div className="database-month-track-shell">
                <div className="database-month-track">
                  {clusters.map((cluster) => (
                    <button
                      key={cluster.key}
                      type="button"
                      className={`database-range-pill ${toneForCluster(cluster)} month${selectedClusterKeys.includes(cluster.key) ? " selected" : ""}`}
                      style={{
                        left: `${positionPercent(cluster.startMinute, 0, MINUTES_PER_DAY)}%`,
                        width: `${Math.max(2.5, positionPercent(cluster.endMinute + 8, cluster.startMinute, MINUTES_PER_DAY))}%`
                      }}
                      onClick={() => {
                        setSelectedClusterKeys([cluster.key]);
                        setAnchor(date);
                        setLevel("day");
                        onSelectCurve(cluster.items[0].curve_id);
                      }}
                    />
                  ))}
                </div>
                <div className="database-month-track-labels">
                  <span>00</span>
                  <span>12</span>
                  <span>24</span>
                </div>
              </div>
              {clusters[0] ? (
                <div className="database-month-summary">
                  <strong>{clusterSummary(clusters[0])}</strong>
                  <small>{clusterSecondary(clusters[0])}</small>
                </div>
              ) : (
                <div className="database-month-summary empty">
                  <small>No tests</small>
                </div>
              )}
            </section>
          );
        })}
      </div>
    );
  }

  function renderWeekView() {
    return (
      <div className="database-time-browser">
      <div className="database-time-hint">
          <span><ZoomIn size={14} /> Scroll to zoom</span>
          <span><MoveHorizontal size={14} /> Drag to pan</span>
          <span><i className="database-legend-dot ntype" /> n-type paired/forward</span>
          <span><i className="database-legend-dot ptype" /> p-type paired/reverse</span>
          <span><i className="database-legend-dot single" /> single / no hysteresis pair</span>
        </div>
        <div className="database-week-board" {...attachInteractionProps()}>
          {renderTimelineTicks()}
          {weekDates.map((date) => {
            const key = formatDateKey(date);
            const clusters = weekClustersByDay.get(key) ?? [];
            return (
              <section key={key} className="database-week-row">
                <button type="button" className="database-week-label" onClick={() => jumpTo("day", date)}>
                  <strong>{date.toLocaleDateString(undefined, { weekday: "short" })}</strong>
                  <span>{date.toLocaleDateString(undefined, { month: "short", day: "numeric" })}</span>
                </button>
                <div className="database-week-lane">
                  {clusters.map((cluster) => renderClusterPill(cluster, clampedPanMinutes, visibleMinutes, "week"))}
                </div>
              </section>
            );
          })}
        </div>
        {selectedClusters.length > 0 ? (
          <SelectionList
            title="Selected week range"
            clusters={activeClusters}
            selectedIds={selectedIds}
            selectedCurveId={selectedCurveId}
            onSelectAll={selectAllActiveClusterCurves}
            onSelectCurve={onSelectCurve}
            onToggleSelection={onToggleSelection}
          />
        ) : (
          <div className="database-calendar-empty-state">Select a range to inspect the curves inside it.</div>
        )}
      </div>
    );
  }

  function renderDayView() {
    return (
      <div className="database-time-browser">
        <div className="database-time-hint">
          <span><ZoomIn size={14} /> Scroll to zoom</span>
          <span><MoveHorizontal size={14} /> Drag to pan</span>
          <span><i className="database-legend-dot ntype" /> n-type paired/forward</span>
          <span><i className="database-legend-dot ptype" /> p-type paired/reverse</span>
          <span><i className="database-legend-dot single" /> single / no hysteresis pair</span>
          <span>{fixed(visibleHours, 1)} h window</span>
        </div>
        <div className="database-day-board" {...attachInteractionProps()}>
          {renderTimelineTicks()}
          <div className="database-day-lane">
            {dayClusters.map((cluster) => renderClusterPill(cluster, clampedPanMinutes, visibleMinutes, "day"))}
          </div>
        </div>
        {selectedClusters.length > 0 ? (
          <SelectionList
            title="Selected day range"
            clusters={activeClusters}
            selectedIds={selectedIds}
            selectedCurveId={selectedCurveId}
            onSelectAll={selectAllActiveClusterCurves}
            onSelectCurve={onSelectCurve}
            onToggleSelection={onToggleSelection}
          />
        ) : (
          <div className="database-calendar-empty-state">Select a day range to inspect and batch-select its curves.</div>
        )}
      </div>
    );
  }

  return (
    <section className="database-calendar-panel">
      <div className="database-calendar-toolbar">
        <div className="database-calendar-toolbar-group">
          <button type="button" className="button secondary compact" onClick={() => setAnchor(shiftAnchor(anchor, level, -1))}>
            <ChevronLeft size={15} />
          </button>
          <button type="button" className="button secondary compact" onClick={() => setAnchor(new Date())}>
            Today
          </button>
          <button type="button" className="button secondary compact" onClick={() => setAnchor(shiftAnchor(anchor, level, 1))}>
            <ChevronRight size={15} />
          </button>
        </div>
        <h3>
          <CalendarDays size={16} />
          {formatRangeTitle(level, anchor)}
        </h3>
        <div className="database-calendar-levels">
          {(["year", "month", "week", "day"] as CalendarLevel[]).map((entry) => (
            <button
              key={entry}
              type="button"
              className={level === entry ? "active" : ""}
              onClick={() => setLevel(entry)}
            >
              {entry}
            </button>
          ))}
        </div>
      </div>
      {error ? <div className="error-banner">{error}</div> : null}
      {loading ? <div className="database-calendar-loading">Loading calendar view</div> : null}
      {!loading && level === "year" ? renderYearView() : null}
      {!loading && level === "month" ? renderMonthView() : null}
      {!loading && level === "week" ? renderWeekView() : null}
      {!loading && level === "day" ? renderDayView() : null}
      <div className="database-calendar-footer">
        <span>Calendar and list now live together inside Database.</span>
        {response?.truncated ? <strong>Calendar response truncated at {response.limit.toLocaleString()} records.</strong> : null}
      </div>
    </section>
  );
}
