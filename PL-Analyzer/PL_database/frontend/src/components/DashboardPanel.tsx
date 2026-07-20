import { useEffect, useMemo, useRef, useState } from "react";
import type { DashboardSummary, HealthPayload } from "../types";
import { StatusCard } from "./StatusCard";

interface DashboardPanelProps {
  dashboard: DashboardSummary | null;
  health: HealthPayload | null;
}

type HealthDeltaState = {
  imported_files: number;
  spectra_count: number;
  failed_imports: number;
  database_size_mb: number;
};

type DistributionItem = {
  label: string;
  value: number;
  percent: number;
  color: string;
};

const EMPTY_DELTAS: HealthDeltaState = {
  imported_files: 0,
  spectra_count: 0,
  failed_imports: 0,
  database_size_mb: 0
};

const TYPE_COLORS = ["#14758B", "#D88A3D", "#7E96A8", "#C7D6E0"];
const ACQUISITION_COLORS = ["#14758B", "#4AA39A", "#D88A3D", "#C96A50", "#7E96A8", "#C7D6E0"];

export function DashboardPanel(props: DashboardPanelProps) {
  const typeEntries = Object.entries(props.dashboard?.type_counts ?? {});
  const acquisitionEntries = Object.entries(props.dashboard?.acquisition_counts ?? {});
  const timeline = props.dashboard?.measurement_timeline ?? [];
  const previousDashboardRef = useRef<DashboardSummary | null>(null);
  const [deltas, setDeltas] = useState<HealthDeltaState>(EMPTY_DELTAS);

  useEffect(() => {
    if (!props.dashboard) {
      return;
    }
    const previous = previousDashboardRef.current;
    if (previous) {
      setDeltas({
        imported_files: Math.max(0, (props.dashboard.imported_files ?? 0) - (previous.imported_files ?? 0)),
        spectra_count: Math.max(0, (props.dashboard.spectra_count ?? 0) - (previous.spectra_count ?? 0)),
        failed_imports: Math.max(0, (props.dashboard.failed_imports ?? 0) - (previous.failed_imports ?? 0)),
        database_size_mb: Math.max(0, Math.round(((props.dashboard.database_size_mb ?? 0) - (previous.database_size_mb ?? 0)) * 1000))
      });
    }
    previousDashboardRef.current = props.dashboard;
  }, [
    props.dashboard?.database_size_mb,
    props.dashboard?.failed_imports,
    props.dashboard?.imported_files,
    props.dashboard?.spectra_count
  ]);

  const healthDeltaMb = useMemo(() => {
    if (deltas.database_size_mb <= 0) {
      return 0;
    }
    return Number((deltas.database_size_mb / 1000).toFixed(3));
  }, [deltas.database_size_mb]);

  return (
    <section className="panel-grid">
      <div className="card card-span-3">
        <div className="card-head">
          <div>
            <p className="eyebrow">Overview</p>
            <h2>Database health</h2>
          </div>
          <span className={`pill ${props.health?.status === "ok" ? "pill-good" : "pill-neutral"}`}>
            API {props.health?.status ?? "loading"}
          </span>
        </div>
        <div className="status-grid">
          <StatusCard
            label="Imported WIP files"
            value={String(props.dashboard?.imported_files ?? 0)}
            delta={deltas.imported_files}
            detail="Distinct real source contents already indexed"
          />
          <StatusCard
            label="Stored spectra"
            value={String(props.dashboard?.spectra_count ?? 0)}
            delta={deltas.spectra_count}
            detail="HDF5 traces linked from SQLite metadata"
          />
          <StatusCard
            label="Failed imports"
            value={String(props.dashboard?.failed_imports ?? 0)}
            delta={deltas.failed_imports}
            detail="Jobs marked failed or partially failed"
          />
          <StatusCard
            label="Database size"
            value={`${props.dashboard?.database_size_mb ?? 0} MB`}
            delta={healthDeltaMb > 0 ? healthDeltaMb : 0}
            detail="SQLite + HDF5 footprint in the local data folder"
          />
        </div>
      </div>

      <DistributionCard
        eyebrow="Type mix"
        title="PL vs Raman"
        entries={typeEntries}
        formatLabel={friendlyTypeLabel}
        colors={TYPE_COLORS}
        emptyMessage="Type distribution will appear after import."
      />

      <DistributionCard
        eyebrow="Acquisition mix"
        title="Point, line, map, series"
        entries={acquisitionEntries}
        formatLabel={friendlyAcquisitionLabel}
        colors={ACQUISITION_COLORS}
        emptyMessage="Acquisition-mode distribution will appear after import."
      />

      <div className="card card-span-1 dashboard-visual-card">
        <div className="card-head">
          <div>
            <p className="eyebrow">Timeline</p>
            <h2>Measurements over time</h2>
          </div>
        </div>
        {timeline.length > 0 ? (
          <TimelineBarChart timeline={timeline} />
        ) : (
          <p className="empty-state">Real measurement-time distribution will appear after import.</p>
        )}
      </div>
    </section>
  );
}

function DistributionCard(props: {
  eyebrow: string;
  title: string;
  entries: Array<[string, number]>;
  formatLabel: (value: string) => string;
  colors: string[];
  emptyMessage: string;
}) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const distribution = buildDistributionItems(props.entries, props.colors, props.formatLabel);
  const total = distribution.reduce((sum, item) => sum + item.value, 0);
  const activeIndex = hoveredIndex ?? 0;
  const activeItem = distribution[activeIndex] ?? null;

  return (
    <div className="card card-span-1 dashboard-visual-card">
      <div className="card-head">
        <div>
          <p className="eyebrow">{props.eyebrow}</p>
          <h2>{props.title}</h2>
        </div>
      </div>
      {distribution.length > 0 ? (
        <div className="dashboard-donut-layout">
          <DonutChart
            activeIndex={hoveredIndex}
            items={distribution}
            onActiveChange={setHoveredIndex}
            total={total}
          />
          <div className="dashboard-donut-meta">
            <div className="dashboard-donut-highlight">
              <span className="dashboard-donut-highlight-label">{activeItem?.label ?? "Total"}</span>
              <strong>{activeItem ? `${activeItem.percent.toFixed(1)}%` : "100%"}</strong>
              <span>{activeItem ? `${activeItem.value} datasets` : `${total} datasets`}</span>
            </div>
            <div className="dashboard-donut-legend">
              {distribution.map((item, index) => (
                <button
                  key={`${item.label}-${index}`}
                  className={`dashboard-legend-item ${hoveredIndex === index ? "dashboard-legend-item-active" : ""}`}
                  onMouseEnter={() => setHoveredIndex(index)}
                  onMouseLeave={() => setHoveredIndex(null)}
                  type="button"
                >
                  <span className="dashboard-legend-dot" style={{ backgroundColor: item.color }} />
                  <span className="dashboard-legend-copy">
                    <strong>{item.label}</strong>
                    <small>{item.value} datasets</small>
                  </span>
                  <span className="dashboard-legend-percent">{item.percent.toFixed(1)}%</span>
                </button>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <p className="empty-state">{props.emptyMessage}</p>
      )}
    </div>
  );
}

function DonutChart(props: {
  items: DistributionItem[];
  total: number;
  activeIndex: number | null;
  onActiveChange: (index: number | null) => void;
}) {
  const size = 230;
  const center = size / 2;
  const outerRadius = 84;
  const innerRadius = 50;
  let currentAngle = -Math.PI / 2;

  return (
    <div className="dashboard-donut-shell">
      <svg
        aria-label="Distribution chart"
        className="dashboard-donut-svg"
        role="img"
        viewBox={`0 0 ${size} ${size}`}
      >
        <defs>
          <filter id="dashboard-donut-shadow" x="-30%" y="-30%" width="160%" height="160%">
            <feDropShadow dx="0" dy="10" stdDeviation="10" floodColor="rgba(17,34,59,0.16)" />
          </filter>
        </defs>
        <circle
          cx={center}
          cy={center}
          fill="none"
          r={(outerRadius + innerRadius) / 2}
          stroke="rgba(227, 236, 242, 0.95)"
          strokeWidth={outerRadius - innerRadius}
        />
        {props.items.map((item, index) => {
          const sliceAngle = props.total > 0 ? (item.value / props.total) * Math.PI * 2 : 0;
          const startAngle = currentAngle;
          const endAngle = currentAngle + sliceAngle;
          currentAngle = endAngle;
          const isActive = props.activeIndex === index;
          const midAngle = (startAngle + endAngle) / 2;
          const offset = isActive ? 8 : 0;
          const translateX = Math.cos(midAngle) * offset;
          const translateY = Math.sin(midAngle) * offset;

          return (
            <path
              key={`${item.label}-${index}`}
              d={describeDonutArc(center, center, outerRadius, innerRadius, startAngle, endAngle)}
              fill={item.color}
              filter={isActive ? "url(#dashboard-donut-shadow)" : undefined}
              onMouseEnter={() => props.onActiveChange(index)}
              onMouseLeave={() => props.onActiveChange(null)}
              style={{
                cursor: "pointer",
                transform: `translate(${translateX}px, ${translateY}px)`,
                transformOrigin: `${center}px ${center}px`,
                transition: "transform 180ms ease, filter 180ms ease"
              }}
            />
          );
        })}
        <circle cx={center} cy={center} fill="rgba(250, 252, 254, 0.98)" r={innerRadius - 4} />
        <text className="dashboard-donut-center-value" textAnchor="middle" x={center} y={center - 2}>
          {props.total}
        </text>
        <text className="dashboard-donut-center-label" textAnchor="middle" x={center} y={center + 18}>
          datasets
        </text>
      </svg>
    </div>
  );
}

function TimelineBarChart(props: {
  timeline: Array<{
    bucket: string;
    count: number;
    granularity: string;
  }>;
}) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const activeIndex = hoveredIndex ?? props.timeline.length - 1;
  const activeItem = props.timeline[activeIndex] ?? props.timeline[0];
  const peakItem = props.timeline.reduce((best, current) => (current.count > best.count ? current : best), props.timeline[0]);
  const tickIndices = buildTickIndices(props.timeline.length);
  const width = 470;
  const height = 250;
  const left = 18;
  const right = 12;
  const top = 18;
  const bottom = 44;
  const chartHeight = 166;
  const innerWidth = width - left - right;
  const slotWidth = innerWidth / props.timeline.length;
  const barWidth = Math.max(8, Math.min(16, slotWidth * 0.44));
  const maxCount = Math.max(...props.timeline.map((item) => item.count), 1);

  return (
    <div className="dashboard-timeline-shell">
      <div className="dashboard-timeline-summary">
        <div>
          <span>Selected</span>
          <strong>{formatTimelineLabel(activeItem.bucket)}</strong>
        </div>
        <div>
          <span>Datasets</span>
          <strong>{activeItem.count}</strong>
        </div>
        <div>
          <span>Peak month</span>
          <strong>{formatTimelineLabel(peakItem.bucket)}</strong>
        </div>
      </div>
      <svg className="dashboard-timeline-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img" aria-label="Measurement timeline by month">
        <defs>
          <linearGradient id="timeline-bar-gradient" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#95DDD5" />
            <stop offset="100%" stopColor="#166D84" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75].map((ratio) => {
          const y = top + chartHeight * ratio;
          return (
            <line
              key={ratio}
              className="dashboard-timeline-grid"
              x1={left}
              x2={width - right}
              y1={y}
              y2={y}
            />
          );
        })}
        <line className="dashboard-timeline-axis" x1={left} x2={width - right} y1={top + chartHeight} y2={top + chartHeight} />
        {props.timeline.map((item, index) => {
          const x = left + index * slotWidth + (slotWidth - barWidth) / 2;
          const barHeight = Math.max(4, (item.count / maxCount) * chartHeight);
          const y = top + chartHeight - barHeight;
          const isActive = index === activeIndex;
          const label = formatTimelineLabel(item.bucket);
          return (
            <g
              key={`${item.bucket}-${index}`}
              className={`dashboard-timeline-bar-group ${isActive ? "dashboard-timeline-bar-group-active" : ""}`}
              onMouseEnter={() => setHoveredIndex(index)}
              onMouseLeave={() => setHoveredIndex(null)}
            >
              <path
                className="dashboard-timeline-bar-shadow"
                d={buildRoundedTopBarPath(x, y, barWidth, barHeight, Math.min(barWidth / 2, 9))}
              />
              <path
                className="dashboard-timeline-bar"
                d={buildRoundedTopBarPath(x, y, barWidth, barHeight, Math.min(barWidth / 2, 9))}
              />
              <rect
                fill="transparent"
                height={chartHeight + bottom}
                width={slotWidth}
                x={left + index * slotWidth}
                y={0}
              />
              {tickIndices.has(index) ? (
                <text className="dashboard-timeline-label" textAnchor="middle" x={left + index * slotWidth + slotWidth / 2} y={height - 12}>
                  {label}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function buildDistributionItems(
  entries: Array<[string, number]>,
  colors: string[],
  formatLabel: (value: string) => string
): DistributionItem[] {
  const total = entries.reduce((sum, [, value]) => sum + value, 0);
  return entries
    .filter(([, value]) => value > 0)
    .map(([label, value], index) => ({
      label: formatLabel(label),
      value,
      percent: total > 0 ? (value / total) * 100 : 0,
      color: colors[index % colors.length]
    }));
}

function describeDonutArc(
  centerX: number,
  centerY: number,
  outerRadius: number,
  innerRadius: number,
  startAngle: number,
  endAngle: number
): string {
  const tau = Math.PI * 2;
  const safeEndAngle = endAngle - startAngle >= tau ? endAngle - 0.0001 : endAngle;
  const largeArcFlag = safeEndAngle - startAngle > Math.PI ? 1 : 0;

  const outerStart = polarToCartesian(centerX, centerY, outerRadius, startAngle);
  const outerEnd = polarToCartesian(centerX, centerY, outerRadius, safeEndAngle);
  const innerEnd = polarToCartesian(centerX, centerY, innerRadius, safeEndAngle);
  const innerStart = polarToCartesian(centerX, centerY, innerRadius, startAngle);

  return [
    `M ${outerStart.x} ${outerStart.y}`,
    `A ${outerRadius} ${outerRadius} 0 ${largeArcFlag} 1 ${outerEnd.x} ${outerEnd.y}`,
    `L ${innerEnd.x} ${innerEnd.y}`,
    `A ${innerRadius} ${innerRadius} 0 ${largeArcFlag} 0 ${innerStart.x} ${innerStart.y}`,
    "Z"
  ].join(" ");
}

function polarToCartesian(centerX: number, centerY: number, radius: number, angle: number) {
  return {
    x: centerX + Math.cos(angle) * radius,
    y: centerY + Math.sin(angle) * radius
  };
}

function buildRoundedTopBarPath(x: number, y: number, width: number, height: number, radius: number): string {
  const safeHeight = Math.max(1, height);
  const safeRadius = Math.min(radius, width / 2, safeHeight);
  const bottom = y + safeHeight;

  return [
    `M ${x} ${bottom}`,
    `L ${x} ${y + safeRadius}`,
    `Q ${x} ${y} ${x + safeRadius} ${y}`,
    `L ${x + width - safeRadius} ${y}`,
    `Q ${x + width} ${y} ${x + width} ${y + safeRadius}`,
    `L ${x + width} ${bottom}`,
    "Z"
  ].join(" ");
}

function buildTickIndices(length: number): Set<number> {
  if (length <= 1) {
    return new Set([0]);
  }
  const targetTicks = Math.min(5, length);
  const indices = new Set<number>();
  for (let tick = 0; tick < targetTicks; tick += 1) {
    indices.add(Math.round((tick / Math.max(1, targetTicks - 1)) * (length - 1)));
  }
  return indices;
}

function formatTimelineLabel(bucket: string): string {
  const parsed = new Date(bucket);
  if (Number.isNaN(parsed.getTime())) {
    return bucket;
  }
  const year = parsed.getFullYear();
  const month = String(parsed.getMonth() + 1).padStart(2, "0");
  return `${year}/${month}`;
}

function friendlyTypeLabel(value: string): string {
  if (value === "PL") {
    return "PL";
  }
  if (value === "Raman") {
    return "Raman";
  }
  return value;
}

function friendlyAcquisitionLabel(value: string): string {
  switch (value) {
    case "point_spectrum":
      return "Point";
    case "line_scan":
      return "Line";
    case "area_map":
      return "Area";
    case "series_scan":
      return "Series";
    default:
      return value.replace(/_/g, " ");
  }
}
