import type { NeuralEpochMetric } from "../types";

interface NeuralTrainingChartProps {
  history: NeuralEpochMetric[];
}

const WIDTH = 360;
const HEIGHT = 170;
const PAD = { top: 18, right: 16, bottom: 28, left: 44 };

interface Series {
  label: string;
  color: string;
  value: (metric: NeuralEpochMetric) => number | null;
}

function metricValue(value: number | null | undefined): number | null {
  return value === null || value === undefined || !Number.isFinite(value)
    ? null
    : value;
}

function linePath(
  history: NeuralEpochMetric[],
  value: (metric: NeuralEpochMetric) => number | null,
  minValue: number,
  maxValue: number
) {
  const plotWidth = WIDTH - PAD.left - PAD.right;
  const plotHeight = HEIGHT - PAD.top - PAD.bottom;
  const denominator = Math.max(history.length - 1, 1);
  const valueSpan = Math.max(maxValue - minValue, 1e-9);
  let started = false;
  return history
    .map((metric, index) => {
      const current = value(metric);
      if (current === null) return "";
      const x = PAD.left + (index / denominator) * plotWidth;
      const y =
        PAD.top + (1 - (current - minValue) / valueSpan) * plotHeight;
      const command = started ? "L" : "M";
      started = true;
      return `${command} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .filter(Boolean)
    .join(" ");
}

function LinePanel({
  title,
  history,
  series,
  yLabel
}: {
  title: string;
  history: NeuralEpochMetric[];
  series: Series[];
  yLabel: string;
}) {
  const values = history.flatMap((metric) =>
    series.flatMap((item) => {
      const value = item.value(metric);
      return value === null ? [] : [value];
    })
  );
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const ticks = [0, 0.5, 1].map((fraction) => ({
    y: PAD.top + fraction * (HEIGHT - PAD.top - PAD.bottom),
    value: maxValue - fraction * (maxValue - minValue)
  }));
  const latest = history.at(-1);

  return (
    <div className="training-chart-panel">
      <div className="training-chart-panel-title">
        <strong>{title}</strong>
        <span>{yLabel}</span>
      </div>
      <svg
        className="training-chart"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={title}
      >
        {ticks.map((tick) => (
          <g key={tick.y}>
            <line
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={tick.y}
              y2={tick.y}
            />
            <text x={PAD.left - 7} y={tick.y + 4} textAnchor="end">
              {tick.value.toFixed(3)}
            </text>
          </g>
        ))}
        <line
          className="training-axis"
          x1={PAD.left}
          x2={PAD.left}
          y1={PAD.top}
          y2={HEIGHT - PAD.bottom}
        />
        <line
          className="training-axis"
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={HEIGHT - PAD.bottom}
          y2={HEIGHT - PAD.bottom}
        />
        {series.map((item) => (
          <path
            key={item.label}
            d={linePath(history, item.value, minValue, maxValue)}
            style={{ stroke: item.color }}
          />
        ))}
        <text x={PAD.left} y={HEIGHT - 8}>Epoch 1</text>
        <text x={WIDTH - PAD.right} y={HEIGHT - 8} textAnchor="end">
          Epoch {latest?.epoch}
        </text>
      </svg>
      <div className="training-chart-legend">
        {series.map((item) => (
          <span key={item.label}>
            <i style={{ background: item.color }} />
            {item.label}
          </span>
        ))}
      </div>
    </div>
  );
}

function BarPanel({ latest }: { latest: NeuralEpochMetric }) {
  const bars = [
    {
      label: "Global",
      value: metricValue(latest.validation_rmse_decades),
      color: "#1769ff"
    },
    {
      label: "Weighted",
      value: metricValue(latest.validation_weighted_rmse_decades),
      color: "#079669"
    },
    {
      label: "Low current",
      value: metricValue(latest.validation_low_current_rmse_decades),
      color: "#df7d10"
    },
    {
      label: "Subthreshold",
      value: metricValue(latest.validation_subthreshold_rmse_decades),
      color: "#6d5dfc"
    }
  ];
  const maxValue = Math.max(...bars.map((bar) => bar.value ?? 0), 1e-6);

  return (
    <div className="training-chart-panel training-bar-panel">
      <div className="training-chart-panel-title">
        <strong>Latest regional error</strong>
        <span>RMSE dec</span>
      </div>
      <div className="training-bars">
        {bars.map((bar) => (
          <div key={bar.label}>
            <span>{bar.label}</span>
            <b>
              <i
                style={{
                  width: `${((bar.value ?? 0) / maxValue) * 100}%`,
                  background: bar.color
                }}
              />
            </b>
            <strong>{bar.value === null ? "n/a" : bar.value.toFixed(4)}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export function NeuralTrainingChart({ history }: NeuralTrainingChartProps) {
  if (history.length === 0) {
    return (
      <div className="training-chart-empty">
        Loss curves will appear after the first training epoch.
      </div>
    );
  }

  const latest = history.at(-1);
  const gapSeries: Series[] = [
    {
      label: "Val - train",
      color: "#d34545",
      value: (metric) => metric.validation_loss - metric.train_loss
    }
  ];

  return (
    <div className="training-chart-wrap">
      <LinePanel
        title="Objective loss"
        history={history}
        yLabel="weighted objective"
        series={[
          { label: "Train", color: "#1769ff", value: (metric) => metric.train_loss },
          {
            label: "Validation",
            color: "#079669",
            value: (metric) => metric.validation_loss
          }
        ]}
      />
      <LinePanel
        title="Validation reconstruction"
        history={history}
        yLabel="RMSE dec"
        series={[
          {
            label: "Global",
            color: "#1769ff",
            value: (metric) => metric.validation_rmse_decades
          },
          {
            label: "Weighted",
            color: "#079669",
            value: (metric) => metricValue(metric.validation_weighted_rmse_decades)
          },
          {
            label: "Low current",
            color: "#df7d10",
            value: (metric) => metricValue(metric.validation_low_current_rmse_decades)
          },
          {
            label: "Subthreshold",
            color: "#6d5dfc",
            value: (metric) => metricValue(metric.validation_subthreshold_rmse_decades)
          }
        ]}
      />
      <LinePanel
        title="Generalization gap"
        history={history}
        yLabel="loss delta"
        series={gapSeries}
      />
      {latest ? <BarPanel latest={latest} /> : null}
    </div>
  );
}
