import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "plotly.js-basic-dist-min";
import type { Data, Layout, Shape } from "plotly.js";
import { useRef, type PointerEvent as ReactPointerEvent } from "react";
import type {
  GeneratedCandidate,
  GenerationCondition,
  GenerationResponse
} from "../types";

const Plot = createPlotlyComponent(Plotly);

interface CurvePlotProps {
  response: GenerationResponse;
  selected: GeneratedCandidate;
  condition: GenerationCondition;
  highlightAll: boolean;
  onConstraintChange: (patch: Partial<GenerationCondition>) => void;
}

export function CurvePlot({
  response,
  selected,
  condition,
  highlightAll,
  onConstraintChange
}: CurvePlotProps) {
  const transferOverlayRef = useRef<HTMLDivElement>(null);
  const outputColors = ["#9fb8df", "#6f99d4", "#3f79c8", "#1769ff", "#0d4da8"];
  const transferDomain: [number, number] = [0, 0.54];
  const outputDomain: [number, number] = [0.64, 1];
  const candidateTraces: Data[] = response.candidates
    .filter((candidate) => candidate.candidate_id !== selected.candidate_id)
    .flatMap((candidate) => [
      {
        x: candidate.voltage,
        y: candidate.forward_current,
        type: "scatter",
        mode: "lines",
        xaxis: "x",
        yaxis: "y",
        hoverinfo: "skip",
        line: {
          color: highlightAll ? "rgba(49, 105, 194, 0.72)" : "rgba(109, 139, 181, 0.18)",
          width: highlightAll ? 1.6 : 1
        },
        showlegend: false
      } as Data,
      {
        x: candidate.voltage,
        y: candidate.reverse_current,
        type: "scatter",
        mode: "lines",
        xaxis: "x",
        yaxis: "y",
        hoverinfo: "skip",
        line: {
          color: highlightAll ? "rgba(49, 105, 194, 0.52)" : "rgba(109, 139, 181, 0.12)",
          width: highlightAll ? 1.35 : 1,
          dash: highlightAll ? "dot" : "solid"
        },
        showlegend: false
      } as Data,
      {
        x: candidate.voltage,
        y: candidate.gate_forward_current,
        type: "scatter",
        mode: "lines",
        xaxis: "x",
        yaxis: "y",
        hoverinfo: "skip",
        line: {
          color: highlightAll ? "rgba(217, 95, 2, 0.42)" : "rgba(217, 104, 54, 0.12)",
          width: highlightAll ? 1.2 : 1
        },
        showlegend: false
      } as Data
    ]);

  const hysteresisLeft = condition.target_vth - condition.hysteresis_v / 2;
  const hysteresisRight = condition.target_vth + condition.hysteresis_v / 2;
  let minimumCurrent = condition.target_ioff;
  let maximumCurrent = condition.target_ion;
  for (const candidate of response.candidates) {
    for (const series of [
      candidate.forward_current,
      candidate.reverse_current,
      candidate.gate_forward_current,
      candidate.gate_reverse_current
    ]) {
      for (const current of series) {
        if (!Number.isFinite(current) || current <= 0) continue;
        minimumCurrent = Math.min(minimumCurrent, current);
        maximumCurrent = Math.max(maximumCurrent, current);
      }
    }
  }
  const logMinimum = Math.log10(minimumCurrent) - 0.35;
  const logMaximum = Math.log10(maximumCurrent) + 0.35;
  const selectedIdsMax = Math.max(
    ...selected.forward_current,
    ...selected.reverse_current,
    condition.target_ion
  );
  const outputTraces: Data[] = selected.output_curves.map((curve, index) => ({
    x: selected.output_drain_voltage,
    y: curve.current,
    type: "scatter",
    mode: "lines",
    xaxis: "x2",
    yaxis: "y2",
    name: `Vg ${curve.gate_voltage.toFixed(1)} V`,
    showlegend: false,
    line: {
      color: outputColors[index] ?? outputColors.at(-1),
      width: 2
    }
  }));

  const traces: Data[] = [
    ...candidateTraces,
    {
      x: selected.voltage,
      y: selected.physics_forward_current,
      type: "scatter",
      mode: "lines",
      xaxis: "x",
      yaxis: "y",
      name: "Physics baseline",
      showlegend: false,
      line: { color: "rgba(18, 36, 62, .38)", width: 1.2 }
    },
    {
      x: selected.voltage,
      y: selected.forward_current,
      type: "scatter",
      mode: "lines",
      xaxis: "x",
      yaxis: "y",
      name: "Ids forward",
      showlegend: false,
      line: { color: "#1769ff", width: 2.4 }
    },
    {
      x: selected.voltage,
      y: selected.reverse_current,
      type: "scatter",
      mode: "lines",
      xaxis: "x",
      yaxis: "y",
      name: "Ids reverse",
      showlegend: false,
      line: { color: "#1769ff", width: 2.1, dash: "dot" }
    },
    {
      x: selected.voltage,
      y: selected.gate_forward_current,
      type: "scatter",
      mode: "lines",
      xaxis: "x",
      yaxis: "y",
      name: "Ig forward",
      showlegend: false,
      line: { color: "#d95f02", width: 1.8 }
    },
    {
      x: selected.voltage,
      y: selected.gate_reverse_current,
      type: "scatter",
      mode: "lines",
      xaxis: "x",
      yaxis: "y",
      name: "Ig reverse",
      showlegend: false,
      line: { color: "#d95f02", width: 1.6, dash: "dot" }
    },
    {
      x: selected.voltage,
      y: selected.forward_current,
      type: "scatter",
      mode: "lines",
      xaxis: "x",
      yaxis: "y3",
      name: "Ids forward linear",
      showlegend: false,
      line: { color: "#1769ff", width: 2.1 }
    },
    {
      x: selected.voltage,
      y: selected.reverse_current,
      type: "scatter",
      mode: "lines",
      xaxis: "x",
      yaxis: "y3",
      name: "Ids reverse linear",
      showlegend: false,
      line: { color: "#1769ff", width: 1.9, dash: "dot" }
    },
    ...outputTraces
  ];

  const shapes: Partial<Shape>[] = [
    {
      type: "line",
      x0: condition.target_vth,
      x1: condition.target_vth,
      y0: 0,
      y1: 1,
      xref: "x",
      yref: "paper",
      line: { color: "#08917b", width: 1.4, dash: "dash" }
    },
    {
      type: "line",
      x0: condition.voltage_min,
      x1: condition.voltage_max,
      y0: condition.target_ion,
      y1: condition.target_ion,
      line: { color: "#08917b", width: 1.2, dash: "dash" }
    },
    {
      type: "line",
      x0: condition.voltage_min,
      x1: condition.voltage_max,
      y0: condition.target_ioff,
      y1: condition.target_ioff,
      line: { color: "#e78a16", width: 1.2, dash: "dot" }
    },
    {
      type: "rect",
      x0: hysteresisLeft,
      x1: hysteresisRight,
      y0: 0,
      y1: 1,
      xref: "x",
      yref: "paper",
      fillcolor: "rgba(23, 105, 255, 0.10)",
      line: { color: "rgba(23, 105, 255, 0.42)", width: 1 }
    }
  ];

  const layout: Partial<Layout> = {
    autosize: true,
    margin: { l: 58, r: 82, t: 42, b: 52 },
    paper_bgcolor: "#ffffff",
    plot_bgcolor: "#fbfdff",
    font: { family: "Segoe UI, system-ui, sans-serif", color: "#263a55", size: 11 },
    showlegend: false,
    xaxis: {
      title: { text: "Gate voltage Vg (V)", standoff: 14 },
      range: [condition.voltage_min, condition.voltage_max],
      domain: transferDomain,
      gridcolor: "#e7edf5",
      zerolinecolor: "#cdd7e5",
      linecolor: "#9cabc0",
      mirror: false,
      ticks: "outside"
    },
    yaxis: {
      title: { text: "Current |Ids|, |Ig| (A)", standoff: 12 },
      type: "log",
      domain: [0, 1],
      range: [logMinimum, logMaximum],
      tickformat: ".0e",
      exponentformat: "e",
      showexponent: "all",
      gridcolor: "#e7edf5",
      linecolor: "#9cabc0",
      ticks: "outside"
    },
    yaxis3: {
      title: { text: "Linear Ids (A)", standoff: 7 },
      anchor: "x",
      overlaying: "y",
      side: "right",
      range: [0, selectedIdsMax * 1.08],
      tickformat: ".1e",
      showgrid: false,
      zeroline: false,
      linecolor: "#9cabc0",
      ticks: "outside"
    },
    xaxis2: {
      title: { text: "Drain voltage Vd (V)", standoff: 14 },
      domain: outputDomain,
      anchor: "y2",
      gridcolor: "#e7edf5",
      zerolinecolor: "#9cabc0",
      linecolor: "#9cabc0",
      ticks: "outside"
    },
    yaxis2: {
      title: { text: "Drain current |Ids| (A)", standoff: 10 },
      anchor: "x2",
      side: "right",
      rangemode: "tozero",
      tickformat: ".1e",
      gridcolor: "#e7edf5",
      zerolinecolor: "#9cabc0",
      linecolor: "#9cabc0",
      ticks: "outside"
    },
    shapes,
    annotations: [
      {
        x: 0.26,
        y: 1.08,
        xref: "paper",
        yref: "paper",
        text: "Transfer characteristics",
        showarrow: false,
        font: { color: "#263a55", size: 12 }
      },
      {
        x: 0.85,
        y: 1.08,
        xref: "paper",
        yref: "paper",
        text: "Output characteristics",
        showarrow: false,
        font: { color: "#263a55", size: 12 }
      }
    ],
    hovermode: false,
    hoverdistance: -1,
    uirevision: `${condition.voltage_min}-${condition.voltage_max}`
  };

  function voltagePercent(voltage: number): number {
    return Math.min(100, Math.max(0,
      (100 * (voltage - condition.voltage_min)) /
      (condition.voltage_max - condition.voltage_min)
    ));
  }

  function currentPercent(current: number): number {
    return Math.min(
      100,
      Math.max(
        0,
        (100 * (logMaximum - Math.log10(current))) / (logMaximum - logMinimum)
      )
    );
  }

  function beginVoltageDrag(
    event: ReactPointerEvent,
    onVoltage: (voltage: number) => void
  ) {
    event.preventDefault();
    const overlay = transferOverlayRef.current;
    if (!overlay) return;
    const move = (pointerEvent: PointerEvent) => {
      const rect = overlay.getBoundingClientRect();
      const ratio = Math.min(1, Math.max(0, (pointerEvent.clientX - rect.left) / rect.width));
      onVoltage(
        condition.voltage_min +
          ratio * (condition.voltage_max - condition.voltage_min)
      );
    };
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, { once: true });
  }

  function beginCurrentDrag(
    event: ReactPointerEvent,
    onCurrent: (current: number) => void
  ) {
    event.preventDefault();
    const overlay = transferOverlayRef.current;
    if (!overlay) return;
    const move = (pointerEvent: PointerEvent) => {
      const rect = overlay.getBoundingClientRect();
      const ratio = Math.min(1, Math.max(0, (pointerEvent.clientY - rect.top) / rect.height));
      onCurrent(10 ** (logMaximum - ratio * (logMaximum - logMinimum)));
    };
    const stop = () => {
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop, { once: true });
  }

  return (
    <div
      className="plot-shell"
      title="Drag Vth, Ion, Ioff, or the hysteresis region edges to edit constraints"
    >
      <Plot
        data={traces}
        layout={layout}
        config={{
          responsive: true,
          displaylogo: false,
          displayModeBar: false,
          editable: false,
          edits: {
            annotationPosition: false,
            annotationTail: false,
            annotationText: false,
            axisTitleText: false,
            colorbarPosition: false,
            colorbarTitleText: false,
            legendPosition: false,
            shapePosition: false,
            titleText: false
          },
          modeBarButtonsToRemove: ["lasso2d", "select2d"]
        }}
        useResizeHandler
        className="curve-plot"
      />
      <div className="plot-legends-layer">
        <div className="plot-legend transfer-plot-legend">
          <span><i className="legend-line ids-forward" />Ids forward</span>
          <span><i className="legend-line ids-reverse" />Ids reverse</span>
          <span><i className="legend-line ig-current" />Ig</span>
        </div>
        <div className="plot-legend output-plot-legend">
          {selected.output_curves.map((curve, index) => (
            <span key={curve.gate_voltage}>
              <i
                className="legend-line"
                style={{ backgroundColor: outputColors[index] ?? outputColors.at(-1) }}
              />
              Vg {curve.gate_voltage.toFixed(1)} V
            </span>
          ))}
        </div>
      </div>
      <div className="constraint-overlay">
        <div ref={transferOverlayRef} className="transfer-constraint-overlay">
          <div
            className="constraint-line vertical vth-control"
            style={{ left: `${voltagePercent(condition.target_vth)}%` }}
            onPointerDown={(event) =>
              beginVoltageDrag(event, (target_vth) =>
                onConstraintChange({ target_vth })
              )
            }
          >
            <span>Vth {condition.target_vth.toFixed(2)} V</span>
          </div>
          <div
            className="constraint-line horizontal ion-control"
            style={{ top: `${currentPercent(condition.target_ion)}%` }}
            onPointerDown={(event) =>
              beginCurrentDrag(event, (target_ion) => {
                if (target_ion > condition.target_ioff) {
                  onConstraintChange({ target_ion });
                }
              })
            }
          >
            <span>Ion {condition.target_ion.toExponential(1)} A</span>
          </div>
          <div
            className="constraint-line horizontal ioff-control"
            style={{ top: `${currentPercent(condition.target_ioff)}%` }}
            onPointerDown={(event) =>
              beginCurrentDrag(event, (target_ioff) => {
                if (target_ioff > 0 && target_ioff < condition.target_ion) {
                  onConstraintChange({ target_ioff });
                }
              })
            }
          >
            <span>Ioff {condition.target_ioff.toExponential(1)} A</span>
          </div>
          <div
            className="hysteresis-edge left"
            style={{ left: `${voltagePercent(hysteresisLeft)}%` }}
            onPointerDown={(event) =>
              beginVoltageDrag(event, (nextLeft) => {
                const clampedLeft = Math.min(nextLeft, hysteresisRight);
                onConstraintChange({
                  target_vth: 0.5 * (clampedLeft + hysteresisRight),
                  hysteresis_v: hysteresisRight - clampedLeft
                });
              })
            }
          />
          <div
            className="hysteresis-edge right"
            style={{ left: `${voltagePercent(hysteresisRight)}%` }}
            onPointerDown={(event) =>
              beginVoltageDrag(event, (nextRight) => {
                const clampedRight = Math.max(nextRight, hysteresisLeft);
                onConstraintChange({
                  target_vth: 0.5 * (hysteresisLeft + clampedRight),
                  hysteresis_v: clampedRight - hysteresisLeft
                });
              })
            }
          />
        </div>
      </div>
    </div>
  );
}
