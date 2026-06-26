import { CheckCircle2, ChevronDown, ChevronUp, Table2 } from "lucide-react";
import { fixed, scientific } from "../format";
import type { GeneratedCandidate } from "../types";

interface CandidateTableProps {
  candidates: GeneratedCandidate[];
  selectedId: number;
  expanded: boolean;
  onSelect: (id: number) => void;
  onToggle: () => void;
}

export function CandidateTable({
  candidates,
  selectedId,
  expanded,
  onSelect,
  onToggle
}: CandidateTableProps) {
  const selected = candidates.find((candidate) => candidate.candidate_id === selectedId);
  const summary = selected
    ? `#${selected.candidate_id} | Ion ${scientific(selected.features.ion)} | SS ${fixed(selected.features.ss_mv_dec, 1)}`
    : `${candidates.length} candidates`;

  return (
    <section
      className={`candidate-data-panel${expanded ? " expanded" : " collapsed"}`}
      aria-label="Candidate data"
    >
      <div className="candidate-data-header">
        <div className="candidate-data-title">
          <span className="candidate-data-icon" aria-hidden="true">
            <Table2 size={15} />
          </span>
          <div>
            <h2>Candidate data</h2>
            <p>{summary}</p>
          </div>
        </div>
        <button
          type="button"
          className="candidate-data-toggle"
          aria-expanded={expanded}
          onClick={onToggle}
        >
          <span>{expanded ? "Hide table" : "Show table"}</span>
          {expanded ? <ChevronDown size={15} /> : <ChevronUp size={15} />}
        </button>
      </div>
      {expanded ? (
        <div className="candidate-table-scroll">
          <table className="candidate-table">
            <thead>
              <tr>
                <th>Candidate</th>
                <th>Seed</th>
                <th>Vth (V)</th>
                <th>Ion (A)</th>
                <th>Ioff (A)</th>
                <th>Ion / Ioff</th>
                <th>SS (mV/dec)</th>
                <th>Hyst. (V)</th>
                <th>Quality</th>
              </tr>
            </thead>
            <tbody>
              {candidates.map((candidate) => (
                <tr
                  key={candidate.candidate_id}
                  className={candidate.candidate_id === selectedId ? "selected" : ""}
                  aria-selected={candidate.candidate_id === selectedId}
                  tabIndex={0}
                  onClick={() => onSelect(candidate.candidate_id)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") {
                      event.preventDefault();
                      onSelect(candidate.candidate_id);
                    }
                  }}
                >
                  <td>
                    <span className="candidate-index">{candidate.candidate_id}</span>
                  </td>
                  <td>{candidate.seed}</td>
                  <td>{fixed(candidate.features.vth, 2)}</td>
                  <td>{scientific(candidate.features.ion)}</td>
                  <td>{scientific(candidate.features.ioff)}</td>
                  <td>{scientific(candidate.features.ion_ioff_ratio)}</td>
                  <td>{fixed(candidate.features.ss_mv_dec, 1)}</td>
                  <td>{fixed(candidate.features.hysteresis_v, 2)}</td>
                  <td>
                    <span className="quality-cell">
                      {candidate.quality_score.toFixed(2)}
                      <CheckCircle2
                        size={13}
                        className={candidate.quality_score >= 0.75 ? "good" : "warn"}
                      />
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
