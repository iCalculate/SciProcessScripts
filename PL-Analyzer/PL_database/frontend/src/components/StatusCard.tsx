interface StatusCardProps {
  label: string;
  value: string;
  detail?: string;
  delta?: number;
}

export function StatusCard(props: StatusCardProps) {
  return (
    <article className="status-card">
      <p className="status-label">{props.label}</p>
      <div className="status-value-row">
        <strong className="status-value">{props.value}</strong>
        {props.delta && props.delta > 0 ? <span className="status-delta">+{props.delta}</span> : null}
      </div>
      {props.detail ? <p className="status-detail">{props.detail}</p> : null}
    </article>
  );
}
