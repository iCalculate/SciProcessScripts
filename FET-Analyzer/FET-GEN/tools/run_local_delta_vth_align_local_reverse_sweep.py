from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

from devicecurvegen.residual import ResidualEngine
from tools.build_hybrid_checkpoint import build_hybrid_checkpoint
from tools.evaluate_postprocess_rechecks import _canonical_metrics
from tools.run_model_experiments import _jump_metrics, _sample_conditions


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments" / "model-experiments-20260627-220813" / "attempt_1_pca16.npz"
FORWARD_GUIDE = ROOT / "experiments" / "local-delta-target-20260630-100253" / "localdelta_dirsplit16-forward-guide.npz"
REVERSE_GUIDE = ROOT / "experiments" / "local-delta-target-20260630-100253" / "localdelta_dirsplit16-reverse-guide.npz"
ACTIVE = ROOT / "models" / "residual-hybrid-threshold-pca.npz"


def _score(item: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        item["jump_p95_decades"],
        item["generated_vth_mae_v"],
        item["canonical_jump_max_decades"],
        item["generated_ss_mae_mv_dec"],
    )


def run() -> Path:
    output_dir = ROOT / "experiments" / f"local-delta-vth-align-local-reverse-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _sample_conditions(limit=72)
    payload: list[dict[str, object]] = []

    active_engine = ResidualEngine(ACTIVE)
    payload.append(
        {
            "name": "active_hybrid_reverse_vth_align050",
            "description": "Current production hybrid checkpoint with reverse-only global post Vth alignment.",
            "checkpoint_path": str(ACTIVE),
            "result": active_engine.info().model_dump(mode="json"),
            "jump_metrics": _jump_metrics(active_engine, rows),
            "canonical_metrics": _canonical_metrics(active_engine),
        }
    )

    for strength, window_scale in itertools.product(
        [0.65, 0.80, 1.00],
        [2.0, 2.5, 3.0],
    ):
        checkpoint = output_dir / (
            f"localdelta_vthalign_local_reverse_"
            f"s{int(round(strength * 100)):03d}_w{int(round(window_scale * 10)):02d}.npz"
        )
        build_hybrid_checkpoint(
            base_path=BASE,
            guide_path=FORWARD_GUIDE,
            reverse_guide_path=REVERSE_GUIDE,
            guide_as_local_delta=True,
            output_path=checkpoint,
            base_scale_multiplier=0.25,
            local_blend=1.0,
            global_blend=0.04,
            window_scale=3.0,
            min_window_v=0.22,
            guide_delta_clip_decades=0.08,
            post_vth_align_strength=strength,
            post_vth_align_reverse_only=True,
            post_vth_align_local_window_scale=window_scale,
            post_vth_align_local_min_window_v=0.18,
        )
        engine = ResidualEngine(checkpoint)
        payload.append(
            {
                "name": f"reverse_local_vth_align_{strength:.2f}_w{window_scale:.1f}",
                "description": "Direction-split local delta guide with reverse-only local-window post Vth alignment.",
                "checkpoint_path": str(checkpoint),
                "result": engine.info().model_dump(mode="json"),
                "jump_metrics": _jump_metrics(engine, rows),
                "canonical_metrics": _canonical_metrics(engine),
            }
        )

    ranked = payload[:1] + sorted(
        payload[1:],
        key=lambda item: _score({**item["jump_metrics"], **item["canonical_metrics"]}),
    )
    (output_dir / "summary.json").write_text(json.dumps(ranked, indent=2), encoding="utf-8")

    lines = [
        "# Local delta reverse-only local-window Vth-align sweep",
        "",
        "| Model | Jump P95 | Canonical max | Canonical reverse max | Gen. Vth | Gen. SS |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in ranked:
        jump = item["jump_metrics"]
        canonical = item["canonical_metrics"]
        lines.append(
            f"| {item['name']} | "
            f"{jump['jump_p95_decades']:.4f} | "
            f"{canonical['canonical_jump_max_decades']:.4f} | "
            f"{canonical['canonical_reverse_jump_max_decades']:.4f} | "
            f"{jump['generated_vth_mae_v']:.3f} | "
            f"{jump['generated_ss_mae_mv_dec']:.1f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    print(run())
