from __future__ import annotations

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


def run() -> Path:
    output_dir = ROOT / "experiments" / f"local-delta-affine-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _sample_conditions(limit=72)
    results = []
    active_engine = ResidualEngine(ACTIVE)
    results.append(
        {
            "name": "active_hybrid_onstate_retuned",
            "jump_metrics": _jump_metrics(active_engine, rows),
            "canonical_metrics": _canonical_metrics(active_engine),
        }
    )
    for affine_strength in [0.0, 0.25, 0.5, 0.75, 1.0]:
        checkpoint = output_dir / f"affine_{int(round(affine_strength * 100)):03d}.npz"
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
            guide_delta_preserve_affine_strength=affine_strength,
        )
        engine = ResidualEngine(checkpoint)
        results.append(
            {
                "name": f"affine_{affine_strength:.2f}",
                "affine_strength": affine_strength,
                "checkpoint_path": str(checkpoint),
                "jump_metrics": _jump_metrics(engine, rows),
                "canonical_metrics": _canonical_metrics(engine),
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    lines = [
        "# Local delta affine sweep",
        "",
        "| Model | Jump P95 | Canonical max | Gen. Vth | Gen. SS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in results:
        jump = item["jump_metrics"]
        canonical = item["canonical_metrics"]
        lines.append(
            f"| {item['name']} | {jump['jump_p95_decades']:.4f} | {canonical['canonical_jump_max_decades']:.4f} | {jump['generated_vth_mae_v']:.3f} | {jump['generated_ss_mae_mv_dec']:.1f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    print(run())
