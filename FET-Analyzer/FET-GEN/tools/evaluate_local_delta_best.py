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
    output_dir = ROOT / "experiments" / f"local-delta-best-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_checkpoint = output_dir / "localdelta_dirsplit16_basescale025.npz"
    build_hybrid_checkpoint(
        base_path=BASE,
        guide_path=FORWARD_GUIDE,
        reverse_guide_path=REVERSE_GUIDE,
        guide_as_local_delta=True,
        output_path=candidate_checkpoint,
        base_scale_multiplier=0.25,
        local_blend=1.0,
        global_blend=0.04,
        window_scale=3.0,
        min_window_v=0.22,
        guide_delta_clip_decades=0.08,
    )
    rows = _sample_conditions(limit=72)
    payload = []
    for name, description, checkpoint in [
        (
            "active_hybrid_onstate_retuned",
            "Current production hybrid checkpoint with the on-state onset limiter.",
            ACTIVE,
        ),
        (
            "localdelta_dirsplit16_basescale025",
            "Direction-split local delta guide with reduced base latent scale, optimized for 100% AI jump suppression.",
            candidate_checkpoint,
        ),
    ]:
        engine = ResidualEngine(checkpoint)
        payload.append(
            {
                "name": name,
                "description": description,
                "checkpoint_path": str(checkpoint),
                "result": engine.info().model_dump(mode="json"),
                "jump_metrics": _jump_metrics(engine, rows),
                "canonical_metrics": _canonical_metrics(engine),
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Local delta best candidate",
        "",
        "| Model | Jump P95 | Canonical max | Canonical reverse max | Gen. Vth | Gen. SS |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in payload:
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
