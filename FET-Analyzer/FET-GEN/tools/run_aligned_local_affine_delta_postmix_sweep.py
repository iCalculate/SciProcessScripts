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
ACTIVE = ROOT / "models" / "residual-hybrid-threshold-pca.npz"
GUIDE_ROOT = ROOT / "experiments" / "aligned-local-affine-delta-training-sweep-20260630-142153"
FORWARD_GUIDE = GUIDE_ROOT / "aligned_affine_delta_dirsplit24_slopefocus-forward-guide.npz"
REVERSE_GUIDE = GUIDE_ROOT / "aligned_affine_delta_dirsplit24_slopefocus-reverse-guide.npz"


def run() -> Path:
    output_dir = ROOT / "experiments" / f"aligned-local-affine-delta-postmix-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _sample_conditions(limit=72)
    results: list[dict[str, object]] = []

    active_engine = ResidualEngine(ACTIVE)
    results.append(
        {
            "name": "active_hybrid_reverse_local_vth_align115_w25",
            "checkpoint_path": str(ACTIVE),
            "jump_metrics": _jump_metrics(active_engine, rows),
            "canonical_metrics": _canonical_metrics(active_engine),
        }
    )

    for anchor_strength in [0.0, 0.15, 0.30, 0.45]:
        for affine_strength in [0.0, 0.15, 0.30, 0.45]:
            name = (
                f"a{int(round(anchor_strength * 100)):03d}"
                f"_f{int(round(affine_strength * 100)):03d}"
            )
            checkpoint = output_dir / f"{name}.npz"
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
                guide_delta_clip_decades=0.05,
                guide_delta_anchor_strength=anchor_strength,
                guide_delta_preserve_affine_strength=affine_strength,
                post_vth_align_strength=1.15,
                post_vth_align_reverse_only=True,
                post_vth_align_local_window_scale=2.5,
                post_vth_align_local_min_window_v=0.18,
            )
            engine = ResidualEngine(checkpoint)
            results.append(
                {
                    "name": name,
                    "anchor_strength": anchor_strength,
                    "affine_strength": affine_strength,
                    "checkpoint_path": str(checkpoint),
                    "jump_metrics": _jump_metrics(engine, rows),
                    "canonical_metrics": _canonical_metrics(engine),
                }
            )

    (output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    ranked = sorted(
        results,
        key=lambda item: (
            float(item["jump_metrics"]["jump_p95_decades"]),
            float(item["canonical_metrics"]["canonical_jump_max_decades"]),
            float(item["jump_metrics"]["generated_vth_mae_v"]),
            float(item["jump_metrics"]["generated_ss_mae_mv_dec"]),
        ),
    )
    lines = [
        "# Affine-delta postmix sweep",
        "",
        "| Rank | Model | Jump P95 | Canonical max | Canonical reverse max | Gen. Vth | Gen. SS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(ranked, start=1):
        jump = item["jump_metrics"]
        canonical = item["canonical_metrics"]
        lines.append(
            f"| {rank} | {item['name']} | "
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
