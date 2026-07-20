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
GUIDE = ROOT / "experiments" / "local-delta-target-20260630-100253" / "localdelta_dirsplit16-forward-guide.npz"
REVERSE_GUIDE = ROOT / "experiments" / "local-delta-target-20260630-100253" / "localdelta_dirsplit16-reverse-guide.npz"
ACTIVE = ROOT / "models" / "residual-hybrid-threshold-pca.npz"


def _score(item: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        item["jump_p95_decades"],
        item["generated_vth_mae_v"],
        item["canonical_jump_max_decades"],
        item["generated_ss_mae_mv_dec"],
    )


def _evaluate(
    output_dir: Path,
    *,
    base_scale_multiplier: float,
    local_blend: float,
    global_blend: float,
    delta_clip: float,
    sampled_conditions: list[dict],
) -> dict[str, float]:
    checkpoint = output_dir / (
        f"refine_bsm{int(round(base_scale_multiplier * 100)):03d}"
        f"_lb{int(round(local_blend * 100)):03d}"
        f"_gb{int(round(global_blend * 100)):02d}"
        f"_dc{int(round(delta_clip * 100)):02d}.npz"
    )
    build_hybrid_checkpoint(
        base_path=BASE,
        guide_path=GUIDE,
        reverse_guide_path=REVERSE_GUIDE,
        guide_as_local_delta=True,
        output_path=checkpoint,
        base_scale_multiplier=base_scale_multiplier,
        local_blend=local_blend,
        global_blend=global_blend,
        window_scale=3.0,
        min_window_v=0.22,
        guide_delta_clip_decades=delta_clip,
    )
    engine = ResidualEngine(checkpoint)
    return {
        "base_scale_multiplier": base_scale_multiplier,
        "local_blend": local_blend,
        "global_blend": global_blend,
        "guide_delta_clip_decades": delta_clip,
        **_jump_metrics(engine, sampled_conditions),
        **_canonical_metrics(engine),
    }


def run() -> Path:
    output_dir = ROOT / "experiments" / f"local-delta-refine-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_conditions = _sample_conditions(limit=24)
    full_conditions = _sample_conditions(limit=72)
    coarse = [
        _evaluate(
            output_dir,
            base_scale_multiplier=base_scale_multiplier,
            local_blend=local_blend,
            global_blend=global_blend,
            delta_clip=delta_clip,
            sampled_conditions=coarse_conditions,
        )
        for base_scale_multiplier, local_blend, global_blend, delta_clip in itertools.product(
            [0.20, 0.25, 0.30, 0.35, 0.40],
            [0.82, 0.94, 1.00],
            [0.00, 0.02, 0.04],
            [0.08, 0.10, 0.12, 0.15],
        )
    ]
    coarse.sort(key=_score)
    shortlist = coarse[:14]
    full = [
        _evaluate(
            output_dir,
            base_scale_multiplier=float(item["base_scale_multiplier"]),
            local_blend=float(item["local_blend"]),
            global_blend=float(item["global_blend"]),
            delta_clip=float(item["guide_delta_clip_decades"]),
            sampled_conditions=full_conditions,
        )
        for item in shortlist
    ]
    full.sort(key=_score)
    active_engine = ResidualEngine(ACTIVE)
    active = {
        **_jump_metrics(active_engine, full_conditions),
        **_canonical_metrics(active_engine),
    }
    payload = {"active": active, "coarse_top": coarse[:20], "full_top": full}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Local delta refine sweep",
        "",
        f"| Baseline | active hybrid | Jump P95 {active['jump_p95_decades']:.4f} | Canonical max {active['canonical_jump_max_decades']:.4f} | Vth {active['generated_vth_mae_v']:.3f} | SS {active['generated_ss_mae_mv_dec']:.1f} |",
        "",
        "| Rank | base scale | local blend | global blend | delta clip | Jump P95 | Canonical max | Gen. Vth | Gen. SS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(full, start=1):
        lines.append(
            f"| {index} | "
            f"{item['base_scale_multiplier']:.2f} | "
            f"{item['local_blend']:.2f} | "
            f"{item['global_blend']:.2f} | "
            f"{item['guide_delta_clip_decades']:.2f} | "
            f"{item['jump_p95_decades']:.4f} | "
            f"{item['canonical_jump_max_decades']:.4f} | "
            f"{item['generated_vth_mae_v']:.3f} | "
            f"{item['generated_ss_mae_mv_dec']:.1f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    print(run())
