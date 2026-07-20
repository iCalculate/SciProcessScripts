from __future__ import annotations

import argparse
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


def _priority_key(candidate: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    return (
        abs(candidate["base_scale"] - 0.25),
        abs(candidate["global_blend"] - 0.04),
        abs(candidate["delta_clip"] - 0.05),
        abs(candidate["post_vth_align_strength"] - 1.15),
        abs(candidate["post_vth_local_window_scale"] - 2.5),
        abs(candidate["post_vth_local_min_window_v"] - 0.18),
    )


def _candidate_space() -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for base_scale in [0.247, 0.250, 0.253]:
        for global_blend in [0.03, 0.04]:
            for delta_clip in [0.045, 0.05]:
                for post_vth_strength in [1.13, 1.15, 1.17]:
                    for post_local_window in [2.3, 2.5, 2.7]:
                        for post_local_min_window in [0.16, 0.18]:
                            candidates.append(
                                {
                                    "base_scale": base_scale,
                                    "global_blend": global_blend,
                                    "delta_clip": delta_clip,
                                    "post_vth_align_strength": post_vth_strength,
                                    "post_vth_local_window_scale": post_local_window,
                                    "post_vth_local_min_window_v": post_local_min_window,
                                }
                            )
    return sorted(candidates, key=_priority_key)


def run(*, max_trials: int | None = 48, sample_limit: int = 48) -> Path:
    if sample_limit <= 0:
        raise ValueError("sample_limit must be positive")

    output_dir = ROOT / "experiments" / f"aligned-local-affine-delta-fine-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _sample_conditions(limit=sample_limit)
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

    candidates = _candidate_space()
    if max_trials is not None and max_trials > 0:
        candidates = candidates[:max_trials]

    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        base_scale = candidate["base_scale"]
        global_blend = candidate["global_blend"]
        delta_clip = candidate["delta_clip"]
        post_vth_strength = candidate["post_vth_align_strength"]
        post_local_window = candidate["post_vth_local_window_scale"]
        post_local_min_window = candidate["post_vth_local_min_window_v"]
        name = (
            f"b{int(round(base_scale * 1000)):03d}"
            f"_g{int(round(global_blend * 1000)):03d}"
            f"_c{int(round(delta_clip * 1000)):03d}"
            f"_s{int(round(post_vth_strength * 1000)):04d}"
            f"_w{int(round(post_local_window * 100)):03d}"
            f"_m{int(round(post_local_min_window * 100)):03d}"
        )
        print(f"[{index}/{total}] evaluating {name}")
        checkpoint = output_dir / f"{name}.npz"
        build_hybrid_checkpoint(
            base_path=BASE,
            guide_path=FORWARD_GUIDE,
            reverse_guide_path=REVERSE_GUIDE,
            guide_as_local_delta=True,
            output_path=checkpoint,
            base_scale_multiplier=base_scale,
            local_blend=1.0,
            global_blend=global_blend,
            window_scale=3.0,
            min_window_v=0.22,
            guide_delta_clip_decades=delta_clip,
            post_vth_align_strength=post_vth_strength,
            post_vth_align_reverse_only=True,
            post_vth_align_local_window_scale=post_local_window,
            post_vth_align_local_min_window_v=post_local_min_window,
        )
        engine = ResidualEngine(checkpoint)
        results.append(
            {
                "name": name,
                "base_scale": base_scale,
                "global_blend": global_blend,
                "delta_clip": delta_clip,
                "post_vth_align_strength": post_vth_strength,
                "post_vth_local_window_scale": post_local_window,
                "post_vth_local_min_window_v": post_local_min_window,
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
        "# Affine-delta fine sweep",
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-trials",
        type=int,
        default=48,
        help="Maximum candidate checkpoints to evaluate. Use 0 or a negative value for the full grid.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=48,
        help="Number of sampled condition rows to use during ranking.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(
        run(
            max_trials=None if args.max_trials <= 0 else args.max_trials,
            sample_limit=args.sample_limit,
        )
    )
