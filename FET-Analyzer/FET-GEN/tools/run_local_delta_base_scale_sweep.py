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
GUIDE_ROOT = ROOT / "experiments" / "local-delta-target-20260630-100253"
ACTIVE = ROOT / "models" / "residual-hybrid-threshold-pca.npz"

GUIDE_SETS = [
    {
        "name": "localdelta_full16",
        "guide": GUIDE_ROOT / "localdelta_full16-guide.npz",
        "reverse_guide": None,
    },
    {
        "name": "localdelta_full20",
        "guide": GUIDE_ROOT / "localdelta_full20-guide.npz",
        "reverse_guide": None,
    },
    {
        "name": "localdelta_dirsplit16",
        "guide": GUIDE_ROOT / "localdelta_dirsplit16-forward-guide.npz",
        "reverse_guide": GUIDE_ROOT / "localdelta_dirsplit16-reverse-guide.npz",
    },
]


def _score(item: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        item["jump_p95_decades"],
        item["canonical_jump_max_decades"],
        item["generated_vth_mae_v"],
        item["generated_ss_mae_mv_dec"],
    )


def _evaluate(
    output_dir: Path,
    guide_set: dict[str, object],
    *,
    base_scale_multiplier: float,
    local_blend: float,
    sampled_conditions: list[dict],
) -> dict[str, float | str]:
    checkpoint = output_dir / (
        f"{guide_set['name']}"
        f"_bsm{int(round(base_scale_multiplier * 100)):03d}"
        f"_lb{int(round(local_blend * 100)):03d}.npz"
    )
    build_hybrid_checkpoint(
        base_path=BASE,
        guide_path=Path(guide_set["guide"]),
        reverse_guide_path=(
            Path(guide_set["reverse_guide"])
            if guide_set["reverse_guide"] is not None
            else None
        ),
        guide_as_local_delta=True,
        output_path=checkpoint,
        base_scale_multiplier=base_scale_multiplier,
        local_blend=local_blend,
        global_blend=0.0,
        window_scale=3.0,
        min_window_v=0.22,
        guide_delta_clip_decades=0.12,
    )
    engine = ResidualEngine(checkpoint)
    return {
        "guide_family": str(guide_set["name"]),
        "base_scale_multiplier": base_scale_multiplier,
        "local_blend": local_blend,
        **_jump_metrics(engine, sampled_conditions),
        **_canonical_metrics(engine),
    }


def run() -> Path:
    output_dir = ROOT / "experiments" / f"local-delta-base-scale-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_conditions = _sample_conditions(limit=24)
    full_conditions = _sample_conditions(limit=72)
    coarse = []
    for guide_set in GUIDE_SETS:
        for base_scale_multiplier, local_blend in itertools.product(
            [0.15, 0.25, 0.35, 0.50, 0.75, 1.00],
            [0.82, 0.94, 1.00],
        ):
            coarse.append(
                _evaluate(
                    output_dir,
                    guide_set,
                    base_scale_multiplier=base_scale_multiplier,
                    local_blend=local_blend,
                    sampled_conditions=coarse_conditions,
                )
            )
    coarse.sort(key=_score)
    shortlist = coarse[:12]
    full = [
        _evaluate(
            output_dir,
            next(item for item in GUIDE_SETS if item["name"] == candidate["guide_family"]),
            base_scale_multiplier=float(candidate["base_scale_multiplier"]),
            local_blend=float(candidate["local_blend"]),
            sampled_conditions=full_conditions,
        )
        for candidate in shortlist
    ]
    full.sort(key=_score)
    active_engine = ResidualEngine(ACTIVE)
    active = {
        "guide_family": "active",
        "base_scale_multiplier": 1.0,
        "local_blend": 0.82,
        **_jump_metrics(active_engine, full_conditions),
        **_canonical_metrics(active_engine),
    }
    payload = {"active": active, "coarse_top": coarse[:18], "full_top": full}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Local delta base-scale sweep",
        "",
        f"| Baseline | active hybrid | Jump P95 {active['jump_p95_decades']:.4f} | Canonical max {active['canonical_jump_max_decades']:.4f} | Vth {active['generated_vth_mae_v']:.3f} | SS {active['generated_ss_mae_mv_dec']:.1f} |",
        "",
        "| Rank | Guide | base scale | local blend | Jump P95 | Canonical max | Gen. Vth | Gen. SS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(full, start=1):
        lines.append(
            f"| {index} | {item['guide_family']} | "
            f"{float(item['base_scale_multiplier']):.2f} | "
            f"{float(item['local_blend']):.2f} | "
            f"{float(item['jump_p95_decades']):.4f} | "
            f"{float(item['canonical_jump_max_decades']):.4f} | "
            f"{float(item['generated_vth_mae_v']):.3f} | "
            f"{float(item['generated_ss_mae_mv_dec']):.1f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    print(run())
