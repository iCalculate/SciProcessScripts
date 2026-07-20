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
BASE = (
    ROOT / "experiments" / "model-experiments-20260627-220813" / "attempt_1_pca16.npz"
)
GUIDE_ROOT = ROOT / "experiments" / "local-threshold-window-20260630-095055"
ACTIVE = ROOT / "models" / "residual-hybrid-threshold-pca.npz"

GUIDE_SETS = [
    {
        "name": "full16",
        "label": "local threshold full16",
        "guide": GUIDE_ROOT / "localwin_full16_hybrid-guide.npz",
        "reverse_guide": None,
    },
    {
        "name": "full20",
        "label": "local threshold full20",
        "guide": GUIDE_ROOT / "localwin_full20_hybrid-guide.npz",
        "reverse_guide": None,
    },
    {
        "name": "dirsplit16",
        "label": "local threshold dirsplit16",
        "guide": GUIDE_ROOT / "localwin_dirsplit16_hybrid-forward-guide.npz",
        "reverse_guide": GUIDE_ROOT / "localwin_dirsplit16_hybrid-reverse-guide.npz",
    },
]


def _score(item: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        item["jump_p95_decades"],
        item["canonical_jump_max_decades"],
        item["generated_vth_mae_v"],
        item["generated_ss_mae_mv_dec"],
    )


def _parameter_grid() -> list[dict[str, float]]:
    grid = []
    for local_blend, global_blend, delta_clip in itertools.product(
        [0.82, 0.94, 1.00],
        [0.00, 0.02],
        [0.12, 0.18, 0.24],
    ):
        grid.append(
            {
                "local_blend": local_blend,
                "global_blend": global_blend,
                "guide_delta_clip_decades": delta_clip,
            }
        )
    return grid


def _evaluate(
    output_dir: Path,
    guide_set: dict[str, object],
    params: dict[str, float],
    sampled_conditions: list[dict],
) -> dict[str, float | str]:
    checkpoint = output_dir / (
        f"{guide_set['name']}"
        f"_lb{int(round(params['local_blend'] * 100)):03d}"
        f"_gb{int(round(params['global_blend'] * 100)):02d}"
        f"_dc{int(round(params['guide_delta_clip_decades'] * 100)):02d}.npz"
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
        local_blend=params["local_blend"],
        global_blend=params["global_blend"],
        window_scale=3.0,
        min_window_v=0.22,
        guide_align_strength=0.0,
        guide_align_window_scale=2.0,
        guide_delta_clip_decades=params["guide_delta_clip_decades"],
        reverse_on_state_blend_scale=1.0,
        reverse_on_state_delta_scale=1.0,
        reverse_on_state_onset_u_scale=1.8,
        reverse_on_state_window_scale=1.2,
    )
    engine = ResidualEngine(checkpoint)
    return {
        "guide_family": str(guide_set["name"]),
        "guide_label": str(guide_set["label"]),
        **params,
        **_jump_metrics(engine, sampled_conditions),
        **_canonical_metrics(engine),
    }


def run() -> Path:
    output_dir = ROOT / "experiments" / f"local-threshold-delta-sweep-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_conditions = _sample_conditions(limit=24)
    full_conditions = _sample_conditions(limit=72)
    coarse: list[dict[str, float | str]] = []
    for guide_set in GUIDE_SETS:
        for params in _parameter_grid():
            coarse.append(_evaluate(output_dir, guide_set, params, coarse_conditions))
    coarse.sort(key=_score)
    shortlist = coarse[:12]
    full = [
        _evaluate(
            output_dir,
            next(item for item in GUIDE_SETS if item["name"] == candidate["guide_family"]),
            {
                "local_blend": float(candidate["local_blend"]),
                "global_blend": float(candidate["global_blend"]),
                "guide_delta_clip_decades": float(candidate["guide_delta_clip_decades"]),
            },
            full_conditions,
        )
        for candidate in shortlist
    ]
    full.sort(key=_score)
    active_engine = ResidualEngine(ACTIVE)
    active = {
        "guide_family": "active",
        "guide_label": "active_hybrid_onstate_retuned",
        "local_blend": 0.82,
        "global_blend": 0.06,
        "guide_delta_clip_decades": 0.0,
        **_jump_metrics(active_engine, full_conditions),
        **_canonical_metrics(active_engine),
    }
    payload = {"active": active, "coarse_top": coarse[:18], "full_top": full}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Local threshold delta sweep",
        "",
        f"| Baseline | active hybrid | Jump P95 {active['jump_p95_decades']:.4f} | Canonical max {active['canonical_jump_max_decades']:.4f} | Vth {active['generated_vth_mae_v']:.3f} | SS {active['generated_ss_mae_mv_dec']:.1f} |",
        "",
        "| Rank | Guide | local blend | global blend | delta clip | Jump P95 | Canonical max | Gen. Vth | Gen. SS |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(full, start=1):
        lines.append(
            f"| {index} | {item['guide_family']} | "
            f"{float(item['local_blend']):.2f} | "
            f"{float(item['global_blend']):.2f} | "
            f"{float(item['guide_delta_clip_decades']):.2f} | "
            f"{float(item['jump_p95_decades']):.4f} | "
            f"{float(item['canonical_jump_max_decades']):.4f} | "
            f"{float(item['generated_vth_mae_v']):.3f} | "
            f"{float(item['generated_ss_mae_mv_dec']):.1f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


if __name__ == "__main__":
    print(run())
