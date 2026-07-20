from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

from devicecurvegen.residual import ResidualEngine
from tools.build_hybrid_checkpoint import build_hybrid_checkpoint
from tools.evaluate_postprocess_rechecks import _canonical_metrics
from tools.run_model_experiments import _jump_metrics, _sample_conditions


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = (
    ROOT / "experiments" / "model-experiments-20260627-220813" / "attempt_1_pca16.npz"
)
DEFAULT_FORWARD_GUIDE = (
    ROOT
    / "experiments"
    / "direction-split-hybrid-20260629-145628"
    / "dirsplit_cond20_cond20-forward-guide.npz"
)
DEFAULT_REVERSE_GUIDE = (
    ROOT
    / "experiments"
    / "direction-split-hybrid-20260629-145628"
    / "dirsplit_cond20_cond20-reverse-guide.npz"
)


def _score(item: dict[str, float]) -> tuple[float, float, float, float]:
    return (
        item["jump_p95_decades"],
        item["canonical_jump_max_decades"],
        item["generated_vth_mae_v"],
        item["generated_ss_mae_mv_dec"],
    )


def _parameter_grid() -> list[dict[str, float]]:
    grid = []
    for local_blend, align_strength, delta_clip, reverse_blend_scale in itertools.product(
        [0.70, 0.76, 0.82],
        [0.0, 0.25, 0.50],
        [0.0, 0.18, 0.26],
        [0.90, 1.00],
    ):
        grid.append(
            {
                "local_blend": local_blend,
                "guide_align_strength": align_strength,
                "guide_delta_clip_decades": delta_clip,
                "reverse_on_state_blend_scale": reverse_blend_scale,
            }
        )
    return grid


def _evaluate(
    output_dir: Path,
    *,
    base_path: Path,
    guide_path: Path,
    reverse_guide_path: Path,
    params: dict[str, float],
    sampled_conditions: list[dict],
) -> dict[str, float]:
    checkpoint = output_dir / (
        "focus"
        f"_lb{int(round(params['local_blend'] * 100)):02d}"
        f"_ga{int(round(params['guide_align_strength'] * 100)):02d}"
        f"_dc{int(round(params['guide_delta_clip_decades'] * 100)):02d}"
        f"_rb{int(round(params['reverse_on_state_blend_scale'] * 100)):02d}.npz"
    )
    build_hybrid_checkpoint(
        base_path=base_path,
        guide_path=guide_path,
        reverse_guide_path=reverse_guide_path,
        output_path=checkpoint,
        local_blend=params["local_blend"],
        global_blend=0.06,
        window_scale=3.0,
        min_window_v=0.22,
        guide_align_strength=params["guide_align_strength"],
        guide_align_window_scale=2.0,
        guide_delta_clip_decades=params["guide_delta_clip_decades"],
        reverse_on_state_blend_scale=params["reverse_on_state_blend_scale"],
        reverse_on_state_delta_scale=1.0,
        reverse_on_state_onset_u_scale=1.8,
        reverse_on_state_window_scale=1.2,
    )
    engine = ResidualEngine(checkpoint)
    return {
        **params,
        **_jump_metrics(engine, sampled_conditions),
        **_canonical_metrics(engine),
    }


def run(*, base_path: Path, guide_path: Path, reverse_guide_path: Path) -> Path:
    output_dir = ROOT / "experiments" / f"direction-split-focus-sweep-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    coarse_conditions = _sample_conditions(limit=24)
    full_conditions = _sample_conditions(limit=72)
    coarse = [
        _evaluate(
            output_dir,
            base_path=base_path,
            guide_path=guide_path,
            reverse_guide_path=reverse_guide_path,
            params=params,
            sampled_conditions=coarse_conditions,
        )
        for params in _parameter_grid()
    ]
    coarse.sort(key=_score)
    shortlist = coarse[:10]
    full = [
        _evaluate(
            output_dir,
            base_path=base_path,
            guide_path=guide_path,
            reverse_guide_path=reverse_guide_path,
            params={
                "local_blend": item["local_blend"],
                "guide_align_strength": item["guide_align_strength"],
                "guide_delta_clip_decades": item["guide_delta_clip_decades"],
                "reverse_on_state_blend_scale": item["reverse_on_state_blend_scale"],
            },
            sampled_conditions=full_conditions,
        )
        for item in shortlist
    ]
    full.sort(key=_score)
    payload = {"coarse_top": coarse[:16], "full_top": full}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Direction-split focus sweep",
        "",
        "| Rank | local blend | align | delta clip | rev blend | Jump P95 | Canonical max | Gen. Vth | Gen. SS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for index, item in enumerate(full, start=1):
        lines.append(
            f"| {index} | "
            f"{item['local_blend']:.2f} | "
            f"{item['guide_align_strength']:.2f} | "
            f"{item['guide_delta_clip_decades']:.2f} | "
            f"{item['reverse_on_state_blend_scale']:.2f} | "
            f"{item['jump_p95_decades']:.4f} | "
            f"{item['canonical_jump_max_decades']:.4f} | "
            f"{item['generated_vth_mae_v']:.3f} | "
            f"{item['generated_ss_mae_mv_dec']:.1f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Focused parameter sweep around a direction-split hybrid guide pair."
    )
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--guide", type=Path, default=DEFAULT_FORWARD_GUIDE)
    parser.add_argument("--reverse-guide", type=Path, default=DEFAULT_REVERSE_GUIDE)
    args = parser.parse_args()
    print(
        run(
            base_path=args.base.expanduser().resolve(),
            guide_path=args.guide.expanduser().resolve(),
            reverse_guide_path=args.reverse_guide.expanduser().resolve(),
        )
    )


if __name__ == "__main__":
    main()
