r"""Train and benchmark the LightGBM fantasy score predictors (base + archetype).

Run:
    .\.venv\Scripts\python.exe src\train_predictor.py
    .\.venv\Scripts\python.exe src\features.py   # rebuild dataset first if needed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor_model import (
    META_ARCHETYPE_PATH,
    META_PATH,
    MODEL_ARCHETYPE_PATH,
    MODEL_PATH,
    ROOT,
    TEST_SEASON,
    VAL_SEASON,
    train_both,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def print_benchmark(meta: dict, label: str) -> None:
    print("\n" + "=" * 72)
    print(f"{label} | val={VAL_SEASON} test={TEST_SEASON}")
    print("=" * 72)
    for split, models in meta["benchmark"].items():
        print(f"\n--- {split.upper()} ---")
        print(f"{'model':<22}{'MAE':>8}{'RMSE':>8}{'corr':>8}")
        print("-" * 46)
        for name in ("roll5", "season_avg", "blend_60_40", "gbm_residual", "gbm_blend"):
            m = models[name]
            print(f"{name:<22}{m['mae']:>8.2f}{m['rmse']:>8.2f}{m['corr']:>8.3f}")
        print(f"top-20 recall roll5: {models['top20_recall_roll5']:.3f}")
        print(f"top-20 recall gbm:   {models['top20_recall_gbm']:.3f}")
    print(f"\nblend weight (roll5): {meta['blend_weight_roll5']:.2f}")


def print_side_by_side(base_meta: dict, arch_meta: dict) -> None:
    print("\n" + "#" * 72)
    print("SIDE-BY-SIDE (gbm_blend on test season)")
    print("#" * 72)
    print(f"{'model':<18}{'MAE':>8}{'corr':>8}{'top20':>8}")
    print("-" * 42)
    for label, meta in [("base", base_meta), ("archetype", arch_meta)]:
        test = meta["benchmark"]["test"]["gbm_blend"]
        recall = meta["benchmark"]["test"]["top20_recall_gbm"]
        print(f"{label:<18}{test['mae']:>8.2f}{test['corr']:>8.3f}{recall:>8.3f}")


def main() -> None:
    (_, base_meta), (_, arch_meta) = train_both(save=True)
    print_benchmark(base_meta, "BASE GBM")
    print_benchmark(arch_meta, "ARCHETYPE GBM")
    print_side_by_side(base_meta, arch_meta)
    print(f"\nsaved base model      -> {MODEL_PATH.relative_to(ROOT)}")
    print(f"saved base meta       -> {META_PATH.relative_to(ROOT)}")
    print(f"saved archetype model -> {MODEL_ARCHETYPE_PATH.relative_to(ROOT)}")
    print(f"saved archetype meta  -> {META_ARCHETYPE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
