r"""Train and benchmark the LightGBM fantasy score predictor.

Run:
    .\.venv\Scripts\python.exe src\train_predictor.py
    .\.venv\Scripts\python.exe src\features.py   # rebuild dataset first if needed
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predictor_model import MODEL_PATH, META_PATH, ROOT, TEST_SEASON, VAL_SEASON, train

sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def print_benchmark(meta: dict) -> None:
    print("\n" + "=" * 72)
    print(f"BENCHMARK | val={VAL_SEASON} test={TEST_SEASON}")
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


def main() -> None:
    _, meta = train(save=True)
    print_benchmark(meta)
    print(f"\nsaved model -> {MODEL_PATH.relative_to(ROOT)}")
    print(f"saved meta  -> {META_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
