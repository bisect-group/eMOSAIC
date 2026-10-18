"""Aggregate eMOSAIC results over seeds.

Reads `<results_dir>/<split_group>[/<threshold>]/seed_<n>/run_summary.json` and writes
mean and variance (across seeds) of each metric for train, val and test.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from emulator_bench.common import DEFAULT_BASE_DIR, DEFAULT_RESULTS_ROOT

METRICS = ["mae", "mse", "rmse", "r2_score", "pearson", "spearman", "loss"]
SPLITS = ("train", "val", "test")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_ROOT / DEFAULT_BASE_DIR.name)
    args = parser.parse_args()

    rows = []
    for summary_path in sorted(args.results_dir.rglob("seed_*/run_summary.json")):
        summary = json.loads(summary_path.read_text())
        seed_dir = summary_path.parent
        for split in SPLITS:
            rows.append(
                {
                    "split_type": seed_dir.parent.relative_to(args.results_dir).as_posix(),
                    "tvt": split,
                    "seed": int(seed_dir.name.removeprefix("seed_")),
                    **{m: v for m, v in summary[f"final_{split}_metrics"].items() if m in METRICS},
                }
            )
    if not rows:
        raise SystemExit(f"No run_summary.json files found under {args.results_dir}")

    grouped = pd.DataFrame(rows).groupby(["split_type", "tvt"], sort=False)
    wide = grouped[METRICS].agg(["mean", "var"])
    wide.columns = [f"{metric}_{stat}" for metric, stat in wide.columns]
    wide["n_seeds"] = grouped["seed"].nunique()
    out_path = args.results_dir / "aggregated_results_wide.csv"
    wide.reset_index().to_csv(out_path, index=False)
    print(wide.to_string(float_format="{:.4f}".format))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
