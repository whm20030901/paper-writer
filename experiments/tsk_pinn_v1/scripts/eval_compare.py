#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_metrics(path):
    return json.load(open(path, "r", encoding="utf-8"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--tsk", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    base = load_metrics(Path(args.baseline) / "metrics.json")
    tsk = load_metrics(Path(args.tsk) / "metrics.json")

    rows = [
        ["model", "rel_l2", "pde_residual_mse", "boundary_initial_mse", "train_time_sec"],
        ["baseline", base.get("rel_l2"), base.get("pde_residual_mse"), base.get("boundary_initial_mse"), base.get("train_time_sec")],
        ["tsk", tsk.get("rel_l2"), tsk.get("pde_residual_mse"), tsk.get("boundary_initial_mse"), tsk.get("train_time_sec")],
    ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")

    summary = {
        "baseline": base,
        "tsk": tsk,
        "tsk_better_rel_l2": tsk["rel_l2"] < base["rel_l2"],
    }
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
