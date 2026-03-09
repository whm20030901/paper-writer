#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.data.burgers import fd_reference, sample_training_points
from src.trainers.train import train_baseline
from src.utils.io import ensure_dir, rel_l2, save_curve, save_json, save_solution_plot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    out = Path(cfg["output_dir"])
    ensure_dir(out)

    data = sample_training_points(
        n_f=cfg["data"]["n_f"],
        n_b=cfg["data"]["n_b"],
        n_i=cfg["data"]["n_i"],
        seed=cfg["seed"],
    )
    model, hist, extra = train_baseline(cfg, data)

    x_ref, t_ref, u_ref = fd_reference(nx=cfg["eval"]["nx"], nt=cfg["eval"]["nt"], nu=cfg.get("nu", 0.01 / np.pi))
    xx, tt = np.meshgrid(x_ref, t_ref)
    xt = np.stack([xx.reshape(-1), tt.reshape(-1)], axis=1)
    xt_t = torch.tensor(xt, dtype=torch.float32, device=cfg.get("device", "cpu"))
    with torch.no_grad():
        u_pred = model(xt_t)[0].cpu().numpy().reshape(len(t_ref), len(x_ref))

    metrics = {
        "rel_l2": rel_l2(u_pred, u_ref),
        "pde_residual_mse": float(np.mean(np.square(hist["pde"][-50:]))),
        "boundary_initial_mse": float(np.mean(np.square(hist["bc"][-50:])) + np.mean(np.square(hist["ic"][-50:]))),
        **extra,
    }

    torch.save(model.state_dict(), out / "model.pt")
    save_curve(out / "loss_curves.png", hist, title="Baseline PINN loss")
    save_solution_plot(out / "solution_compare.png", x_ref, t_ref, u_pred, u_ref)
    save_json(out / "metrics.json", metrics)
    save_json(out / "history.json", hist)


if __name__ == "__main__":
    main()
