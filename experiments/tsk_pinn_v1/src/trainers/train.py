import time

import numpy as np
import torch

from src.models.backbone import MLPBackbone
from src.models.losses import burgers_residual, mse
from src.models.tsk import TSKWeightHead


def _to_torch(arr, device):
    return torch.tensor(arr, dtype=torch.float32, device=device)


def train_baseline(cfg, data):
    device = torch.device(cfg.get("device", "cpu"))
    nu = cfg.get("nu", 0.01 / np.pi)

    model = MLPBackbone(
        in_dim=2,
        hidden_dim=cfg["model"]["hidden_dim"],
        depth=cfg["model"]["depth"],
        out_dim=1,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    xf = _to_torch(data["xf"], device)
    xb = _to_torch(data["xb"], device)
    ub = _to_torch(data["ub"], device)
    xi = _to_torch(data["xi"], device)
    ui = _to_torch(data["ui"], device)

    hist = {"total": [], "pde": [], "bc": [], "ic": []}
    t0 = time.time()
    for _ in range(cfg["train"]["epochs"]):
        opt.zero_grad()
        pde = mse(burgers_residual(model, xf, nu), torch.zeros((xf.size(0), 1), device=device))
        ub_pred, _ = model(xb)
        ui_pred, _ = model(xi)
        bc = mse(ub_pred, ub)
        ic = mse(ui_pred, ui)
        total = pde + bc + ic
        total.backward()
        opt.step()

        hist["total"].append(float(total.item()))
        hist["pde"].append(float(pde.item()))
        hist["bc"].append(float(bc.item()))
        hist["ic"].append(float(ic.item()))

    return model, hist, {"train_time_sec": time.time() - t0}


def train_tsk(cfg, data):
    device = torch.device(cfg.get("device", "cpu"))
    nu = cfg.get("nu", 0.01 / np.pi)

    model = MLPBackbone(
        in_dim=2,
        hidden_dim=cfg["model"]["hidden_dim"],
        depth=cfg["model"]["depth"],
        out_dim=1,
    ).to(device)
    tsk = TSKWeightHead(
        in_dim=2,
        hidden_dim=cfg["model"]["hidden_dim"],
        n_rules=cfg["tsk"]["n_rules"],
        bottleneck_dim=cfg["tsk"]["bottleneck_dim"],
        drop_rule=cfg["tsk"].get("drop_rule", 0.0),
    ).to(device)
    opt = torch.optim.Adam(list(model.parameters()) + list(tsk.parameters()), lr=cfg["train"]["lr"])

    xf = _to_torch(data["xf"], device)
    xb = _to_torch(data["xb"], device)
    ub = _to_torch(data["ub"], device)
    xi = _to_torch(data["xi"], device)
    ui = _to_torch(data["ui"], device)

    x_all = torch.cat([xf, xb, xi], dim=0)

    hist = {"total": [], "pde": [], "bc": [], "ic": [], "ent": []}
    t0 = time.time()
    mu_last = None
    w_last = None
    for _ in range(cfg["train"]["epochs"]):
        opt.zero_grad()

        uf, hf = model(xf)
        rf = burgers_residual(model, xf, nu)
        ub_pred, hb = model(xb)
        ui_pred, hi = model(xi)

        h_all = torch.cat([hf, hb, hi], dim=0)
        w_all, mu = tsk(x_all, h_all)
        w_f = w_all[: xf.size(0)]
        w_b = w_all[xf.size(0) : xf.size(0) + xb.size(0)]
        w_i = w_all[xf.size(0) + xb.size(0) :]

        pde = (w_f[:, 1:2] * (rf**2)).mean()
        bc = (w_b[:, 2:3] * ((ub_pred - ub) ** 2)).mean()
        ic = (w_i[:, 0:1] * ((ui_pred - ui) ** 2)).mean()

        mu_mean = mu.mean(dim=0)
        ent = -(mu_mean * (mu_mean + 1e-8).log()).sum()
        ent_reg = -cfg["tsk"].get("entropy_reg", 0.0) * ent

        total = pde + bc + ic + ent_reg
        total.backward()
        opt.step()

        hist["total"].append(float(total.item()))
        hist["pde"].append(float(pde.item()))
        hist["bc"].append(float(bc.item()))
        hist["ic"].append(float(ic.item()))
        hist["ent"].append(float(ent.item()))
        mu_last = mu.detach().cpu().numpy()
        w_last = w_all.detach().cpu().numpy()

    artifacts = {
        "train_time_sec": time.time() - t0,
        "rule_utilization": mu_last.mean(axis=0).tolist() if mu_last is not None else [],
        "weights_sample": w_last[: min(512, len(w_last))].tolist() if w_last is not None else [],
    }
    return model, tsk, hist, artifacts
