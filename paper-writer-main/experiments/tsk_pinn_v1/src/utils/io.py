import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_curve(path, series_dict, title="Loss curves"):
    plt.figure(figsize=(7, 4))
    for name, vals in series_dict.items():
        plt.plot(vals, label=name)
    plt.yscale("log")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_heatmap(path, arr, title, xlabel="x", ylabel="t"):
    plt.figure(figsize=(6, 4))
    plt.imshow(arr, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_solution_plot(path, x, t, u_pred, u_ref):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(u_pred, origin="lower", aspect="auto", extent=[x.min(), x.max(), t.min(), t.max()])
    axes[0].set_title("u_pred")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(u_ref, origin="lower", aspect="auto", extent=[x.min(), x.max(), t.min(), t.max()])
    axes[1].set_title("u_ref")
    plt.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def rel_l2(u_pred, u_ref):
    p = np.asarray(u_pred)
    r = np.asarray(u_ref)
    return float(np.linalg.norm(p - r) / (np.linalg.norm(r) + 1e-12))
