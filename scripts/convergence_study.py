"""Grid-convergence study on uniform Cartesian grids.

Computes max errors of the gradient, Laplacian and divergence operators against
exact solutions for a smooth test field and writes docs/convergence.png.

Usage (from the repository root):  python scripts/convergence_study.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "tests")]

from fvm_helpers import cartesian_grid, centers, divergence, gradient, laplacian, max_error  # noqa: E402

SIZES = [11, 21, 41, 81]


def run():
    rows = []
    for n in SIZES:
        NI, NJ, X, Y, m = cartesian_grid(n)
        x, y = centers(m)
        P = np.sin(np.pi * x) * np.cos(np.pi * y)
        G = gradient(NI, NJ, P, m, iterations=3)
        L = laplacian(NI, NJ, P, G, m)
        V = np.stack([np.sin(np.pi * x), np.cos(np.pi * y)], axis=-1)
        D = divergence(NI, NJ, V, m)
        gx = np.pi * np.cos(np.pi * x) * np.cos(np.pi * y)
        lap = -2.0 * np.pi**2 * P
        div = np.pi * np.cos(np.pi * x) - np.pi * np.sin(np.pi * y)
        rows.append((1.0 / (n - 1),
                     max_error(G[..., 0], gx, NI, NJ),
                     max_error(L, lap, NI, NJ),
                     max_error(L, lap, NI, NJ, ring=1),
                     max_error(D, div, NI, NJ, ring=1)))
    return np.array(rows)


def main():
    data = run()
    labels = ["gradient (all cells)", "Laplacian (all cells)",
              "Laplacian (interior)", "divergence (interior)"]
    print(f"{'h':>10}" + "".join(f"{l:>24}" for l in labels))
    for row in data:
        print(f"{row[0]:10.4f}" + "".join(f"{v:24.3e}" for v in row[1:]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = data[:, 0]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for k, (label, marker) in enumerate(zip(labels, "osd^"), start=1):
        ax.loglog(h, data[:, k], marker=marker, label=label)
    for order, style in [(1, ":"), (2, "--")]:
        ax.loglog(h, data[0, 3] * (h / h[0]) ** order, "k" + style, lw=1, label=f"order {order}")
    ax.set_xlabel("grid spacing h")
    ax.set_ylabel("max error vs. exact solution")
    ax.set_title("Grid convergence on uniform Cartesian grids")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    out = os.path.join(ROOT, "docs", "convergence.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    main()
