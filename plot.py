"""Plot per-dimension frequency scaling for each PE method.

The single most informative visualization of what each PE scaling method
actually does under the hood: how much it scales each dimension's rotation
frequency relative to plain RoPE.
"""
import math
import numpy as np
import matplotlib.pyplot as plt

# ---- match your GPTConfig defaults ----
d_head = 64
L = 256           # original context (= block_size for from-scratch training)
s = 2.0           # scale factor for extrapolation
base = 10000.0
alpha = 1.0
beta = 32.0

# ---- dimension pair indices: i = 0, 2, 4, ..., d-2 ----
i = np.arange(0, d_head, 2)

# ---- base RoPE frequencies ----
theta_base = np.exp(i * (-math.log(base) / d_head))

# ---- PI: divides positions by s, equivalent to scaling frequency by 1/s uniformly ----
theta_pi = theta_base / s

# ---- NTK-aware: scale the base ----
base_ntk = base * (s ** (d_head / (d_head - 2)))
theta_ntk_aware = np.exp(i * (-math.log(base_ntk) / d_head))

# ---- NTK-by-parts: per-dim γ ramp ----
cycles = L * theta_base / (2 * math.pi)
gamma = np.clip((cycles - alpha) / (beta - alpha), 0.0, 1.0)
theta_ntkbp = (1 - gamma) * (theta_base / s) + gamma * theta_base

# YaRN uses the same frequency mapping as NTK-by-parts (temperature affects amplitude only).
theta_yarn_freq = theta_ntkbp

# ---- plot ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

colors = {
    "RoPE":         "#1f1f1f",
    "PI":           "#e63946",
    "NTK-aware":    "#457b9d",
    "NTK-by-parts": "#2a9d8f",
    "YaRN":         "#f4a261",
}

# Left panel: absolute frequency, log scale
ax = axes[0]
ax.semilogy(i, theta_base,      'o-', color=colors["RoPE"],         label="RoPE",         markersize=5)
ax.semilogy(i, theta_pi,        's-', color=colors["PI"],           label="PI",           markersize=5)
ax.semilogy(i, theta_ntk_aware, '^-', color=colors["NTK-aware"],    label="NTK-aware",    markersize=5)
ax.semilogy(i, theta_ntkbp,     'd-', color=colors["NTK-by-parts"], label="NTK-by-parts", markersize=5)
ax.semilogy(i, theta_yarn_freq, 'x--', color=colors["YaRN"],        label="YaRN (freq)",  markersize=7)
ax.set_xlabel("dimension index i")
ax.set_ylabel(r"effective frequency $\theta_i$  (log scale)")
ax.set_title(f"Per-dimension frequency  |  d_head={d_head}, s={s}, L={L}")
ax.grid(True, alpha=0.3)
ax.legend(frameon=False)

# Right panel: scaling ratio relative to plain RoPE
ax = axes[1]
ax.axhline(1.0,   color="gray", linestyle=":", alpha=0.6)
ax.axhline(1 / s, color="gray", linestyle=":", alpha=0.6)
ax.text(d_head - 4, 1.02,     "no scaling",    color="gray", fontsize=9)
ax.text(d_head - 4, 1/s + 0.02, f"full PI (1/s={1/s:g})", color="gray", fontsize=9)

ax.plot(i, theta_base      / theta_base, 'o-', color=colors["RoPE"],         label="RoPE",         markersize=5)
ax.plot(i, theta_pi        / theta_base, 's-', color=colors["PI"],           label="PI",           markersize=5)
ax.plot(i, theta_ntk_aware / theta_base, '^-', color=colors["NTK-aware"],    label="NTK-aware",    markersize=5)
ax.plot(i, theta_ntkbp     / theta_base, 'd-', color=colors["NTK-by-parts"], label="NTK-by-parts", markersize=5)

ax.set_xlabel("dimension index i")
ax.set_ylabel(r"scaling ratio  $\theta_\mathrm{effective} / \theta_\mathrm{base}$")
ax.set_title("How much each method scales each dimension")
ax.set_ylim(0.4, 1.1)
ax.grid(True, alpha=0.3)
ax.legend(frameon=False, loc="center right")

plt.tight_layout()
plt.savefig("pe_comparison.png", dpi=150, bbox_inches="tight")
plt.savefig("pe_comparison.svg", bbox_inches="tight")
print("Saved: pe_comparison.png, pe_comparison.svg")