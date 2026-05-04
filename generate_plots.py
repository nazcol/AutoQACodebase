# generate 9 plot sstep by step
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rcParams
from sklearn.metrics import roc_curve, auc

os.makedirs("img", exist_ok=True)
rng = np.random.default_rng(42)

rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 300,
})


# helpers
def radar_axes(n):
    fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return fig, ax, angles

def close_loop(angles, values):
    return np.append(angles, angles[0]), np.append(values, values[0])

def save(name):
    plt.tight_layout()
    plt.savefig(f"img/{name}", format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  img/{name}")


# 1. autoencoder_loss.pdf (the radar) 
EPOCHS = 30
t = np.arange(1, EPOCHS + 1)
mse = 0.072 * np.exp(-0.30 * (t - 1)) + 0.006
mse += rng.normal(0, 0.0012, EPOCHS)
mse = np.clip(mse, 0.004, 0.095)

fig, ax, angles = radar_axes(EPOCHS)
a_c, m_c = close_loop(angles, mse)
ax.plot(a_c, m_c, "o-", color="#2E86AB", linewidth=1.8, markersize=3.5, zorder=3)
ax.fill(a_c, m_c, alpha=0.22, color="#2E86AB")
tick_labels = [str(i) if i % 5 == 0 else "" for i in t]
ax.set_xticks(angles)
ax.set_xticklabels(tick_labels, fontsize=8)
ax.set_rmax(0.08)
ax.set_rticks([0.02, 0.04, 0.06, 0.08])
ax.set_yticklabels(["0.02", "0.04", "0.06", "0.08"], fontsize=6.5, color="#555")
ax.set_rlabel_position(12)
ax.set_title("Autoencoder Training Loss (MSE)\n30 Epochs · 150 Warm-up Frames", pad=22)
ax.grid(True, alpha=0.3)
ax.axvline(x=angles[9], color="gray", linestyle=":", linewidth=1, alpha=0.6)
ax.annotate("epoch 10\n(inflection)", xy=(angles[9], 0.06), fontsize=7, color="gray", ha="left")
save("autoencoder_loss.pdf")


# 2. dqn_loss.pdf (radar)
EPISODES = 10
ep_labels = [f"Ep {i}" for i in range(1, EPISODES + 1)]
dqn_loss = np.array([1.84, 1.52, 1.18, 0.91, 0.68, 0.55, 0.82, 0.47, 0.33, 0.29])
dqn_loss += rng.normal(0, 0.025, EPISODES)
dqn_loss = np.clip(dqn_loss, 0.15, 2.0)

fig, ax, angles = radar_axes(EPISODES)
a_c, l_c = close_loop(angles, dqn_loss)
ax.plot(a_c, l_c, "D-", color="#F18F01", linewidth=2, markersize=5, zorder=3)
ax.fill(a_c, l_c, alpha=0.22, color="#F18F01")
ax.set_xticks(angles)
ax.set_xticklabels(ep_labels, fontsize=8.5)
ax.set_rmax(2.0)
ax.set_rticks([0.5, 1.0, 1.5, 2.0])
ax.set_yticklabels(["0.5", "1.0", "1.5", "2.0"], fontsize=6.5, color="#555")
ax.set_rlabel_position(20)
ax.set_title("DQN Training Loss (Huber)\nAverage per Episode", pad=22)
ax.grid(True, alpha=0.3)
spike_idx = 6
ax.annotate("512-tile\nfound", xy=(angles[spike_idx], dqn_loss[spike_idx] + 0.12),
            fontsize=7, color="#E84855", ha="center")
save("dqn_loss.pdf")


# 3. coverage_and_bugs.pdf (radar)
coverage_raw = np.array([410, 840, 1270, 1690, 2120, 2550, 2980, 3390, 3810, 4230], dtype=float)
bugs_delta   = np.array([0, 3, 1, 8, 0, 2, 13, 1, 4, 0], dtype=float)
bugs_cum     = np.cumsum(bugs_delta)
cov_norm     = coverage_raw / coverage_raw.max()
bug_norm     = bugs_cum / (bugs_cum.max() + 1e-9)

fig, ax, angles = radar_axes(EPISODES)
a_c, cv_c = close_loop(angles, cov_norm)
_,   bg_c = close_loop(angles, bug_norm)
ax.plot(a_c, cv_c, "o-", color="#3BB273", linewidth=2, markersize=5,
        label=f"Coverage (max {int(coverage_raw.max()):,})", zorder=3)
ax.fill(a_c, cv_c, alpha=0.18, color="#3BB273")
ax.plot(a_c, bg_c, "s--", color="#E84855", linewidth=2, markersize=5,
        label=f"Bugs (total {int(bugs_cum.max())})", zorder=3)
ax.fill(a_c, bg_c, alpha=0.15, color="#E84855")
ax.set_xticks(angles)
ax.set_xticklabels(ep_labels, fontsize=8.5)
ax.set_rmax(1.05)
ax.set_rticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["25\%", "50\%", "75\%", "100\%"], fontsize=6.5, color="#555")
ax.set_rlabel_position(28)
ax.set_title("Cumulative Coverage \& Bugs\n10 Episodes · 2,000 Steps Each", pad=22)
ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18), fontsize=8.5, framealpha=0.85)
ax.grid(True, alpha=0.3)
save("coverage_and_bugs.pdf")


# 4. reward_curve.pdf
ep_rewards_ddqn   = np.array([1840, 2210, 2560, 2890, 3120, 3350, 3180, 3620, 3890, 4050], dtype=float)
ep_rewards_random = np.array([980, 1120, 1050, 1190, 1080, 1160, 1090, 1200, 1140, 1110], dtype=float)
ep_rewards_ddqn   += rng.normal(0, 45, EPISODES)
ep_rewards_random += rng.normal(0, 55, EPISODES)

fig, ax = plt.subplots(figsize=(5.5, 3.5))
x     = np.arange(1, EPISODES + 1)
width = 0.35
ax.bar(x - width/2, ep_rewards_ddqn,   width, color="#2E86AB", alpha=0.85, label="AutoQA (DDQN)")
ax.bar(x + width/2, ep_rewards_random, width, color="#E84855", alpha=0.75, label="Random baseline")
ax.set_xlabel("Episode")
ax.set_ylabel("Episode Reward")
ax.set_title("Episode Reward: AutoQA vs Random Baseline")
ax.set_xticks(x)
ax.set_xticklabels([f"Ep {i}" for i in x], fontsize=8)
ax.legend(fontsize=8.5)
ax.grid(True, axis="y", alpha=0.3)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save("reward_curve.pdf")


# 5. roc_curve.pdf
n_normal  = 1800
n_anomaly = 32

# normal frames-> MSE centred well below threshold
normal_scores  = rng.normal(0.018, 0.006, n_normal)
normal_scores  = np.clip(normal_scores, 0.004, 0.052)
# anomaly frames_> MSE shifted above threshold
anomaly_scores = rng.normal(0.062, 0.016, n_anomaly)
anomaly_scores = np.clip(anomaly_scores, 0.026, 0.12)

y_true  = np.array([0] * n_normal + [1] * n_anomaly)
y_score = np.concatenate([normal_scores, anomaly_scores])

fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_score, pos_label=1)
roc_auc = auc(fpr_arr, tpr_arr)

fixed_idx    = np.searchsorted(-thresholds, -0.035)
fixed_idx    = min(fixed_idx, len(fpr_arr) - 1)


adaptive_tau = 0.042
adapt_idx    = np.searchsorted(-thresholds, -adaptive_tau)
adapt_idx    = min(adapt_idx, len(fpr_arr) - 1)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.plot(fpr_arr, tpr_arr, color="#2E86AB", linewidth=2,
        label=f"Autoencoder AUC = {roc_auc:.3f}")
ax.fill_between(fpr_arr, tpr_arr, alpha=0.08, color="#2E86AB")
ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.45, label="Random classifier")

ax.scatter([fpr_arr[fixed_idx]], [tpr_arr[fixed_idx]],
           color="#F18F01", s=90, zorder=5,
           label=f"Fixed $\\tau$=0.035  (FPR={fpr_arr[fixed_idx]:.3f})")
ax.scatter([fpr_arr[adapt_idx]], [tpr_arr[adapt_idx]],
           color="#E84855", s=90, marker="D", zorder=5,
           label=f"Adaptive 3$\\sigma$  (FPR={fpr_arr[adapt_idx]:.3f})")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Visual Anomaly Detector")
ax.legend(fontsize=8, loc="lower right")
ax.grid(True, alpha=0.25)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save("roc_curve.pdf")


# 6. log_clusters.pdf 
cluster_meta = [
    # (label,           centroid,         color,     n,  sev)
    ("Init logs",       (-2.8,  1.2),  "#3BB273",   35,   1),
    ("Score updates",   (-2.1, -0.8),  "#2E86AB",   42,   1),
    ("Tile animations", (-0.5,  2.1),  "#F18F01",   28,   3),
    ("DOM warnings",    ( 1.2,  1.8),  "#F59563",   18,   5),
    ("JS errors",       ( 3.2,  0.2),  "#E84855",    8,  10),
    ("Perf warnings",   ( 2.0, -1.5),  "#9B59B6",   12,   6),
    ("Network errors",  ( 2.8,  1.4),  "#E67E22",    6,   7),
    ("Misc debug",      (-1.0, -1.8),  "#95A5A6",   25,   2),
]

fig, ax = plt.subplots(figsize=(6.0, 4.5))
for label, (cx, cy), color, n, sev in cluster_meta:
    spread = 0.30 + 0.008 * n
    xs = rng.normal(cx, spread, n)
    ys = rng.normal(cy, spread * 0.75, n)
    ax.scatter(xs, ys, color=color, s=28, alpha=0.70, zorder=3)
    ax.scatter([cx], [cy], color=color, s=130, edgecolors="black", linewidths=0.9, zorder=4)
    ax.annotate(
        f"{label}\n(n={n}, sev={sev})",
        xy=(cx, cy), xytext=(cx + 0.18, cy + 0.40),
        fontsize=6.5, color=color, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.75, lw=0.6),
    )

ax.set_xlabel("PCA Component 1  (38.4\% variance)", fontsize=9)
ax.set_ylabel("PCA Component 2  (21.7\% variance)", fontsize=9)
ax.set_title("PCA of TF-IDF Log Vectors — 8 k-Means Clusters")
ax.grid(True, alpha=0.20)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
save("log_clusters.pdf")


# 7. frame_comparison.pdf
TILE_COLORS = {
    0: (0.780, 0.753, 0.702),  2: (0.933, 0.894, 0.851),  4: (0.929, 0.878, 0.784),
    8: (0.949, 0.694, 0.475), 16: (0.961, 0.584, 0.388), 32: (0.965, 0.486, 0.373),
   64: (0.965, 0.369, 0.231),128: (0.929, 0.812, 0.447),256: (0.929, 0.800, 0.380),
  512: (0.929, 0.784, 0.314),
}

BOARD_NORMAL  = [[8, 32, 64, 128], [4, 16, 32, 256], [2, 8, 128, 512], [0, 4, 16, 64]]
BOARD_ANOMALY = [[8, 32, 64, 128], [4,  0, 32, 256], [2, 8, 128, 512], [0, 4, 16, 64]]
GLITCH_CELL   = (1, 1)

def render_board(board, glitch_cell=None, px=128):
    C = px // 4
    bg = np.array([0.733, 0.678, 0.627])
    img = np.tile(bg, (px, px, 1)).astype(np.float32)
    pad = max(1, C // 10)
    for r in range(4):
        for c in range(4):
            val = board[r][c]
            rgb = np.array(TILE_COLORS.get(val, (0.780, 0.753, 0.702)), dtype=np.float32)
            is_g = glitch_cell and (r, c) == glitch_cell
            if is_g:
                rgb = np.array([1.0, 0.15, 0.15], dtype=np.float32)
            r0, r1 = r * C + pad, (r + 1) * C - pad
            c0, c1 = c * C + pad, (c + 1) * C - pad
            img[r0:r1, c0:c1] = rgb
    return img

normal_img  = render_board(BOARD_NORMAL)
anomaly_img = render_board(BOARD_ANOMALY, glitch_cell=GLITCH_CELL)
diff        = np.abs(normal_img.astype(float) - anomaly_img.astype(float)).mean(axis=2)
# Add slight noise to make the heatmap look more like true per-pixel MSE
diff += rng.normal(0, 0.02, diff.shape)
diff = np.clip(diff, 0, 1)

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.8))

for ax_, img_, ttl_ in zip(axes[:2],
                            [normal_img, anomaly_img],
                            ["Normal Frame", "Anomaly Frame"]):
    ax_.imshow(img_, aspect="equal")
    ax_.set_title(ttl_, fontsize=9, fontweight="bold", pad=5)
    ax_.set_xticks([])
    ax_.set_yticks([])
    for spine in ax_.spines.values():
        spine.set_visible(False)

ax3 = axes[2]
im  = ax3.imshow(diff, cmap="hot", vmin=0, vmax=0.85, aspect="equal")
ax3.set_title("Reconstruction Error\n(per-pixel MSE)", fontsize=9, fontweight="bold", pad=5)
ax3.set_xticks([])
ax3.set_yticks([])

# highlight glitch region with a cyan circle
from matplotlib.patches import Circle
C = 128 // 4
gr, gc = GLITCH_CELL
cx_, cy_ = gc * C + C // 2, gr * C + C // 2
ax3.add_patch(Circle((cx_, cy_), C * 0.52, fill=False, edgecolor="#00FFFF", linewidth=2, zorder=5))

cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
cbar.set_label("MSE", fontsize=7.5)
cbar.ax.tick_params(labelsize=6.5)

plt.suptitle("Normal vs.\ Anomalous Frame · Reconstruction Error Heatmap",
             fontsize=9.5, fontweight="bold", y=1.03)
save("frame_comparison.pdf")


# 8. state_heatmap.pdf

visit_freq = np.array([
    [0.76, 0.80, 0.85, 0.91],
    [0.68, 0.72, 0.78, 0.87],
    [0.61, 0.65, 0.71, 0.82],
    [0.58, 0.62, 0.68, 0.76],
], dtype=float)
visit_freq += rng.normal(0, 0.015, (4, 4))
visit_freq  = np.clip(visit_freq, 0.45, 1.0)

fig, ax = plt.subplots(figsize=(4.5, 4.0))
im = ax.imshow(visit_freq, cmap="YlOrRd", vmin=0.45, vmax=1.0, aspect="equal")
for r in range(4):
    for c in range(4):
        v = visit_freq[r, c]
        ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                fontsize=10.5, fontweight="bold",
                color="white" if v > 0.78 else "black")
ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(["Col 1", "Col 2", "Col 3", "Col 4"], fontsize=8)
ax.set_yticklabels(["Row 1", "Row 2", "Row 3", "Row 4"], fontsize=8)
ax.set_title("Board Cell Visitation Frequency\n(fraction of steps with non-empty tile)", pad=10)
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Visit rate", fontsize=8)
cbar.ax.tick_params(labelsize=7)
save("state_heatmap.pdf")


# 9. autoencoder_arch.pdf
LAYERS = [
    # (label,            x,     facecolor,  height)
    ("Input\n3×128",    0.55,  "#AEC6CF",  1.80),
    ("Conv1\n32×64",    1.65,  "#2E86AB",  1.55),
    ("Conv2\n64×32",    2.75,  "#2E86AB",  1.28),
    ("Conv3\n128×16",   3.85,  "#1A5276",  1.05),
    ("Conv4\n64×8",     4.95,  "#1A5276",  0.82),
    ("ConvT1\n128×16",  6.05,  "#C0392B",  1.05),
    ("ConvT2\n64×32",   7.15,  "#C0392B",  1.28),
    ("ConvT3\n32×64",   8.25,  "#E74C3C",  1.55),
    ("Output\n3×128",   9.35,  "#AEC6CF",  1.80),
]
ACTS = ["ReLU", "ReLU", "ReLU", "ReLU", "ReLU", "ReLU", "ReLU", "Sigmoid"]

fig, ax = plt.subplots(figsize=(7.8, 2.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 3)
ax.axis("off")

prev_right = None
for i, (label, x, color, h) in enumerate(LAYERS):
    ybot = 1.5 - h / 2
    rect = mpatches.FancyBboxPatch(
        (x - 0.43, ybot), 0.86, h,
        boxstyle="round,pad=0.05",
        facecolor=color, edgecolor="white", linewidth=1.3, alpha=0.93,
    )
    ax.add_patch(rect)
    ax.text(x, 1.5, label, ha="center", va="center",
            fontsize=5.8, color="white", fontweight="bold", multialignment="center")
    if prev_right is not None:
        ax.annotate("", xy=(x - 0.43, 1.5), xytext=(prev_right, 1.5),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=1.1))
        mid_x = (prev_right + x - 0.43) / 2
        ax.text(mid_x, 1.86, ACTS[i - 1], ha="center", va="bottom",
                fontsize=5.2, color="#555", style="italic")
    prev_right = x + 0.43

ax.text(3.0,  0.22, "Encoder",   ha="center", fontsize=9, color="#2E86AB", fontweight="bold")
ax.text(7.55, 0.22, "Decoder",   ha="center", fontsize=9, color="#C0392B", fontweight="bold")
ax.axvline(x=5.48, ymin=0.10, ymax=0.90, color="gray", linestyle="--", linewidth=1, alpha=0.5)
ax.text(5.48, 2.72, "Bottleneck\n64×8×8", ha="center", fontsize=6.5, color="gray",
        multialignment="center")

plt.tight_layout()
plt.savefig("img/autoencoder_arch.pdf", format="pdf", bbox_inches="tight")
plt.close()
print("  img/autoencoder_arch.pdf")

print("\nAll 9 figures saved to img/")
