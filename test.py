import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# DỮ LIỆU - ImageNet-R (10 Tasks)
# ============================================================

tasks   = list(range(1, 11))
tasks_f = list(range(2, 11))   # Forgetting & CAA từ task 2

# ---------- NorGa-Prompt ----------
norga_acc1    = [81.12, 77.42, 77.95, 76.63, 76.71, 76.10, 74.76, 73.18, 72.92, 72.44]
norga_acctask = [100.00, 49.82, 35.01, 26.18, 19.91, 16.88, 14.42, 12.91, 11.18, 10.35]
norga_acc5    = [93.48, 92.98, 92.56, 90.63, 89.82, 89.10, 88.41, 87.36, 87.25, 87.09]
norga_forget  = [1.31, 2.12, 1.78, 2.03, 2.15, 2.47, 2.79, 3.05, 3.22]
norga_caa     = [79.36, 78.89, 78.34, 78.02, 77.70, 77.26, 76.79, 76.33, 75.97]

# ---------- HiDe-Prompt ----------
hide_acc1    = [80.70, 76.94, 77.18, 75.81, 76.07, 75.11, 74.19, 72.48, 72.43, 72.29]
hide_acctask = [100.00, 49.23, 34.45, 25.64, 19.35, 16.26, 13.83, 12.30, 10.59, 9.72]
hide_acc5    = [91.95, 90.39, 90.86, 89.66, 89.70, 88.63, 87.85, 87.09, 86.65, 86.48]
hide_forget  = [3.20, 4.10, 3.70, 3.29, 3.55, 3.28, 3.77, 3.88, 3.84]
hide_caa     = [78.86, 78.31, 77.69, 77.40, 77.04, 76.63, 76.15, 75.73, 75.40]

# ============================================================
# STYLE
# ============================================================

NORGA_C = "#2196F3"
HIDE_C  = "#FF5722"
LW, MS  = 2.2, 7

plt.rcParams.update({
    "font.family":         "DejaVu Sans",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.linestyle":      "--",
    "grid.alpha":          0.4,
    "legend.framealpha":   0.9,
    "axes.titlesize":      12,
    "axes.labelsize":      10,
})

# ============================================================
# LAYOUT: 2 hàng × 3 cột
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    "ImageNet-R Continual Learning: NorGa-Prompt vs HiDe-Prompt  (10 Tasks)",
    fontsize=14, fontweight="bold", y=1.01
)

def plot_metric(ax, x, y_norga, y_hide, title, ylabel, invert=False):
    ax.plot(x, y_norga, color=NORGA_C, marker="o", linewidth=LW,
            markersize=MS, label="NorGa-Prompt", zorder=3)
    ax.plot(x, y_hide,  color=HIDE_C,  marker="s", linewidth=LW,
            markersize=MS, label="HiDe-Prompt",  zorder=3)

    # Điểm giá trị cuối
    ax.annotate(f"{y_norga[-1]:.2f}", (x[-1], y_norga[-1]),
                textcoords="offset points", xytext=(7, 0),
                va="center", fontsize=8.5, color=NORGA_C, fontweight="bold")
    ax.annotate(f"{y_hide[-1]:.2f}",  (x[-1], y_hide[-1]),
                textcoords="offset points", xytext=(7, 0),
                va="center", fontsize=8.5, color=HIDE_C,  fontweight="bold")

    ax.set_title(title, fontweight="bold", pad=8)
    ax.set_xlabel("Task")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.legend(fontsize=9, loc="best")
    if invert:
        ax.invert_yaxis()

# --- Hàng 1 ---
plot_metric(axes[0,0], tasks,   norga_acc1,    hide_acc1,
            "Average Top-1 Accuracy (Acc@1) ↑", "Acc@1 (%)")

plot_metric(axes[0,1], tasks,   norga_acctask, hide_acctask,
            "Average Task Accuracy (Acc@task) ↑", "Acc@task (%)")

plot_metric(axes[0,2], tasks,   norga_acc5,    hide_acc5,
            "Average Top-5 Accuracy (Acc@5) ↑", "Acc@5 (%)")

# --- Hàng 2 ---
plot_metric(axes[1,0], tasks_f, norga_forget,  hide_forget,
            "Forgetting ↓  (thấp hơn = tốt hơn)", "Forgetting (%)")

plot_metric(axes[1,1], tasks_f, norga_caa,     hide_caa,
            "Class-Average Accuracy (CAA) ↑", "CAA (%)")

# --- Bảng tổng kết @ Task 10 ---
ax_t = axes[1, 2]
ax_t.axis("off")
ax_t.set_title("Summary @ Task 10", fontweight="bold", pad=8)

rows = [
    ["Acc@1 (%)",       norga_acc1[-1],    hide_acc1[-1],    True],
    ["Acc@task (%)",    norga_acctask[-1], hide_acctask[-1], True],
    ["Acc@5 (%)",       norga_acc5[-1],    hide_acc5[-1],    True],
    ["Forgetting (%)",  norga_forget[-1],  hide_forget[-1],  False],  # thấp = tốt
    ["CAA (%)",         norga_caa[-1],     hide_caa[-1],     True],
]

cell_text   = [[r[0], f"{r[1]:.2f}", f"{r[2]:.2f}"] for r in rows]
cell_colors = []
for r in rows:
    norga_wins = (r[1] > r[2]) if r[3] else (r[1] < r[2])
    if norga_wins:
        cell_colors.append(["#f5f5f5", NORGA_C+"55", "#f5f5f5"])
    else:
        cell_colors.append(["#f5f5f5", "#f5f5f5", HIDE_C+"55"])

tbl = ax_t.table(
    cellText=cell_text,
    colLabels=["Metric", "NorGa-Prompt", "HiDe-Prompt"],
    cellLoc="center", loc="center",
    bbox=[0, 0.05, 1, 0.88],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
# Header
for j in range(3):
    tbl[0, j].set_facecolor("#37474F")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
# Data rows
for i, colors in enumerate(cell_colors):
    for j, c in enumerate(colors):
        tbl[i+1, j].set_facecolor(c)

plt.tight_layout()
plt.savefig("imagenetr_comparison.png", dpi=150, bbox_inches="tight")
print("✅ Đã lưu: imagenetr_comparison.png")
plt.show()