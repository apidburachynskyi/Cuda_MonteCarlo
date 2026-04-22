import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.colors import TwoSlopeNorm, Normalize

# =============================================================================
# Configuration
# =============================================================================

INPUT_CSV = "res/results.csv"
OUTDIR = "res"

METHODS = [
    {
        "key": "euler1000",
        "family": "Euler",
        "dt_label": "dt=1/1000",
        "label": "Euler\n1/1000",
        "time_col": "time_euler1000",
        "price_col": "price_euler1000",
    },
    {
        "key": "euler30",
        "family": "Euler",
        "dt_label": "dt=1/30",
        "label": "Euler\n1/30",
        "time_col": "time_euler30",
        "price_col": "price_euler30",
    },
    {
        "key": "exact1000",
        "family": "Exact",
        "dt_label": "dt=1/1000",
        "label": "Exact\n1/1000",
        "time_col": "time_exact1000",
        "price_col": "price_exact1000",
    },
    {
        "key": "exact30",
        "family": "Exact",
        "dt_label": "dt=1/30",
        "label": "Exact\n1/30",
        "time_col": "time_exact30",
        "price_col": "price_exact30",
    },
    {
        "key": "ae1000",
        "family": "Almost Exact",
        "dt_label": "dt=1/1000",
        "label": "Almost Exact\n1/1000",
        "time_col": "time_ae1000",
        "price_col": "price_ae1000",
    },
    {
        "key": "ae30",
        "family": "Almost Exact",
        "dt_label": "dt=1/30",
        "label": "Almost Exact\n1/30",
        "time_col": "time_ae30",
        "price_col": "price_ae30",
    },
]

REFERENCE_PRICE_KEY = "exact1000"

FAMILY_COLORS = {
    "Euler": "tab:blue",
    "Exact": "tab:green",
    "Almost Exact": "tab:orange",
}

DT_HATCHES = {
    "dt=1/1000": "",
    "dt=1/30": "//",
}

plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 180,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
})


# =============================================================================
# Helpers
# =============================================================================

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_method(method_key: str) -> dict:
    for m in METHODS:
        if m["key"] == method_key:
            return m
    raise KeyError(f"Unknown method key: {method_key}")


def check_columns(df: pd.DataFrame) -> None:
    required = {"kappa", "theta", "sigma"}
    for m in METHODS:
        required.add(m["time_col"])
        required.add(m["price_col"])

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing columns in results.csv:\n" + "\n".join(f" - {c}" for c in missing)
        )


def nearest_value(values, target):
    values = np.asarray(sorted(values), dtype=float)
    idx = np.argmin(np.abs(values - target))
    return float(values[idx])


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary statistics for each method.

    Prices are converted to % of underlying:
      raw price 0.11 -> 11.0 (%)
    """
    ref = get_method(REFERENCE_PRICE_KEY)
    ref_price = df[ref["price_col"]].to_numpy(dtype=float)

    rows = []

    for m in METHODS:
        prices_raw = df[m["price_col"]].to_numpy(dtype=float)
        times_ms = df[m["time_col"]].to_numpy(dtype=float)

        prices_pct = 100.0 * prices_raw
        abs_err_pct = 100.0 * np.abs(prices_raw - ref_price)

        row = {
            "method": m["label"].replace("\n", " "),
            "family": m["family"],
            "dt": m["dt_label"],

            "price_mean_pct": np.mean(prices_pct),
            "price_std_pct": np.std(prices_pct, ddof=1),
            "price_min_pct": np.min(prices_pct),
            "price_max_pct": np.max(prices_pct),

            "time_mean_ms": np.mean(times_ms),
            "time_std_ms": np.std(times_ms, ddof=1),
            "time_min_ms": np.min(times_ms),
            "time_max_ms": np.max(times_ms),

            "abs_error_mean_pct": np.mean(abs_err_pct),
            "abs_error_std_pct": np.std(abs_err_pct, ddof=1),
            "abs_error_min_pct": np.min(abs_err_pct),
            "abs_error_max_pct": np.max(abs_err_pct),
        }
        rows.append(row)

    return pd.DataFrame(rows)

def build_summary_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build summary statistics for each method.

    Prices are converted to % of underlying:
      raw price 0.11 -> 11.0 (%)

    Output:
      index   = method
      columns = MultiIndex:
                [("Price","Mean"), ("Price","Std"), ("Price","Min"), ("Price","Max"),
                 ("Time","Mean"),  ("Time","Std"),  ("Time","Min"),  ("Time","Max")]
    """
    rows = []

    for m in METHODS:
        prices_raw = df[m["price_col"]].to_numpy(dtype=float)
        times_ms = df[m["time_col"]].to_numpy(dtype=float)

        prices_pct = 100.0 * prices_raw

        row = {
            ("Price (%)", "Mean"): np.mean(prices_pct),
            ("Price (%)", "Std"): np.std(prices_pct, ddof=1),
            ("Price (%)", "Min"): np.min(prices_pct),
            ("Price (%)", "Max"): np.max(prices_pct),

            ("Time (ms)", "Mean"): np.mean(times_ms),
            ("Time (ms)", "Std"): np.std(times_ms, ddof=1),
            ("Time (ms)", "Min"): np.min(times_ms),
            ("Time (ms)", "Max"): np.max(times_ms),
        }
        rows.append(row)

    summary = pd.DataFrame(
        rows,
        index=[m["label"].replace("\n", " ") for m in METHODS]
    )

    summary.index.name = "Method"
    summary.columns = pd.MultiIndex.from_tuples(summary.columns)

    return summary

def add_dual_legend(ax):
    family_handles = [
        Patch(facecolor=FAMILY_COLORS["Euler"], edgecolor="black", label="Euler"),
        Patch(facecolor=FAMILY_COLORS["Exact"], edgecolor="black", label="Exact"),
        Patch(facecolor=FAMILY_COLORS["Almost Exact"], edgecolor="black", label="Almost Exact"),
    ]

    hatch_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=DT_HATCHES["dt=1/1000"], label="dt=1/1000"),
        Patch(facecolor="white", edgecolor="black", hatch=DT_HATCHES["dt=1/30"], label="dt=1/30"),
    ]

    legend1 = ax.legend(
        handles=family_handles,
        title="Method family",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.00),
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=hatch_handles,
        title="Time step",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.58),
    )


def add_bar_value_labels(ax, bars, values, fmt="{:.2f}"):
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


# =============================================================================
# Figure 1 / Figure 2: Bar charts
# =============================================================================

def save_bar_chart(
    summary: pd.DataFrame,
    value_col: str,
    std_col: str,
    ylabel: str,
    title: str,
    filename: str,
    value_fmt: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(summary))
    labels = summary["method"].tolist()
    values = summary[value_col].to_numpy(dtype=float)
    stds = summary[std_col].to_numpy(dtype=float)

    colors = [FAMILY_COLORS[f] for f in summary["family"]]
    hatches = [DT_HATCHES[d] for d in summary["dt"]]

    bars = []
    for xi, val, std, color, hatch in zip(x, values, stds, colors, hatches):
        b = ax.bar(
            xi,
            val,
            yerr=std,
            capsize=6,
            color=color,
            edgecolor="black",
            hatch=hatch,
            width=0.75,
        )
        bars.append(b[0])

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    add_bar_value_labels(ax, bars, values, fmt=value_fmt)
    add_dual_legend(ax)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, filename))
    plt.close(fig)


# =============================================================================
# Figure 3: Heatmap grid
# =============================================================================

def annotate_heatmap(ax, data, fmt="{:.2f}", fontsize=8):
    arr = np.asarray(data, dtype=float)
    finite_vals = arr[np.isfinite(arr)]

    if finite_vals.size == 0:
        return

    vmin = np.min(finite_vals)
    vmax = np.max(finite_vals)
    mid = 0.5 * (vmin + vmax)

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=fontsize, color="gray")
            else:
                color = "white" if val < mid else "black"
                ax.text(
                    j, i, fmt.format(val),
                    ha="center", va="center",
                    fontsize=fontsize,
                    color=color,
                    fontweight="bold"
                )


def save_comparison_heatmap_grid(
    df: pd.DataFrame,
    outdir: str,
    sigma_targets=(0.10, 0.50, 1.00),
) -> None:
    """
    Top row:
        100 * (price_ae30 - price_euler1000)
        => percentage points of underlying

    Bottom row:
        time_euler1000 / time_ae30
        => speedup
    """

    available_sigmas = sorted(df["sigma"].unique())
    sigmas = [nearest_value(available_sigmas, s) for s in sigma_targets]

    kappas = sorted(df["kappa"].unique())
    thetas = sorted(df["theta"].unique())

    price_maps = []
    speed_maps = []

    for sigma in sigmas:
        sub = df[df["sigma"] == sigma].copy()

        sub["price_diff_pct"] = 100.0 * (sub["price_ae30"] - sub["price_euler1000"])
        sub["speedup"] = sub["time_euler1000"] / sub["time_ae30"]

        price_mat = (
            sub.pivot(index="kappa", columns="theta", values="price_diff_pct")
            .reindex(index=kappas, columns=thetas)
            .to_numpy(dtype=float)
        )

        speed_mat = (
            sub.pivot(index="kappa", columns="theta", values="speedup")
            .reindex(index=kappas, columns=thetas)
            .to_numpy(dtype=float)
        )

        price_maps.append(price_mat)
        speed_maps.append(speed_mat)

    # Shared scale across top row (centered on 0)
    price_abs_max = max(np.nanmax(np.abs(m)) for m in price_maps)
    price_norm = TwoSlopeNorm(vmin=-price_abs_max, vcenter=0.0, vmax=price_abs_max)

    # Shared scale across bottom row (centered on 1 when possible)
    speed_min = min(np.nanmin(m) for m in speed_maps)
    speed_max = max(np.nanmax(m) for m in speed_maps)

    if speed_min < 1.0 < speed_max:
        speed_norm = TwoSlopeNorm(vmin=speed_min, vcenter=1.0, vmax=speed_max)
    else:
        speed_norm = Normalize(vmin=speed_min, vmax=speed_max)

    fig, axes = plt.subplots(2, 3, figsize=(18, 9))

    # -------------------------
    # Top row: price diff heatmaps
    # -------------------------
    for j, (sigma, mat) in enumerate(zip(sigmas, price_maps)):
        ax = axes[0, j]
        ax.grid(False)

        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            norm=price_norm
        )

        ax.set_title(f"Price diff: AE 1/30 - Euler 1/1000\nsigma = {sigma:.2f}")
        ax.set_xlabel("theta")
        ax.set_ylabel("kappa")
        ax.set_xticks(range(len(thetas)))
        ax.set_xticklabels([f"{x:.2f}" for x in thetas])
        ax.set_yticks(range(len(kappas)))
        ax.set_yticklabels([f"{x:.1f}" for x in kappas])

        annotate_heatmap(ax, mat, fmt="{:.2f}", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Δ price (% of underlying)")

    # -------------------------
    # Bottom row: speedup heatmaps
    # -------------------------
    for j, (sigma, mat) in enumerate(zip(sigmas, speed_maps)):
        ax = axes[1, j]
        ax.grid(False)

        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            cmap="RdBu_r",
            norm=speed_norm
        )

        ax.set_title(f"Speedup: Euler 1/1000 / AE 1/30\nsigma = {sigma:.2f}")
        ax.set_xlabel("theta")
        ax.set_ylabel("kappa")
        ax.set_xticks(range(len(thetas)))
        ax.set_xticklabels([f"{x:.2f}" for x in thetas])
        ax.set_yticks(range(len(kappas)))
        ax.set_yticklabels([f"{x:.1f}" for x in kappas])

        annotate_heatmap(ax, mat, fmt="{:.2f}", fontsize=8)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Speedup")

    fig.suptitle(
        "Comparison between Euler dt=1/1000 and Almost Exact dt=1/30",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(outdir, "heatmap_price_diff_and_speedup_ae30_vs_euler1000.png"))
    plt.close(fig)


# =============================================================================
# Summary table outputs
# =============================================================================

def save_summary_table(summary: pd.DataFrame, outdir: str) -> None:
    # Save CSV
    rounded = summary.copy()

    for col in rounded.columns:
        if col.endswith("_pct"):
            rounded[col] = rounded[col].round(2)
        elif col.endswith("_ms"):
            rounded[col] = rounded[col].round(2)

    rounded.to_csv(os.path.join(outdir, "summary_table.csv"), index=False)

    # Save a PNG table as well
    display_cols = [
        "method",
        "price_mean_pct", "price_std_pct", "price_min_pct", "price_max_pct",
        "time_mean_ms", "time_std_ms", "time_min_ms", "time_max_ms",
        "abs_error_mean_pct", "abs_error_std_pct"
    ]

    disp = rounded[display_cols].copy()
    disp.columns = [
        "Method",
        "Price mean (%)", "Price std (%)", "Price min (%)", "Price max (%)",
        "Time mean (ms)", "Time std (ms)", "Time min (ms)", "Time max (ms)",
        "Abs err mean (%)", "Abs err std (%)"
    ]

    fig, ax = plt.subplots(figsize=(18, 3.8))
    ax.axis("off")

    table = ax.table(
        cellText=disp.values,
        colLabels=disp.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.05, 1.6)

    # Header style
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#D9EAF7")

    plt.title("Summary statistics across the parameter grid", fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "summary_table.png"))
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    ensure_outdir(OUTDIR)

    df = pd.read_csv(INPUT_CSV)
    check_columns(df)
    df = df.sort_values(["sigma", "kappa", "theta"]).reset_index(drop=True)

    summary = build_summary(df)

    try:
        from better_tables import build_table

        table = build_table(build_summary_clean(df),
                            title="Summary statistics across the parameter grid")
        table.to_latex(save_as_tex=True, save_as_pdf=True, path="res/summary_table")
    except ValueError as e:
        print(f"Could not build LaTeX table: {e}")
        
    # -------------------------------------------------------------------------
    # Figure 1: Mean GPU execution time
    # -------------------------------------------------------------------------
    save_bar_chart(
        summary=summary,
        value_col="time_mean_ms",
        std_col="time_std_ms",
        ylabel="GPU execution time (ms)",
        title="Mean GPU execution time across parameter grid\n(error bars = ±1 std)",
        filename="mean_gpu_execution_time_linear_clean.png",
        value_fmt="{:.2f}",
    )

    # -------------------------------------------------------------------------
    # Figure 2: Mean absolute pricing error vs Exact dt=1/1000
    # Expressed in % of underlying
    # -------------------------------------------------------------------------
    save_bar_chart(
        summary=summary,
        value_col="abs_error_mean_pct",
        std_col="abs_error_std_pct",
        ylabel="Absolute pricing error (% of underlying)",
        title="Mean absolute pricing error relative to Exact dt=1/1000\n(error bars = ±1 std)",
        filename="mean_pricing_error_vs_exact1000_clean.png",
        value_fmt="{:.2f}",
    )

    # -------------------------------------------------------------------------
    # Figure 3: Heatmap grid
    # -------------------------------------------------------------------------
    save_comparison_heatmap_grid(
        df=df,
        outdir=OUTDIR,
        sigma_targets=(0.10, 0.50, 1.00),
    )

    print("\nSaved files:")
    print(" - res/mean_gpu_execution_time_linear_clean.png")
    print(" - res/mean_pricing_error_vs_exact1000_clean.png")
    print(" - res/heatmap_price_diff_and_speedup_ae30_vs_euler1000.png")
    print(" - res/summary_table.csv")
    print(" - res/summary_table.png")


if __name__ == "__main__":
    main()