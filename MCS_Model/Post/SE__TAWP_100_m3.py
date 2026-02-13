import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.ticker import NullLocator

KEEP_COLS = [
    "start","realization","scenario","raw_materials","transportation_1",
    "processing","transportation_2","operation","transportation_3",
    "construction","use","deconstruction","transportation_4","EoL",
    "TAWP","GWP"
]

def get_TAWP_file(region_name, fil_name, analysis_year):
    base = os.path.join("..", "Scenarios", region_name, "D_CUBE", "TAWP_results")
    low  = pd.read_csv(os.path.join(base, f"{fil_name}_low_{analysis_year}.csv"))
    high = pd.read_csv(os.path.join(base, f"{fil_name}_high_{analysis_year}.csv"))

    low["start"] = 0
    high["start"] = 0

    m1 = (high.scenario == "EoL_1")
    m2 = (high.scenario == "EoL_2")

    low_e1  = low.loc[m1].reset_index(drop=True)[KEEP_COLS]
    low_e2  = low.loc[m2].reset_index(drop=True)[KEEP_COLS]
    high_e1 = high.loc[m1].reset_index(drop=True)[KEEP_COLS]
    high_e2 = high.loc[m2].reset_index(drop=True)[KEEP_COLS]
    return low_e1, low_e2, high_e1, high_e2


def _lump_transport(df):
    num = df.apply(pd.to_numeric, errors="coerce")
    idx = num.index
    Z   = pd.Series(0.0, index=idx)
    g   = lambda c: (num[c] if c in num.columns else Z)

    out = pd.DataFrame(index=idx)
    if "start" in num.columns:
        out["start"] = g("start")
    out["raw_materials"]  = g("raw_materials")   + g("transportation_1")
    out["processing"]     = g("processing")      + g("transportation_2")
    out["operation"]      = g("operation")       + g("transportation_3")
    out["construction"]   = g("construction")
    out["use"]            = g("use")
    out["deconstruction"] = g("deconstruction")  + g("transportation_4")
    out["EoL"]            = g("EoL")
    return out


def _csum(df, cols):
    return df[cols].cumsum(axis=1)


def _kde_on_ygrid(samples, y_grid, bw=0.25):
    a = np.asarray(samples)
    a = a[np.isfinite(a)]
    if a.size < 2 or np.allclose(np.std(a), 0.0):
        return np.zeros_like(y_grid)
    kde = gaussian_kde(a)
    kde.covariance_factor = (lambda: bw)
    kde._compute_covariance()
    return kde(y_grid)


def draw_TAWP_panel_ax(
    ax, clean_eol1, clean_eol2, dirty_eol3, dirty_eol4, *,
    ylims=(-4000, 0),
    phase_label_map=None,
    p_low=0.05, p_high=0.95,
    pdf_width=1.05, right_pad_frac=0.10,
    med_lw=2.8, band_alpha=0.20, pdf_alpha=0.16,
    sep_color="0.85", grid_alpha=0.30,
    show_phase_bands=True, show_vseps=True, show_grid=True,
    use_step=False, show_markers=False,
    fs_tick=18,
):
    L_c1 = _lump_transport(clean_eol1)
    L_c2 = _lump_transport(clean_eol2)
    L_d3 = _lump_transport(dirty_eol3)
    L_d4 = _lump_transport(dirty_eol4)

    ORDERED = ["start","raw_materials","processing","operation","construction","use","deconstruction","EoL"]
    phases  = [p for p in ORDERED if any(p in df.columns for df in (L_c1, L_c2, L_d3, L_d4))]
    x = np.arange(len(phases))

    c_c1, c_c2, c_d3, c_d4 = map(lambda d: _csum(d, phases), (L_c1, L_c2, L_d3, L_d4))

    med_c1 = c_c1.median(axis=0)
    med_c2 = c_c2.median(axis=0)
    med_d3 = c_d3.median(axis=0)
    med_d4 = c_d4.median(axis=0)

    clean_camp = pd.concat([c_c1, c_c2], axis=0, ignore_index=True)
    dirty_camp = pd.concat([c_d3, c_d4], axis=0, ignore_index=True)
    clean_lo, clean_hi = clean_camp.quantile(p_low, axis=0), clean_camp.quantile(p_high, axis=0)
    dirty_lo, dirty_hi = dirty_camp.quantile(p_low, axis=0), dirty_camp.quantile(p_high, axis=0)

    color_clean, color_dirty = "#542788", "#E66101"

    if show_phase_bands:
        for k in range(1, len(phases) + 1):
            if k % 2 == 0:
                ax.axvspan(k - 1, k, facecolor="0.97", edgecolor="none", zorder=0)

    ax.fill_between(x, clean_lo.values, clean_hi.values, alpha=band_alpha, color=color_clean, edgecolor="none", zorder=1)
    ax.fill_between(x, dirty_lo.values, dirty_hi.values, alpha=band_alpha, color=color_dirty, edgecolor="none", zorder=1)

    labels = (
        [phase_label_map.get(p, p.replace("_", "\n")) for p in phases]
        if phase_label_map else
        [p.replace("_", "\n") for p in phases]
    )

    ax.set_ylim(*ylims)
    y_grid = np.linspace(ylims[0], ylims[1], 1000)

    f_c1 = c_c1.iloc[:, -1].to_numpy()
    f_c2 = c_c2.iloc[:, -1].to_numpy()
    f_d3 = c_d3.iloc[:, -1].to_numpy()
    f_d4 = c_d4.iloc[:, -1].to_numpy()

    d_c1 = _kde_on_ygrid(f_c1, y_grid)
    d_c2 = _kde_on_ygrid(f_c2, y_grid)
    d_d3 = _kde_on_ygrid(f_d3, y_grid)
    d_d4 = _kde_on_ygrid(f_d4, y_grid)
    dmax = max(d_c1.max(), d_c2.max(), d_d3.max(), d_d4.max(), 1e-12)

    SCENARIOS = [
        (med_c2, d_c2, color_clean, "-"),   # Clean — Reuse
        (med_c1, d_c1, color_clean, "-."),  # Clean — Biochar
        (med_d4, d_d4, color_dirty, "-"),   # Dirty — Landfill
        (med_d3, d_d3, color_dirty, "-."),  # Dirty — Combustion
    ]

    def _draw_med(series, color, ls):
        if use_step:
            ax.step(x, series.values, where="post", color=color, lw=med_lw, linestyle=ls, zorder=3)
        else:
            ax.plot(x, series.values, color=color, lw=med_lw, linestyle=ls, zorder=3)
        if show_markers:
            ax.plot(x, series.values, marker="o", ms=3.0, lw=0, color=color, zorder=4)

    for med_series, _, col, ls in SCENARIOS:
        _draw_med(med_series, col, ls)

    x_edge = len(phases) - 1
    pdf_pad = right_pad_frac * pdf_width
    ax.axvspan(x_edge, x_edge + pdf_width + pdf_pad, facecolor="0.97", edgecolor="none", zorder=0.5)

    def _draw_pdf(d, color, ls):
        x_right = x_edge + (d / dmax) * pdf_width
        ax.plot(x_right, y_grid, color=color, lw=max(1.0, med_lw - 0.6), ls=ls, zorder=4)
        ax.fill_betweenx(y_grid, x_edge, x_right, color=color, alpha=pdf_alpha, zorder=2)

    for _, dens, col, ls in SCENARIOS:
        _draw_pdf(dens, col, ls)

    ax.axhline(0, linewidth=1, color="k")
    if show_vseps:
        for xb in range(1, len(phases)):
            ax.axvline(xb, color=sep_color, lw=0.8, linestyle=":", zorder=2)
    if show_grid:
        ax.grid(True, axis="y", alpha=grid_alpha, linewidth=0.8)

    ax.set_xticks(x - 0.5, labels)
    ax.tick_params(axis="x", labelsize=fs_tick)
    ax.tick_params(axis="y", labelsize=fs_tick)
    ax.tick_params(axis="both", pad=1)

    ax.axvline(x_edge, color="k", lw=1.6, alpha=1.0, zorder=7)
    ax.set_xlim(0, x_edge + pdf_width + right_pad_frac * pdf_width)

    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax.grid(False, axis="x", which="minor")


region_    = "SE"
region2_   = "SE"
pre_type_  = "softwood_"
prod_type_ = "CLT"

years = [2020, 2100]
anchor_labels = ["2025", "2100"]
panel_letters = ["D", "F"]

PHASE_LABELS_ROMAN = {
    "raw_materials":  "I",
    "processing":     "II",
    "operation":      "III",
    "construction":   "IV",
    "use":            "V",
    "deconstruction": "VI",
    "EoL":            "VII",
}

fil_name = f"{region_}_{pre_type_}{prod_type_}"
year_data = [(yr, *get_TAWP_file(region2_, fil_name, yr)) for yr in years]

shared_ylims = (-4000, 0)

fig, axes = plt.subplots(2, 1, figsize=(6.5, 7), dpi=300, sharey=True)

for i, (yr, c1, c2, d3, d4) in enumerate(year_data):
    draw_TAWP_panel_ax(
        axes[i], c1, c2, d3, d4,
        ylims=shared_ylims,
        phase_label_map=PHASE_LABELS_ROMAN,
        fs_tick=18,
        med_lw=2.8,
    )

    axes[i].text(
        0.985, 0.965, panel_letters[i],
        transform=axes[i].transAxes,
        ha="right", va="top",
        fontsize=14, fontweight="bold", color="0.15"
    )

    y_frac = 0.035 if i == 0 else 0.07
    axes[i].text(
        0.99, y_frac, anchor_labels[i],
        transform=axes[i].transAxes,
        ha="right", va="bottom",
        fontsize=14, fontweight="bold", color="k", zorder=10
    )

axes[0].set_xticklabels([])
axes[0].xaxis.set_minor_locator(NullLocator())

axes[-1].annotate(
    region2_,
    xy=(0.5, -0.26),
    xycoords="axes fraction",
    ha="center", va="top",
    fontsize=20, fontweight="bold"
)

for ax in axes:
    ax.tick_params(axis="y", which="major", labelsize=18, length=6, width=1.0)
    ax.tick_params(axis="x", which="major", labelsize=18, length=6, width=1.0)
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight("bold")

fig.subplots_adjust(bottom=0.18, left=0.18)

fig.canvas.draw()
p0, p1 = axes[0].get_position(), axes[1].get_position()
y_mid = 0.5 * (min(p0.y0, p1.y0) + max(p0.y1, p1.y1))

fig.text(
    -0.01, y_mid, prod_type_+r" TAWP$_{100}$ (kgCO$_2$e m$^{-3}$)",
    rotation="vertical", va="center", ha="center",
    fontsize=20
)

plt.show()
