##### Total TAWP100 Plot 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os, warnings, re, glob
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedLocator, FixedFormatter, MaxNLocator
import matplotlib.patheffects as pe

#########################################################################
##### Read and Prep Data
#########################################################################
# Read Product Data:
# Out DFs
pnw_CLT_S1=pd.read_csv(r"Results\Projections\pnw_CLT_S1.csv",index_col = 0)
se_CLT_S1=pd.read_csv(r"Results\Projections\se_CLT_S1.csv",index_col = 0)
pnw_CLT_S2=pd.read_csv(r"Results\Projections\pnw_CLT_S2.csv",index_col = 0)
se_CLT_S2=pd.read_csv(r"Results\Projections\se_CLT_S2.csv",index_col = 0)
pnw_CLT_S3=pd.read_csv(r"Results\Projections\pnw_CLT_S3.csv",index_col = 0)
se_CLT_S3=pd.read_csv(r"Results\Projections\se_CLT_S3.csv",index_col = 0)

pnw_glt_S1=pd.read_csv(r"Results\Projections\pnw_GLT_S1.csv",index_col = 0)
se_glt_S1=pd.read_csv(r"Results\Projections\se_GLT_S1.csv",index_col = 0)
pnw_glt_S2=pd.read_csv(r"Results\Projections\pnw_GLT_S2.csv",index_col = 0)
se_glt_S2=pd.read_csv(r"Results\Projections\se_GLT_S2.csv",index_col = 0)
pnw_glt_S3=pd.read_csv(r"Results\Projections\pnw_GLT_S3.csv",index_col = 0)
se_glt_S3=pd.read_csv(r"Results\Projections\se_GLT_S3.csv",index_col = 0)

# ---------------------------
# Paths & years
# ---------------------------
analysis_years      = [2020,2030,2040,2050,2060,2070,2080,2090,2100,2110,2120]
EMISSION_SCENARIOS  = ["low", "high"]  # emissions scenarios

def resolve_code_root() -> Path:

    env = os.getenv("MCS_CODE_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"MCS_CODE_DIR was set but is not a directory: {p}")
        return p

    try:
        here = Path(__file__).resolve()
    except NameError:
        here = Path.cwd().resolve()

    for parent in [here] + list(here.parents):
        if parent.name.lower() == "code":
            return parent.resolve()

    raise FileNotFoundError(
        f"Could not locate the 'Code' folder walking up from:\n  {here}\n"
        "If needed, set environment variable MCS_CODE_DIR to the full path of the Code folder."
    )

CODE_ROOT = resolve_code_root()

PNW_CLT_path = CODE_ROOT / "MCS_Model" / "Scenarios" / "NW" / "D_CUBE" / "TAWP_results"
SE_CLT_path  = CODE_ROOT / "MCS_Model" / "Scenarios" / "SE" / "D_CUBE" / "TAWP_results"

if not PNW_CLT_path.is_dir():
    raise FileNotFoundError(f"Missing directory: {PNW_CLT_path}")
if not SE_CLT_path.is_dir():
    raise FileNotFoundError(f"Missing directory: {SE_CLT_path}")

PNW_CLT_path_str = str(PNW_CLT_path)
SE_CLT_path_str  = str(SE_CLT_path)


# Single directory map (used for BOTH products)
REGION_DIRS = {"PNW": PNW_CLT_path, "SE": SE_CLT_path}

# ---------------------------
#  TAWP loader (CLT + Glulam)
# ---------------------------
_EMIS_RE = re.compile(r"_(low|high)_(\d{4})\.csv$", re.IGNORECASE)

def _read_one_csv(path):
    df = pd.read_csv(path)
    # normalize expected columns
    if "TAWP" not in df.columns:
        alt = [c for c in df.columns if c.lower() in {"tawp100","tawp_100","tawp"}]
        if alt: df = df.rename(columns={alt[0]:"TAWP"})
        else:   raise ValueError(f"'TAWP' column not found in {path}")
    if "scenario" not in df.columns:
        raise ValueError(f"'scenario' (EoL_1/EoL_2) not found in {path}")
    if "realization" not in df.columns:
        df = df.copy(); df["realization"] = np.arange(len(df), dtype=int)
    return df.rename(columns={"scenario":"eol"})

def load_tawp_wide_from_clt_dirs(products,
                                 region_dirs,
                                 years):

    records = []
    report = []

    for product in products:
        for region, root in region_dirs.items():
            if not os.path.isdir(root):
                warnings.warn(f"Directory not found: {root}")
                continue

            patt = os.path.join(root, f"{region}_softwood_{product}_*.csv")
            hits = sorted(glob.glob(patt))
            found = 0

            for path in hits:
                m = _EMIS_RE.search(os.path.basename(path))
                if not m:
                    continue
                emis, ystr = m.group(1).lower(), m.group(2)
                yr = int(ystr)
                if yr not in years: 
                    continue

                df = _read_one_csv(path)
                out = (df[["realization","eol","TAWP"]]
                       .rename(columns={"TAWP":"tawp_kgco2e_per_m3"}))
                out["product"]           = product          # "CLT" or "Glulam"
                out["region"]            = region           # "PNW" / "SE"
                out["emission_scenario"] = emis             # "low"/"high"
                out["year"]              = yr
                records.append(out); found += 1

            report.append((product, region, root, found))

    # Discovery report
    rep = pd.DataFrame(report, columns=["product","region","root","files_found"])
    print("\nTAWP discovery summary (CLT dirs):\n", rep.to_string(index=False))

    if not records:
        raise RuntimeError("No TAWP CSVs discovered. Check folder paths and filename tokens.")

    tawp_long = pd.concat(records, ignore_index=True)
    tawp_wide = (tawp_long
        .pivot_table(index=["product","region","emission_scenario","year","realization"],
                     columns="eol", values="tawp_kgco2e_per_m3", aggfunc="first")
        .reset_index()
        .rename(columns={"EoL_1":"TAWP_EoL_1", "EoL_2":"TAWP_EoL_2"}))

    # hygiene
    tawp_wide = tawp_wide.sort_values(
        ["product","region","emission_scenario","year","realization"]
    ).reset_index(drop=True)
    tawp_wide["year"] = tawp_wide["year"].astype(int)
    tawp_wide["realization"] = tawp_wide["realization"].astype(int)

    return tawp_wide

# Load both products from CLT directories
tawp_wide = load_tawp_wide_from_clt_dirs(["CLT","Glulam"], REGION_DIRS, analysis_years)

# ---------------------------
# Projections 
# ---------------------------
def proj_matrix_to_long(df_matrix: pd.DataFrame, region: str,
                        product: str,
                        keep_years: list[int] | None = None) -> pd.DataFrame:
    df = df_matrix.copy()
    df.columns = df.columns.astype(int)
    if keep_years is not None:
        df = df.loc[:, [y for y in df.columns if int(y) in keep_years]]
    long = (df.reset_index(names="realization")
              .melt(id_vars="realization", var_name="year", value_name="production_m3"))
    long["product"] = product
    long["region"]  = region
    long["year"]    = long["year"].astype(int)
    return long[["product","region","year","realization","production_m3"]]

# ===== Build projections (S1 reused across emissions) =====
proj = pd.concat([
    proj_matrix_to_long(pnw_CLT_S1, region="PNW", product="CLT",    keep_years=analysis_years),
    proj_matrix_to_long(se_CLT_S1,  region="SE",  product="CLT",    keep_years=analysis_years),
    proj_matrix_to_long(pnw_glt_S1, region="PNW", product="Glulam", keep_years=analysis_years),
    proj_matrix_to_long(se_glt_S1,  region="SE",  product="Glulam", keep_years=analysis_years),
], ignore_index=True)

# cross-join emissions
proj = proj.merge(pd.DataFrame({"emission_scenario": EMISSION_SCENARIOS}), how="cross")

# Merge with TAWP (wide) + compute impacts
merged = proj.merge(
    tawp_wide,
    on=["product","region","emission_scenario","year","realization"],
    how="left",
    validate="one_to_one"
)
miss = merged[merged[["TAWP_EoL_1","TAWP_EoL_2"]].isna().any(axis=1)]
if not miss.empty:
    print("\nWARNING: Missing TAWP for some rows. Example keys:\n",
          miss[["product","region","emission_scenario","year","realization"]].head())

merged["impact_EoL_1_kgCO2e"] = merged["production_m3"] * merged["TAWP_EoL_1"]
merged["impact_EoL_2_kgCO2e"] = merged["production_m3"] * merged["TAWP_EoL_2"]

# Totals (PNW+SE) per product × emission_scenario × year × realization
totals_by_product = (merged
    .groupby(["product","emission_scenario","year","realization"], as_index=False)
    .agg(production_total_m3 = ("production_m3","sum"),
         impact_EoL_1_kgCO2e = ("impact_EoL_1_kgCO2e","sum"),
         impact_EoL_2_kgCO2e = ("impact_EoL_2_kgCO2e","sum"))
)

# Combined CLT + Glulam totals
totals_combined = (totals_by_product
    .groupby(["emission_scenario","year","realization"], as_index=False)
    .agg(production_total_m3 = ("production_total_m3","sum"),
         impact_EoL_1_kgCO2e = ("impact_EoL_1_kgCO2e","sum"),
         impact_EoL_2_kgCO2e = ("impact_EoL_2_kgCO2e","sum"))
)

#########################################################################
##### Plot
#########################################################################
plt.rcParams.update({
    "savefig.dpi": 600,            
    "figure.dpi": 300,             
    "font.size": 20,               
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 13,
    "axes.linewidth": 1.2,
    "xtick.major.size": 5.5,
    "ytick.major.size": 5.5,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})

# ---------------------------
# Projections 
# ---------------------------
def proj_matrix_to_long(df_matrix, region,
                        product,
                        keep_years):
    df = df_matrix.copy()
    df.columns = df.columns.astype(int)
    df = df.loc[:, [y for y in df.columns if int(y) in keep_years]]
    long = (df.reset_index(names="realization")
              .melt(id_vars="realization", var_name="year", value_name="production_m3"))
    long["product"] = product
    long["region"]  = region
    long["year"]    = long["year"].astype(int)
    return long[["product","region","year","realization","production_m3"]]

def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["product_key"] = out["product"].astype(str).str.casefold()
    out["region_key"]  = out["region"].astype(str).str.upper()
    # column may be named 'emission_scenario' or 'adoption' upstream; we expect 'emission_scenario' here
    if "emission_scenario" in out.columns:
        out["emis_key"] = out["emission_scenario"].astype(str).str.lower()
    elif "adoption" in out.columns:
        out["emis_key"] = out["adoption"].astype(str).str.lower()
    else:
        out["emis_key"] = ""
    out["year"]        = out["year"].astype(int)
    out["realization"] = out["realization"].astype(int)
    return out

def build_totals_for_projection(pnw_df, se_df,
                                product,
                                tawp_wide,
                                years,
                                emission_scenarios = ["low","high"]):
    # Projections (PNW + SE), then cross-join emissions
    proj = pd.concat([
        proj_matrix_to_long(pnw_df, region="PNW", product=product, keep_years=years),
        proj_matrix_to_long(se_df,  region="SE",  product=product, keep_years=years),
    ], ignore_index=True)
    proj = proj.merge(pd.DataFrame({"emission_scenario": emission_scenarios}), how="cross")

    # Normalize keys 
    proj_n = _norm_keys(proj)
    tawp_n = _norm_keys(tawp_wide.rename(columns={"TAWP_EoL_1":"TAWP1",
                                                  "TAWP_EoL_2":"TAWP2"}))

    # FILTER TAWP rows 
    tawp_n = tawp_n.loc[tawp_n["product_key"] == product.casefold()].copy()

    # Merge 
    merged = proj_n.merge(
        tawp_n[["product_key","region_key","emis_key","year","realization","TAWP1","TAWP2"]],
        on=["product_key","region_key","emis_key","year","realization"],
        how="left",
        validate="one_to_one"
    )

    # Reconstruct canonical columns for grouping
    merged["product"]            = product                           # constant for this call
    merged["emission_scenario"]  = merged["emis_key"]                # low/high (normalized)

    matched = merged["TAWP1"].notna().sum()
    total   = len(merged)
    print(f"[{product}] matched {matched:,}/{total:,} rows ({matched/total:.1%}).")

    # Compute impacts
    merged["impact_EoL_1_kgCO2e"] = merged["production_m3"] * merged["TAWP1"]
    merged["impact_EoL_2_kgCO2e"] = merged["production_m3"] * merged["TAWP2"]

    # Group to PNW+SE totals per product × emission_scenario × year × realization
    totals = (merged
        .groupby(["product","emission_scenario","year","realization"], as_index=False)
        .agg(production_total_m3 = ("production_m3","sum"),
             impact_EoL_1_kgCO2e = ("impact_EoL_1_kgCO2e","sum"),
             impact_EoL_2_kgCO2e = ("impact_EoL_2_kgCO2e","sum"))
    )
    return totals

def combine_products_totals(*totals_list: pd.DataFrame) -> pd.DataFrame:
    """Sum across products to get CLT+Glulam combined totals."""
    comb = pd.concat(totals_list, ignore_index=True)
    comb = (comb
        .groupby(["emission_scenario","year","realization"], as_index=False)
        .agg(production_total_m3 = ("production_total_m3","sum"),
             impact_EoL_1_kgCO2e = ("impact_EoL_1_kgCO2e","sum"),
             impact_EoL_2_kgCO2e = ("impact_EoL_2_kgCO2e","sum"))
    )
    return comb

# ---------------------------
# Plot Fxns
# ---------------------------
def _as_float(a): return np.asarray(a, dtype="float64")

def plot_TAWP_timepanel(ax, totals_by_realization: pd.DataFrame, title: str = "",
                        q_low=0.05, q_high=0.95,
                        color_low="tab:blue", color_high="tab:red",
                        med_lw=2.8, band_alpha=0.18,
                        label_fs=16, tick_fs=14,
                        face_gray=0.975, ylims=None,
                        anchor_years=(2020, 2040, 2060, 2080, 2100, 2120)):
    """Draw one panel with pooled 95% bands and median lines; minimal ticks and no title."""
    # long with EoL split; convert to Mt CO2e/yr
    imp_long = totals_by_realization.melt(
        id_vars=["emission_scenario","year","realization"],
        value_vars=["impact_EoL_1_kgCO2e","impact_EoL_2_kgCO2e"],
        var_name="eol_var", value_name="impact_kgCO2e"
    )
    imp_long["eol"] = imp_long["eol_var"].map({
        "impact_EoL_1_kgCO2e": "EoL_1",
        "impact_EoL_2_kgCO2e": "EoL_2",
    })
    imp_long.drop(columns="eol_var", inplace=True)
    imp_long["impact_MtCO2e"] = imp_long["impact_kgCO2e"] / 1e9
    imp_long["year"] = imp_long["year"].astype(int)
    years = np.sort(imp_long["year"].unique())

    def pooled_band(scn):
        q = (imp_long.loc[imp_long.emission_scenario.eq(scn)]
             .groupby("year")["impact_MtCO2e"]
             .quantile([q_low, q_high]).unstack())
        q = q.reindex(years).astype(float)
        return years.astype(float), q[q_low].to_numpy(), q[q_high].to_numpy()

    # medians per (scenario, EoL)
    def med_line(scn, eol):
        s = (imp_long.loc[imp_long.emission_scenario.eq(scn) & imp_long.eol.eq(eol)]
             .groupby("year")["impact_MtCO2e"].median().reindex(years))
        return years.astype(float), s.to_numpy()

    ax.set_facecolor(str(face_gray))
    x, p_lo, p_hi = pooled_band("low");  ax.fill_between(x, p_lo, p_hi, color=color_low,  alpha=band_alpha, linewidth=0, zorder=1)
    x, p_lo, p_hi = pooled_band("high"); ax.fill_between(x, p_lo, p_hi, color=color_high, alpha=band_alpha, linewidth=0, zorder=1)

    halo = [pe.Stroke(linewidth=med_lw+1.6, foreground="white", alpha=0.95), pe.Normal()]
    def _plot_line(x, y, color, ls):
        ax.plot(x, y, color=color, lw=med_lw, ls=ls, zorder=3,
                solid_capstyle="round", solid_joinstyle="round",
                dash_capstyle="round",  dash_joinstyle="round",
                path_effects=halo)
    x, y = med_line("low",  "EoL_2");  _plot_line(x, y, color_low,  "-")
    x, y = med_line("low",  "EoL_1");  _plot_line(x, y, color_low,  "-.")
    x, y = med_line("high", "EoL_2");  _plot_line(x, y, color_high, "-")
    x, y = med_line("high", "EoL_1");  _plot_line(x, y, color_high, "-.")

    ax.set_xlim(years.min(), years.max())
    ax.xaxis.set_major_locator(FixedLocator(list(anchor_years)))
    ax.xaxis.set_major_formatter(FixedFormatter([str(y) for y in anchor_years]))
    ax.tick_params(axis="x", which="major", length=5.5, width=1.0)

    ax.axhline(0, color="k", lw=1.1, alpha=0.9)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.9)
    ax.grid(axis="x", visible=True)

    # y-lims 
    if ylims is not None:
        ax.set_ylim(*ylims)
    else:
        pooled_min = float(imp_long.groupby("year")["impact_MtCO2e"].quantile(q_low).min())
        ax.set_ylim(pooled_min, 0.0)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.tick_params(axis="y", which="major", length=5.5, width=1.0)

    for side in ("top","right","left","bottom"):
        ax.spines[side].set_visible(True); ax.spines[side].set_linewidth(1.2); ax.spines[side].set_color("0.2")
    ax.tick_params(axis="both", which="both", labelsize=18, top=False, right=False)

# ---------------------------
# Build totals 
# ---------------------------
years = sorted(pd.unique(pd.to_numeric(tawp_wide["year"], errors="coerce").dropna()).astype(int))
emis  = (tawp_wide["emission_scenario"] if "emission_scenario" in tawp_wide.columns
         else tawp_wide["adoption"]).astype(str).str.lower().unique().tolist()

CLT_S1 = build_totals_for_projection(
    pnw_df=pnw_CLT_S1, se_df=se_CLT_S1, product="CLT",
    tawp_wide=tawp_wide, years=years, emission_scenarios=emis
)
GLU_S1 = build_totals_for_projection(
    pnw_df=pnw_glt_S1, se_df=se_glt_S1, product="Glulam",
    tawp_wide=tawp_wide, years=years, emission_scenarios=emis
)
COMB_S1 = combine_products_totals(CLT_S1, GLU_S1)

CLT_S2 = build_totals_for_projection(
    pnw_df=pnw_CLT_S2, se_df=se_CLT_S2, product="CLT",
    tawp_wide=tawp_wide, years=years, emission_scenarios=emis
)
GLU_S2 = build_totals_for_projection(
    pnw_df=pnw_glt_S2, se_df=se_glt_S2, product="Glulam",
    tawp_wide=tawp_wide, years=years, emission_scenarios=emis
)
COMB_S2 = combine_products_totals(CLT_S2, GLU_S2)

CLT_S3 = build_totals_for_projection(
    pnw_df=pnw_CLT_S3, se_df=se_CLT_S3, product="CLT",
    tawp_wide=tawp_wide, years=years, emission_scenarios=emis
)
GLU_S3 = build_totals_for_projection(
    pnw_df=pnw_glt_S3, se_df=se_glt_S3, product="Glulam",
    tawp_wide=tawp_wide, years=years, emission_scenarios=emis
)
COMB_S3 = combine_products_totals(CLT_S3, GLU_S3)

def _pooled_min_mt(q=0.05, *dfs):
    chunks = []
    for df in dfs:
        m = df.melt(
            id_vars=["emission_scenario","year","realization"],
            value_vars=["impact_EoL_1_kgCO2e","impact_EoL_2_kgCO2e"],
            var_name="eol", value_name="kg"
        )
        chunks.append(m["kg"] / 1e9)  # kg -> Mt
    ser = pd.concat(chunks, ignore_index=True)
    return float(ser.quantile(q))

# ---------------------------
# Build totals 
# ---------------------------
ylims = (-95, 0.0)
fig, axes = plt.subplots(1, 2, figsize=(11.25, 5), sharey=True, constrained_layout=True, dpi=300)

low_color = "#542788"
high_color = "#E66101"

plot_TAWP_timepanel(axes[0], COMB_S1, "", ylims=ylims,color_low=low_color, color_high=high_color)
plot_TAWP_timepanel(axes[1], COMB_S3, "", ylims=ylims,color_low=low_color, color_high=high_color)


axes[0].set_ylabel("Cohort TAWP$_{100}$ (Mt CO$_2$e)", fontsize=22)
axes[0].yaxis.label.set_size(20)
fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.02)

for i, a in enumerate(axes):
    a.text(0.985, 0.965, chr(ord('A')+i), transform=a.transAxes,
           ha="right", va="top", fontsize=18, fontweight="bold", color="0.15")

from matplotlib.lines import Line2D
h_low_eol2  = Line2D([0],[0], color=low_color,  lw=2.8, ls="-",  label="LEP — Reuse")
h_low_eol1  = Line2D([0],[0], color=low_color,  lw=2.8, ls="-.", label="LEP — Pyrolysis")
h_high_eol2 = Line2D([0],[0], color=high_color,   lw=2.8, ls="-",  label="HEP — Landfill")
h_high_eol1 = Line2D([0],[0], color=high_color,   lw=2.8, ls="-.", label="HEP — Energy")


handles = [h_low_eol2, h_low_eol1, h_high_eol2, h_high_eol1]
leg = fig.legend(handles=handles,
                 loc="lower center", bbox_to_anchor=(0.5, 1.02),
                 ncol=4, frameon=False,
                 fontsize=18,
                 handlelength=2.8, handletextpad=0.8,
                 columnspacing=1.6, labelspacing=0.5, borderaxespad=0.0)
fig.subplots_adjust(top=0.88)  
plt.show()