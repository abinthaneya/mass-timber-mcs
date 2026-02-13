###############################################################################
###############################################################################
# Dynamic Plot
###############################################################################
###############################################################################
from pathlib import Path
import os
import Post_Backend_Fxns2 as fx
import pandas as pd

def resolve_code_root():
    env = os.getenv("MCS_CODE_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(f"MCS_CODE_DIR is not a directory: {p}")
        return p

    try:
        here = Path(__file__).resolve()
        start = here.parent
    except NameError:
        start = Path.cwd().resolve()

    for parent in [start] + list(start.parents):
        if parent.name.casefold() == "code":
            return parent.resolve()

    raise FileNotFoundError(f"Could not find 'Code' folder walking up from: {start}")


CODE_ROOT = resolve_code_root()

POST_ROOT = CODE_ROOT / "MCS_Model" / "Post"
SCEN_ROOT = CODE_ROOT / "MCS_Model" / "Scenarios"

PROJ_DIR = POST_ROOT / "Results" / "Projections"
BUNDLE_DIR = POST_ROOT / "Results" / "PlotBundles"
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

# ---- change name if want building from scratch ----
# ---- current name uses the results that show up in the paper ----
BUNDLE_NAME = "cum_net_bundle_v2_new" 
BUNDLE_FUTURE_NAME = f"{BUNDLE_NAME}__futureonly_uptake"

# ---- load existing bundle (or build it if missing) ----
try:
    bundle = fx.load_bundle(folder=str(BUNDLE_DIR), name=BUNDLE_NAME)
except FileNotFoundError:
    bundle = fx.build_cum_plot_bundle(
        str(PROJ_DIR),
        str(SCEN_ROOT),
        include_scenarios=("low", "high"),
        prod_scenarios=("S1", "S3"),
    )
    fx.save_bundle(bundle, folder=str(BUNDLE_DIR), name=BUNDLE_NAME)

# ---- ensure production matrices exist in backend module globals ----
scens = tuple(bundle["meta"].get("prod_scenarios", list(bundle["panels"].keys())))
fx.load_all_projections(str(PROJ_DIR), scenarios=scens, products=("CLT", "GLT"))

# ---- future-only uptake patch ----
cfg = fx.DcubeConfig()
bundle2, checks = fx.extend_bundle_with_future_uptake(
    bundle,
    str(SCEN_ROOT),
    last_real_cohort=2120,
    future_last=2150,
    growth_years=cfg.growth_years,
    growth_model=cfg.growth_model,
    growth_params=cfg.growth_params,
    save_as=None,  # avoid relative-path saves inside the function
)

# ---- save patched bundle to the same absolute bundle directory ----
fx.save_bundle(bundle2, folder=str(BUNDLE_DIR), name=BUNDLE_FUTURE_NAME)

# ---- plot ----
fig, axes = fx.plot_two_panel_from_bundle(
    bundle2,
    cut_year=2120,
    to_units="Gt",
    style="line",
    fig_size=(12, 6),
    dpi=300,
    pdf_width_years=15,
    right_pad_frac=0.2,
    y_major_step=1.0,
    anchor_years=(2000, 2020, 2040, 2060, 2080, 2100, 2120),
    bau_lw=3.5,
    fs_axis_label=19,
    fs_tick=18,
    letter_fs=18,
)
#%%
###############################################################################
###############################################################################
# Dynamic Stats
###############################################################################
###############################################################################
deltas = fx.compute_case_deltas(
    bundle2,
    cut_year=2120,
    percentiles=("median", "p05", "p95"),
    to_units="Mt",
    include_bau=True,
)

pivot = (
    deltas.pivot_table(
        index=["panel", "stat"],
        columns="variant",
        values=["abs_delta", "rel_delta"],
    )
    .sort_index()
)

pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
print(pivot)

