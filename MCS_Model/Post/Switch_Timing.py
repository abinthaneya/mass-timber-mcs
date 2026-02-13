from __future__ import annotations

###############################################################################
# Switch Timing Analysis 
###############################################################################

from pathlib import Path
import os

import Post_Backend_Fxns2 as fx


def resolve_code_root() -> Path:
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


# -----------------------------------------------------------------------------
# Paths 
# -----------------------------------------------------------------------------
code_root = resolve_code_root()

post_root = code_root / "MCS_Model" / "Post"
scen_root = code_root / "MCS_Model" / "Scenarios"

PROJ_BASE = (post_root / "Results" / "Projections").as_posix()
MAIN_PATH = scen_root.as_posix()

BUNDLE_DIR = (post_root / "Results" / "PlotBundles")
BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------
switch_years = [2025, 2050, 2075, 2100]
cut_year = 2120

bundle_name = "switch_follow_lep_cum_net_bundle_v2_low"
patched_name = "switch_follow__futureonly_uptake"

# -----------------------------------------------------------------------------
# Build or load base switch bundle
# -----------------------------------------------------------------------------
meta_path = BUNDLE_DIR / f"{bundle_name}__meta.json"
if meta_path.exists():
    switch_follow = fx.load_switch_bundle(folder=str(BUNDLE_DIR), name=bundle_name)
else:
    switch_follow = fx.build_switch_paths_fast(
        PROJ_BASE,
        MAIN_PATH,
        switch_years=switch_years,
        cut_year=cut_year,
        prod_scenarios=("S1", "S3"),
        include_regions=("PNW", "SE"),
        eol_policy="follow_lep",
        hep_bau_weights={("high", 2): 0.70, ("high", 1): 0.30},
        lep_weights={("low", 2): 0.50, ("low", 1): 0.50},
        gwp100_ch4=28.0,
        include_baselines=True,
    )
    # fx.save_switch_bundle(switch_follow, folder=str(BUNDLE_DIR), name=bundle_name)

# -----------------------------------------------------------------------------
# Patch with future cohorts uptake (2121–2150), truncated at 2120
# -----------------------------------------------------------------------------
cfg = fx.DcubeConfig()

patched_meta_path = BUNDLE_DIR / f"{patched_name}__meta.json"
if patched_meta_path.exists():
    switch_follow_patched = fx.load_switch_bundle(folder=str(BUNDLE_DIR), name=patched_name)
    checks_follow = None
else:
    switch_follow_patched, checks_follow = fx.extend_switch_bundle_with_future_uptake(
        switch_follow,
        MAIN_PATH,
        PROJ_BASE=PROJ_BASE,
        last_real_cohort=2120,
        future_last=2150,
        growth_years=cfg.growth_years,
        growth_model=cfg.growth_model,
        growth_params=cfg.growth_params,
        save_as=patched_name,
        save_folder=str(BUNDLE_DIR),
    )

    if checks_follow is not None and not checks_follow.empty:
        print(checks_follow.tail())

# -----------------------------------------------------------------------------
# Plot: forgone storage relative to baseline switch year 2025
# -----------------------------------------------------------------------------
fig_path = (post_root / "Results" / "fig_switch_forgone_storage.png").as_posix()

fx.plot_switch_storage_bars_from_summaries(
    bundle_for_median=switch_follow_patched,  # bars
    bundle_for_p05=switch_follow_patched,     # lower whisker
    bundle_for_p95=switch_follow_patched,     # upper whisker
    panels=("S1", "S3"),
    baseline_year=2025,
    fig_size=(12, 6),
    dpi=600,
    bar_color="#2A9D8F",
    letters=("C", "D"),
    letter_xy=(0.98, 0.96),
    save_path=fig_path,
)
