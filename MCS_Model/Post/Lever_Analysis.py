from __future__ import annotations

# ###############################################################################
# ###############################################################################
# # Lever Analysis Plot
# ###############################################################################
# ###############################################################################
from pathlib import Path
import os

import pandas as pd
import Post_Backend_Fxns2 as fx


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


# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
code_root = resolve_code_root()

post_root = code_root / "MCS_Model" / "Post"
scen_root = code_root / "MCS_Model" / "Scenarios"

proj_base = (post_root / "Results" / "Projections").as_posix()
main_path = scen_root.as_posix()

bundle_folder = (post_root / "Results" / "PlotBundles").as_posix()
bundle_name = "lever_bundle_v3_new"


# -----------------------------------------------------------------------------
# Load (cached) or build + save
# -----------------------------------------------------------------------------
meta_path = Path(bundle_folder) / f"{bundle_name}__meta.json"

if meta_path.exists():
    lever_bundle = fx.load_lever_bundle(folder=bundle_folder, name=bundle_name)
else:
    lever_bundle = fx.build_lever_bundle(
        proj_base,
        main_path,
        prod_scenarios=("S1", "S3"),
        include_regions=("PNW", "SE"),
        include_scenarios=("low", "high"),
        cohort_end=2120,
        gwp100_ch4=28.0,
    )
    # fx.save_lever_bundle(lever_bundle, folder=bundle_folder, name=bundle_name)


# -----------------------------------------------------------------------------
# Resolve Uptake
# -----------------------------------------------------------------------------
lever_bundle, max_delta = fx.patch_process_logistics_in_bundle(lever_bundle)
if max_delta > 0:
    print(f"Patched process_logistics (max |delta| = {max_delta:.6g} t)")
    fx.save_lever_bundle(lever_bundle, folder=bundle_folder, name=bundle_name)


# -----------------------------------------------------------------------------
# Quick summary
# -----------------------------------------------------------------------------
summ = fx.summarize_lever_bundle(lever_bundle, cut_year=2120, to_units="Mt")
print(summ.head())

# -----------------------------------------------------------------------------
# Extend with future cohorts’ uptake only (2121–2150), truncated at 2120
# -----------------------------------------------------------------------------
cfg = fx.DcubeConfig()

lever_bundle2, lever_checks = fx.extend_lever_bundle_with_future_uptake(
    lever_bundle,
    MAIN_PATH=main_path,
    PROJ_BASE=proj_base,
    last_real_cohort=2120,
    future_last=2150,
    growth_years=cfg.growth_years,
    growth_model=cfg.growth_model,
    growth_params=cfg.growth_params,
    save_as="lever_bundle__futureonly_uptake",
    save_folder=bundle_folder,
)

print(lever_checks.tail())


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig_path = (post_root / "Results" / "fig_waterfall_pub.png").as_posix()

fx.plot_two_levers_waterfall_pub(
    lever_bundle2,
    panels=("S1", "S3"),
    cut_year=2120,
    to_units="Gt",
    fig_size=(12, 6),
    dpi=600,
    panel_letters=("A", "B"),
    en_yoff_frac=-0.035,
    bc_yoff_frac=-0.015,
    pl_yoff_frac=+0.005,
    mr_yoff_frac=+0.012,
)


# -----------------------------------------------------------------------------
# Deltas vs reference (from lever arrays only) - compare against dynamics
# -----------------------------------------------------------------------------
lever_deltas = fx.compute_case_deltas_from_levers(
    lever_bundle2,
    cut_year=2120,
    percentiles=("median", "p05", "p95"),
    to_units="Mt",
    variants=[("low", 1), ("high", 2), ("high", 1)],  # biochar, landfill, combustion
    reference=("low", 2),  # reuse
    include_bau=True,
    bau_weights={("high", 2): 0.7, ("high", 1): 0.3},
)

print(lever_deltas.head())


# -----------------------------------------------------------------------------
# Breakdown tables
# -----------------------------------------------------------------------------
breakdown_2120 = fx.build_lever_breakdown_table(
    lever_bundle2, panels=("S1", "S3"), cut_year=2120, to_units="Gt", round_to=3
)
breakdown_2500 = fx.build_lever_breakdown_table(
    lever_bundle2, panels=("S1", "S3"), cut_year=2500, to_units="Gt", round_to=3
)

totals = breakdown_2120[breakdown_2120["component"].isin(["total_release", "uptake", "net_storage"])]

pivot_totals = (
    totals.pivot_table(
        index=["panel", "group", "eol_case"],
        values=["median", "p05", "p95"],
        columns="component",
    )
    .sort_index()
)

print(pivot_totals)

