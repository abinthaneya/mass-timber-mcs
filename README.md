# README — Code (MCS + D-CUBE/TAWP + Post Processing)

Code in support of: **“Missed opportunities forgo one-third of mass timber’s carbon storage potential.”**  
**Authors:** Ahmad Bin Thaneya, Aysegul Petek Gursel, Seth Kane, Elisabeth Van Roijen, Baishakhi Bose, Thomas P. Hendrickson, Corinne D. Scown, Sabbie A. Miller, and Arpad Horvath  
**Code prepared by:** Ahmad Bin Thaneya

---

## MCS_Data_Input
Contains **scenario-aware Monte Carlo simulation (SA-MCS)** parameter inputs used by the inventory model (distributions, bounds, and sampled parameter sets). The parameterization and sampling approach follows Bin Thaneya et al. (see References). Full distribution definitions and parameter values are documented in the **Supplementary Data** (distributions/parameters) and **Supplementary Information** (parameter descriptions) submitted with this study.

---

## MCS_Model
Contains all scripts to run the **SA-MCS life-cycle inventory** and the **TAWP100 (D-CUBE)** analysis.

### 1) Scenarios (regional MCS runs)
Location: `MCS_Model/Scenarios/`

Regional folders:
- `NW/` (Northwest; referred to as **PNW** in some instances)
- `SE/` (Southeast)

Each region folder includes:
- `MCS_main.py` — runs the SA-MCS inventory model for that region (by cohort year, scenario, and product).
- `MCS_backend_fxns.py` — helper functions used by `MCS_main.py`.
- `MCS_Results/` — output directory written by `MCS_main.py`.

**Outputs:** `MCS_main.py` writes life-cycle inventory results (emissions + material flows) to `MCS_Results/`, organized by scenario (e.g., low/high), product (CLT/GLT), and life-cycle stage.

**Note:** The `NW/` folder contains the PNW model subfolders (e.g., `PNW_softwood_CLT`, `PNW_softwood_GLT` / Glulam variants), and `SE/` contains the SE equivalents.

### 2) Biochar inputs
Location: `MCS_Model/Scenarios/Biochar_python/`

Contains biochar-related data generated from a separate biochar model study (see References) and used as an input to this MCS framework.

### 3) D-CUBE / TAWP100
Location: `MCS_Model/Scenarios/<region>/D_CUBE/`

Computes **TAWP100** using the MCS inventory outputs:
- Run: `DCUBE_main_NEW.py`
- Helper functions: `DCUBE_fxns_final_NEW.py`
- Outputs: `TAWP_results/`

---

## Post (post-processing + paper figures)
Location: `MCS_Model/Post/`

Contains analysis scripts that consume the **MCS inventory outputs** and **TAWP/D-CUBE outputs** to reproduce paper figures and summary tables.

Key scripts:
- `Total_TAWP_100.py`, `NW_TAWP_100_m3.py`, `SE_TAWP_100_m3.py`  
  Generates TAWP100 summary plots/tables (**Figure 2**) using `TAWP_results/`.

- `Dynamic_Carbon_Analysis.py`  
  Runs the dynamic carbon/storage analysis and outputs **Figure 3** plus storage summary statistics (uses `MCS_Results` + adoption projections).

- `Lever_Analysis.py`  
  Produces lever decomposition waterfall plots (**Figure 4A–B**) and summary breakdown tables (uses `MCS_Results` + adoption projections).

- `Switch_Timing.py`  
  Produces LEP implementation timing results (**Figure 4C–D**) and summary tables (uses `MCS_Results` + adoption projections).

Shared backend:
- `Post_Backend_Fxns2.py` — common functions used by Post scripts (loading MCS outputs, scaling to projections, plotting, bundling/caching, etc.).

Outputs and saved runs:
- `Post/Results/Projections/` — mass timber adoption/production projection inputs used for scaling.
- `Post/Results/PlotBundles/` — saved “bundle” outputs to avoid recomputing long runs.
- Figures are written to `Post/Results/` (and/or paths specified within scripts).

---

## Script running order

### 1) Run regional MCS inventories
- Run: `MCS_Model/Scenarios/NW/MCS_main.py`
- Run: `MCS_Model/Scenarios/SE/MCS_main.py`
- Confirm outputs appear under:
  - `MCS_Model/Scenarios/<region>/MCS_Results/`

### 2) Run TAWP100
- Run: `DCUBE_main_NEW.py` (in each region’s `D_CUBE/` folder)
- Confirm outputs appear under:
  - `MCS_Model/Scenarios/<region>/D_CUBE/TAWP_results/`

### 3) Reproduce figures (Post)
- Run TAWP scripts (**Figure 2**):
  - `Total_TAWP_100.py`
  - `NW_TAWP_100_m3.py`
  - `SE_TAWP_100_m3.py`
- Run: `Dynamic_Carbon_Analysis.py` (**Figure 3**)
- Run: `Lever_Analysis.py` (**Figure 4A–B**)
- Run: `Switch_Timing.py` (**Figure 4C–D**)

---

## Notes on runtime + caching
Several Post analyses can be slow. Saved outputs are stored in:
- `MCS_Model/Post/Results/PlotBundles/`

If PlotBundles are present, scripts typically load saved runs instead of recomputing them. To trigger a new run, change the output/bundle name used in the script (or delete the saved bundle directory and rerun).

---

## Definitions
- **PNW / NW:** Northwest region (PNW terminology appears in some code paths)
- **SE:** Southeast region
- **CLT / GLT:** cross-laminated timber / glue-laminated timber
- **low / high:** emissions scenario variants used in the MCS inventory
- **S1–S3:** production/adoption projection scenario labels used in Post analyses
- **HEP / LEP:** high-emissions / low-emissions policy cases (per paper definitions)

---

## References
- Bin Thaneya et al. (2024) — SA-MCS methods: https://iopscience.iop.org/article/10.1088/2634-4505/ad40ce  
- Kane et al. (2025) — biochar model source paper: https://iopscience.iop.org/article/10.1088/1748-9326/ad99e9  
