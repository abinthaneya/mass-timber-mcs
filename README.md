\# README — Code (MCS + D-CUBE/TAWP + Post Processing)



Code in support of: \*\*“Missed opportunities forgo one-third of mass timber’s carbon storage potential.”\*\*  

Authors: Ahmad Bin Thaneya, Aysegul Petek Gursel, Seth Kane, Elisabeth Van Roijen, Baishakhi Bose, Thomas P. Hendrickson, Corinne D. Scown, Sabbie A. Miller, and Arpad Horvath  

Code prepared by: Ahmad Bin Thaneya



---



\## MCS\_Data\_Input

Contains \*\*scenario-aware Monte Carlo simulation (SA-MCS)\*\* parameter inputs used by the inventory model (distributions, bounds, and sampled parameter sets). The parameterization and sampling approach follows Bin Thaneya et al. (see References). Full distribution definitions and parameter values are documented in the \*\*Supplementary Data\*\* (distributions/parameters) and \*\*Supplementary Information\*\* (parameter descriptions) submitted with this study.



---



\## MCS\_Model

Contains all scripts to run the \*\*SA-MCS life-cycle inventory\*\* and the \*\*TAWP100 (D-CUBE)\*\* analysis.



\### 1) Scenarios (regional MCS runs)

Location: `MCS\_Model/Scenarios/`



Regional folders:

\- `NW/` (Northwest; referred to as \*\*PNW\*\* in some instances)

\- `SE/` (Southeast)



Each region folder includes:

\- `MCS\_main.py` — runs the SA-MCS inventory model for that region (by cohort year, scenario, and product).

\- `MCS\_backend\_fxns.py` — helper functions used by `MCS\_main.py`.

\- `MCS\_Results/` — output directory written by `MCS\_main.py`.



\*\*Outputs:\*\* `MCS\_main.py` writes life-cycle inventory results (emissions + material flows) to `MCS\_Results/`, organized by scenario (e.g., low/high), product (CLT/GLT), and life-cycle stage.



\*\*Note:\*\* The `NW/` folder contains the PNW model subfolders (e.g., `PNW\_softwood\_CLT`, `PNW\_softwood\_GLT` / Glulam variants), and `SE/` contains the SE equivalents.



\### 2) Biochar inputs

Location: `MCS\_Model/Scenarios/Biochar\_python/`



Contains biochar-related data generated from a separate biochar model study (see References) and used as an input to this MCS framework.



\### 3) D-CUBE / TAWP100

Location: `MCS\_Model/Scenarios/<region>/D\_CUBE/`



Computes \*\*TAWP100\*\* using the MCS inventory outputs:

\- Run: `DCUBE\_main\_NEW.py`

\- Helper functions: `DCUBE\_fxns\_final\_NEW.py`

\- Outputs: `TAWP\_results/`



---



\## Post (post-processing + paper figures)

Location: `MCS\_Model/Post/`



Contains analysis scripts that consume the \*\*MCS inventory outputs\*\* and \*\*TAWP/D-CUBE outputs\*\* to reproduce paper figures and summary tables.



Key scripts:

\- `Total\_TAWP\_100.py`, `NW\_TAWP\_100\_m3.py`, `SE\_TAWP\_100\_m3.py`  

&nbsp; Generates TAWP100 summary plots/tables (\*\*Figure 2\*\*) using `TAWP\_results/`.



\- `Dynamic\_Carbon\_Analysis.py`  

&nbsp; Runs the dynamic carbon/storage analysis and outputs \*\*Figure 3\*\* plus storage summary statistics (uses `MCS\_Results` + adoption projections).



\- `Lever\_Analysis.py`  

&nbsp; Produces lever decomposition waterfall plots (\*\*Figure 4A–B\*\*) and summary breakdown tables (uses `MCS\_Results` + adoption projections).



\- `Switch\_Timing.py`  

&nbsp; Produces LEP implementation timing results (\*\*Figure 4C–D\*\*) and summary tables (uses `MCS\_Results` + adoption projections).



Shared backend:

\- `Post\_Backend\_Fxns2.py` — common functions used by Post scripts (loading MCS outputs, scaling to projections, plotting, bundling/caching, etc.).



Outputs and caching:

\- `Post/Results/Projections/` — mass timber adoption/production projection inputs used for scaling.

\- `Post/Results/PlotBundles/` — cached “bundle” outputs to avoid recomputing long runs.

\- Figures are written to `Post/Results/` (and/or paths specified within scripts).



---



\## Script running order



\### 1) Run regional MCS inventories

\- Run: `MCS\_Model/Scenarios/NW/MCS\_main.py`

\- Run: `MCS\_Model/Scenarios/SE/MCS\_main.py`

\- Confirm outputs appear under:

&nbsp; - `MCS\_Model/Scenarios/<region>/MCS\_Results/`



\### 2) Run TAWP100

\- Run: `DCUBE\_main\_NEW.py` (in each region’s `D\_CUBE/` folder)

\- Confirm outputs appear under:

&nbsp; - `MCS\_Model/Scenarios/<region>/D\_CUBE/TAWP\_results/`



\### 3) Reproduce figures (Post)

\- Run TAWP scripts (\*\*Figure 2\*\*):

&nbsp; - `Total\_TAWP\_100.py`

&nbsp; - `NW\_TAWP\_100\_m3.py`

&nbsp; - `SE\_TAWP\_100\_m3.py`

\- Run: `Dynamic\_Carbon\_Analysis.py` (\*\*Figure 3\*\*)

\- Run: `Lever\_Analysis.py` (\*\*Figure 4A–B\*\*)

\- Run: `Switch\_Timing.py` (\*\*Figure 4C–D\*\*)



---



\## Notes on runtime + caching

Several Post analyses can be slow. Cached outputs are stored in:

\- `MCS\_Model/Post/Results/PlotBundles/`



If PlotBundles are present, scripts typically load cached arrays instead of recomputing them. To trigger a new run, change the output/bundle name used in the script (or delete the cached bundle directory and rerun).



---



\## Definitions (quick)

\- \*\*PNW / NW:\*\* Northwest region (PNW terminology appears in some code paths)

\- \*\*SE:\*\* Southeast region

\- \*\*CLT / GLT:\*\* cross-laminated timber / glued-laminated timber

\- \*\*low / high:\*\* emissions scenario variants used in the MCS inventory

\- \*\*S1–S3:\*\* production/adoption projection scenario labels used in Post analyses

\- \*\*HEP / LEP:\*\* high-emissions / low-emissions policy cases (per paper definitions)



---



\## References

\- Bin Thaneya et al. (2024) — SA-MCS methods: https://iopscience.iop.org/article/10.1088/2634-4505/ad40ce

\- Kane et al. (2024) — biochar model source paper: https://iopscience.iop.org/article/10.1088/1748-9326/ad99e9





