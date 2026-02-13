README — Code Folder (MCS + D‑CUBE/TAWP + Post Processing)
=========================================================
Code in support of: Missed opportunities forgo one-third of mass timber’s carbon storage potential
By: Ahmad Bin Thaneya1, Aysegul Petek Gursel, Seth Kane, Elisabeth Van Roijen, Baishakhi Bose, Thomas P. Hendrickson,
Corinne D. Scown, Sabbie A. Miller, and Arpad Horvath
Code Prepared By: Ahmad Bin Thaneya  

MCS_Data_Input
--------------
Contains the scenario-aware Monte Carlo simulation (SA‑MCS) parameter 
inputs used by the SA-MCSinventory model (distributions, bounds, and sampled parameter sets). 
Theparameterization and sampling approach follows Bin Thaneya et al. (see link
below). Full distribution definitions and parameter values are documented in
the Supplementary Data (distributions/parameters) and Supplementary Information
(parameter descriptions) submitted with this study.


MCS_Model
---------
Contains all scripts to run the SA-MCs life‑cycle
inventory and the TAWP100 (D‑CUBE) analysis.

1) Scenarios (regional MCS runs)
   Location: MCS_Model/Scenarios/

   Regional folders:
     - NW/  (Northwest / “NW” in the paper - also referred to as PNW in some instances)
     - SE/  (Southeast / “SE” in the paper)

   Each region folder includes:
     - MCS_main.py
         Runs the SA‑MCs inventory model for that region (by cohort year,
         scenario, and product).
     - MCS_backend_fxns.py
         Helper functions used by MCS_main.py.
     - MCS_Results/
         Output directory filled by MCS_main.py.

   Outputs:
     MCS_main.py writes life‑cycle inventory results (emissions + material flows)
     to MCS_Results/, organized by scenario (e.g., low/high), product (CLT/GLT),
     and life‑cycle stage.

   Note:
     The NW folder contains the NW model folders (e.g., PNW_softwood_CLT,
     PNW_softwood_GLT/Glulam variants), and SE contains the SE equivalents.


2) Biochar inputs
   Location: MCS_Model/Scenarios/Biochar_python/

   Contains biochar-related data generated from a separate biochar model study
   (see link below) and used as an input to this MCS framework.


3) D‑CUBE / TAWP100
   Location: MCS_Model/Scenarios/<region>/D_CUBE/  

   Computes TAWP100 using the MCS inventory outputs.
     - Run: DCUBE_main_NEW.py
     - Uses helper functions from: DCUBE_fxns_final_NEW.py
     - Writes results to: TAWP_results/


Post (post‑processing + paper figures)
--------------------------------------
Location: MCS_Model/Post/

Contains analysis scripts that consume the MCS inventory outputs and the
TAWP/D‑CUBE outputs to reproduce paper figures and summary tables.

Key scripts:
  - Total_TAWP_100.py
  - NW_TAWP_100_m3.py
  - SE_TAWP_100_m3.py
      Generates TAWP100 summary plots/tables (Figure 2 in the paper) using
      TAWP_results outputs.

  - Dynamic_Carbon_Analysis.py
      Runs the dynamic carbon/storage analysis and outputs Figure 3 plus
      storage summary statistics (uses MCS_Results data + adoption projections).

  - Lever_Analysis.py
      Produces lever decomposition waterfall plots (Figure 4A–B) and summary
      breakdown tables (uses MCS_Results + adoption projections).

  - Switch_timing.py
      Produces LEP implementation timing results (Figure 4C–D) and summary
      tables (uses MCS_Results + adoption projections).

Shared backend:
  - Post_Backend_Fxns2.py
      Common functions used by all Post scripts (loading MCS outputs, scaling to
      production projections, plotting, bundling/caching results, etc.).

Outputs and caching:
  - Post/Results/Projections/
      Mass timber adoption/production projection inputs used for scaling.

  - Post/Results/PlotBundles/
      Cached “bundle” outputs to avoid recomputing long runs. The scripts can be
      rerun to regenerate these bundles, but they may take time.

  - Figures are written to Post/Results/ (and/or to paths specified in scripts).


References
----------
Bin Thaneya et al. 2024 (SA‑MCS methods):
  https://iopscience.iop.org/article/10.1088/2634-4505/ad40ce

Kane et al. (2024) (Biochar model source paper):
  https://iopscience.iop.org/article/10.1088/1748-9326/ad99e9


Script running order 
---------------------

1) Run regional MCS inventories
   - Run: MCS_Model/Scenarios/NW/MCS_main.py
   - Run: MCS_Model/Scenarios/SE/MCS_main.py
   - Confirm outputs appear under each region’s:
       MCS_Model/Scenarios/<region>/MCS_Results/

2) Run TAWP100
   - Run: DCUBE_main_NEW.py (in each region’s D_CUBE/ folder)
   - Confirm outputs appear under:
       MCS_Model/Scenarios/<region>/D_CUBE/TAWP_results/

3) Reproduce figures (Post)
   - Run the TAWP scripts (Figure 2):
       Total_TAWP_100.py
       NW_TAWP_100_m3.py
       SE_TAWP_100_m3.py
   - Run: Dynamic_Carbon_Analysis.py (Figure 3)
   - Run: Lever_Analysis.py (Figure 4A–B)
   - Run: Switch_Timing.py (Figure 4C–D)


Notes on runtime + caching
--------------------------
Several Post analyses can be slow. Cached outputs are stored in:
- MCS_Model/Post/Results/PlotBundles/

If PlotBundles are present, scripts typically load cached arrays instead of recomputing them.
Change the name of the run in the script to trigger a new run. 


Definitions 
------------
- PNW / NW: Northwest region
- SE: Southeast region
- CLT / GLT: cross-laminated timber / glued-laminated timber
- low / high: emissions scenario variants used in the MCS inventory
- S1–S3: production/adoption projection scenario labels used in Post analyses
- HEP / LEP: high-emissions / low-emissions policy cases (per paper definitions)
