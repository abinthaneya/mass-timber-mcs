# -*- coding: utf-8 -*-
"""
Created on Tue Sep  9 10:21:17 2025

@author: aab8693
"""

from pathlib import Path
import os, tempfile, shutil, time
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import xlwings as xw
from tqdm.auto import tqdm
import sys
import os, queue as _queue
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor

def make_out_df(seq_df,harvest_df,sawmill_df,operation_df,const_dem_df,EoL_df):
    
    out_df = pd.DataFrame()
    ### Prepare for D-CUBE - use D-CUBE designations

    ### Life Cycle Stage: Raw Materials
    out_df['raw_materials_emissions'] = harvest_df.harvest_ops_CO2 + harvest_df.fert_herb_CO2
    out_df['raw_materials_uptake'] = seq_df.Sequestered_CO2
    out_df['raw_materials_CH4'] = 0.0
    out_df['raw_materials_years'] = 21
    
    ### Life Cycle Stage: Transportation 1
    out_df['transportation_1_emissions'] = harvest_df.harvest_haul_CO2
    out_df['transportation_1_uptake'] = 0.0
    out_df['transportation_1_CH4'] = 0.0
    out_df['transportation_1_years'] = 1
    
    ### Life Cycle Stage: Processing
    out_df['processing_bio_emissions'] = sawmill_df.sawmill_biomass_CO2_biogenic
    out_df['processing_fossil_emissions'] = sawmill_df.sawmill_ops_CO2
    
    out_df['processing_emissions'] = sawmill_df.sawmill_biomass_CO2_biogenic + sawmill_df.sawmill_ops_CO2
    out_df['processing_uptake'] = 0.0
    out_df['processing_CH4'] = 0.0
    out_df['processing_years'] = 1
    
    ### Life Cycle Stage: Transportation 2
    out_df['transportation_2_emissions'] = sawmill_df.sawmill_haul_CO2
    out_df['transportation_2_uptake'] = 0.0
    out_df['transportation_2_CH4'] = 0.0
    out_df['transportation_2_years'] = 1
    
    ### Life Cycle Stage: Manufacture
    out_df['operation_emissions'] = operation_df.total_op_process_CO2_fossil 
    out_df['operation_uptake'] = 0.0
    out_df['operation_CH4'] = 0.0
    out_df['operation_years'] = 1
    
    ### Life Cycle Stage: Transportation 3
    out_df['transportation_3_emissions'] = operation_df.op_haul_CO2
    out_df['transportation_3_uptake'] = 0.0
    out_df['transportation_3_CH4'] = 0.0
    out_df['transportation_3_years'] = 1
    
    ### Life Cycle Stage: Construction
    out_df['construction_emissions'] = const_dem_df.construction_CO2
    out_df['construction_uptake'] = 0.0
    out_df['construction_CH4'] = 0.0
    out_df['construction_years'] = 1
    
    ### Life Cycle Stage: Use
    out_df['use_emissions'] = seq_df.Residue_CO2
    out_df['use_uptake'] = 0.0
    out_df['use_CH4'] = 0.0
    out_df['use_years'] = 50
    
    ### Life Cycle Stage: Deconstruction
    out_df['deconstruction_emissions'] = const_dem_df.deconstruction_CO2
    out_df['deconstruction_uptake'] = 0.0
    out_df['deconstruction_CH4'] = 0.0
    out_df['deconstruction_years'] = 1
    
    ### Life Cycle Stage: Transportation 4
    trans_4_scale = 0.2
    out_df['transportation_4_emissions'] = trans_4_scale * operation_df.op_haul_CO2
    out_df['transportation_4_uptake'] = 0.0
    out_df['transportation_4_CH4'] = 0.0
    out_df['transportation_4_years'] = 1
    
    ### Life Cycle Stage: EoL_1
    out_df['EoL_1_bio_emissions'] = EoL_df.EoL_1_bio_CO2
    out_df['EoL_1_fossil_emissions'] = EoL_df.EoL_1_fossil_CO2
    out_df['EoL_1_avoided_fossil_emissions'] = EoL_df.EoL_1_avoided_fossil_CO2
    
    
    out_df['EoL_1_emissions'] = EoL_df.EoL_1_bio_CO2 + EoL_df.EoL_1_fossil_CO2
    out_df['EoL_1_uptake'] = EoL_df.EoL_1_avoided_fossil_CO2
    out_df['EoL_1_CH4'] = 0.0
    out_df['EoL_1_years'] = 1
    
    ### Life Cycle Stage: EoL_2
    out_df['EoL_2_bio_emissions'] = EoL_df.EoL_2_bio_CO2
    out_df['EoL_2_fossil_emissions'] = 0.0
    out_df['EoL_2_avoided_fossil_emissions'] = EoL_df.EoL_2_avoided_fossil_CO2
    
    
    out_df['EoL_2_emissions'] = EoL_df.EoL_2_bio_CO2 + 0
    out_df['EoL_2_uptake'] = EoL_df.EoL_2_avoided_fossil_CO2
    out_df['EoL_2_CH4'] = EoL_df.EoL_2_fossil_CH4
    out_df['EoL_2_years'] = 27  ########################## Might Want to Change
    
    ### Life Cycle Stage: EoL_3
    out_df['EoL_3_bio_emissions'] = EoL_df.EoL_3_bio_CO2
    out_df['EoL_3_fossil_emissions'] = 0.0
    out_df['EoL_3_avoided_fossil_emissions'] = EoL_df.EoL_3_avoided_fossil_CO2
    
    
    out_df['EoL_3_emissions'] = EoL_df.EoL_3_bio_CO2 + 0
    out_df['EoL_3_uptake'] = EoL_df.EoL_3_avoided_fossil_CO2
    out_df['EoL_3_CH4'] = 0.0
    out_df['EoL_3_years'] = 1
    
    return(out_df)


# ---------- CONFIG ----------
FOLDER = Path(r"..\es5c00080_si_002")
BASE_NAME = "CarbonUptakeCalculator_locked"      # expects .xlsm
DEFAULT_SHEET = "Datasheet 2"

# Speed knobs (sane defaults)
FAST_BLOCK_WRITES = True    # batch-write all non-EoL inputs in 3 calls (D/E/G columns)
CALC_MODE = "calculate"     # "none" | "calculate" | "full" | "fullrebuild"

# Row numbers (fixed cells on Datasheet 2)
raw_materials_num = 7
transportation_1_num = 9
processing_num = 11
transportation_2_num = 13
operation_num = 15
transportation_3_num = 17
construction_num = 19
use_num = 21
deconstruction_num = 23
transportation_4_num = 25   # <-- fix retained
EoL_num = 27

# Map out_df columns -> Excel cells (A1). If no "Sheet!", DEFAULT_SHEET is used.
# Note: Any keys that start with "EoL_" in this dict are ignored (EoL handled per-scenario).
INPUT_MAP = {
    # Raw Materials
    "raw_materials_emissions": f"D{raw_materials_num}",
    "raw_materials_uptake":    f"D{raw_materials_num+1}",
    "raw_materials_CH4":       f"E{raw_materials_num}",
    "raw_materials_years":     f"G{raw_materials_num}",

    # Transportation 1
    "transportation_1_emissions": f"D{transportation_1_num}",
    "transportation_1_uptake":    f"D{transportation_1_num+1}",
    "transportation_1_CH4":       f"E{transportation_1_num}",
    "transportation_1_years":     f"G{transportation_1_num}",

    # Processing
    "processing_emissions": f"D{processing_num}",
    "processing_uptake":    f"D{processing_num+1}",
    "processing_CH4":       f"E{processing_num}",
    "processing_years":     f"G{processing_num}",

    # Transportation 2
    "transportation_2_emissions": f"D{transportation_2_num}",
    "transportation_2_uptake":    f"D{transportation_2_num+1}",
    "transportation_2_CH4":       f"E{transportation_2_num}",
    "transportation_2_years":     f"G{transportation_2_num}",

    # Operation
    "operation_emissions": f"D{operation_num}",
    "operation_uptake":    f"D{operation_num+1}",
    "operation_CH4":       f"E{operation_num}",
    "operation_years":     f"G{operation_num}",

    # Transportation 3
    "transportation_3_emissions": f"D{transportation_3_num}",
    "transportation_3_uptake":    f"D{transportation_3_num+1}",
    "transportation_3_CH4":       f"E{transportation_3_num}",
    "transportation_3_years":     f"G{transportation_3_num}",

    # Construction
    "construction_emissions": f"D{construction_num}",
    "construction_uptake":    f"D{construction_num+1}",
    "construction_CH4":       f"E{construction_num}",
    "construction_years":     f"G{construction_num}",

    # Use
    "use_emissions": f"D{use_num}",
    "use_uptake":    f"D{use_num+1}",
    "use_CH4":       f"E{use_num}",
    "use_years":     f"G{use_num}",

    # Deconstruction
    "deconstruction_emissions": f"D{deconstruction_num}",
    "deconstruction_uptake":    f"D{deconstruction_num+1}",
    "deconstruction_CH4":       f"E{deconstruction_num}",
    "deconstruction_years":     f"G{deconstruction_num}",

    # Transportation 4
    "transportation_4_emissions": f"D{transportation_4_num}",
    "transportation_4_uptake":    f"D{transportation_4_num+1}",
    "transportation_4_CH4":       f"E{transportation_4_num}",
    "transportation_4_years":     f"G{transportation_4_num}",

    # --- DO NOT PUT EoL here; it's written per-scenario below ---
}

# EoL target cells (constant on sheet); we reuse these for each scenario
EOL_TARGETS = {
    "emissions": f"D{EoL_num}",
    "uptake":    f"D{EoL_num+1}",
    "CH4":       f"E{EoL_num}",
    "years":     f"G{EoL_num}",
}

# Macros (run every time after writing EoL for a scenario)
MACROS = ["CalculatingImpacts_100yr"]

# Results to pull each time (read after macro for each scenario)
# (We also include a fast block reader for R7:R19; this map remains as a reference.)
RESULT_MAP = {
    "raw_materials":   "R7",
    "transportation_1":"R8",
    "processing":      "R9",
    "transportation_2":"R10",
    "operation":       "R11",
    "transportation_3":"R12",
    "construction":    "R13",
    "use":             "R14",
    "deconstruction":  "R15",
    "transportation_4":"R16",
    "EoL":             "R17",
    "TAWP":            "R18",
    "GWP":             "R19",
}

# Optional "done" flag in the workbook
STATUS_CELL = None        # e.g., "Z99"
STATUS_EXPECTED = "OK"
STATUS_TIMEOUT_S = 180
# ---------- END CONFIG ----------


# ---------- helpers ----------
try:
    from tqdm import tqdm as _tqdm_base
except Exception:
    from tqdm.auto import tqdm as _tqdm_base
    


def _TQDM(iterable, total=None, desc=None, leave=True):
    """
    Progress bar that updates reliably in Spyder/QtConsole.
    Forces stdout, fixed width, and frequent refresh.
    """
    return _tqdm_base(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        file=sys.stdout,          # <— key: Spyder flushes stdout more reliably than stderr
        dynamic_ncols=False,      # avoid auto-resizing glitches
        ncols=80,                 # fixed width; tweak if desired
        mininterval=0.2,          # refresh cadence
        miniters=1,               # update every iteration
        smoothing=0
    )

# If you already added _TQDM earlier for Spyder, we'll try to use it; else fallback to tqdm.
try:
    _PBAR = _TQDM  # defined earlier in your file
except NameError:
    def _PBAR(iterable, total=None, desc=None, leave=True):
        return tqdm(iterable, total=total, desc=desc, leave=leave)

def _find_workbook(folder: Path, base: str) -> Path:
    for ext in (".xlsm", ".xlsb", ".xlsx"):
        p = folder / f"{base}{ext}"
        if p.exists(): return p
    raise FileNotFoundError(f"Couldn't find {base}.* in {folder.resolve()}")
    
def _open_copy_with_excel(src_path: Path):
    tmpdir = Path(tempfile.mkdtemp(prefix="xl_run_"))
    copy_path = tmpdir / src_path.name
    shutil.copy2(src_path, copy_path)

    app = xw.App(visible=False, add_book=False)
    # Quiet + faster
    app.display_alerts = False
    app.screen_updating = False
    try:
        app.api.EnableEvents = False
        app.api.DisplayStatusBar = False
        from xlwings.constants import Calculation
        app.api.Calculation = Calculation.xlCalculationManual
    except Exception:
        pass

    wb = app.books.open(str(copy_path), read_only=False, update_links=False)
    try:
        wb.api.EnableAutoRecover = False
    except Exception:
        pass
    return app, wb, tmpdir

def _close_excel(app: xw.App | None, wb: xw.Book | None):
    try:
        if wb: wb.close()
    finally:
        if app:
            try:
                app.display_alerts = True
                app.screen_updating = True
                app.api.EnableEvents = True
                app.api.DisplayStatusBar = True
            except Exception:
                pass
            app.quit()

def _resolve_range(wb: xw.Book, target: str) -> xw.Range:
    if "!" in target:
        sheet, a1 = target.split("!", 1)
        return wb.sheets[sheet].range(a1)
    # try named range first
    try:
        return wb.names[target].refers_to_range
    except Exception:
        return wb.sheets[DEFAULT_SHEET].range(target)

def _to_excel_value(v):
    if v is None or (isinstance(v, float) and np.isnan(v)) or pd.isna(v): return None
    if isinstance(v, (np.floating,)): return float(v)
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (bool, int, float, str)): return v
    return str(v)

def _write_map(wb: xw.Book, series: pd.Series, mapping: dict[str, str]):
    for key, target in mapping.items():
        rng = _resolve_range(wb, target)
        rng.value = _to_excel_value(series[key])

# Predefine the stage base rows (excluding EoL)
STAGE_ROWS = [
    ("raw_materials",   raw_materials_num),
    ("transportation_1",transportation_1_num),
    ("processing",      processing_num),
    ("transportation_2",transportation_2_num),
    ("operation",       operation_num),
    ("transportation_3",transportation_3_num),
    ("construction",    construction_num),
    ("use",             use_num),
    ("deconstruction",  deconstruction_num),
    ("transportation_4",transportation_4_num),
]

def _write_non_eol_block_fast(wb: xw.Book, row: pd.Series):
    """Batch-write all non-EoL inputs in 3 big writes: D, E, and G columns."""
    sh = wb.sheets[DEFAULT_SHEET]
    start = min(r for _, r in STAGE_ROWS)         # 7
    end   = max(r for _, r in STAGE_ROWS) + 1     # include uptake row of last stage (e.g., 26)

    d_rng = sh.range(f"D{start}:D{end}").options(ndim=2)
    e_rng = sh.range(f"E{start}:E{end}").options(ndim=2)
    g_rng = sh.range(f"G{start}:G{end}").options(ndim=2)

    d_vals = d_rng.value  # (rows,1)
    e_vals = e_rng.value
    g_vals = g_rng.value

    for stage, r in STAGE_ROWS:
        off = r - start
        d_vals[off][0]     = _to_excel_value(row[f"{stage}_emissions"])
        d_vals[off + 1][0] = _to_excel_value(row[f"{stage}_uptake"])
        e_vals[off][0]     = _to_excel_value(row[f"{stage}_CH4"])
        g_vals[off][0]     = _to_excel_value(row[f"{stage}_years"])

    d_rng.value = d_vals
    e_rng.value = e_vals
    g_rng.value = g_vals

def _write_non_eol_inputs(wb: xw.Book, row: pd.Series):
    if FAST_BLOCK_WRITES:
        _write_non_eol_block_fast(wb, row)
    else:
        base_map = {k: v for k, v in INPUT_MAP.items() if not k.lower().startswith("eol_")}
        _write_map(wb, row, base_map)

def _write_eol_fast(wb: xw.Book, emissions, uptake, ch4, years):
    """Write EoL values with minimal COM calls."""
    sh = wb.sheets[DEFAULT_SHEET]
    sh.range(f"D{EoL_num}:D{EoL_num+1}").value = [[_to_excel_value(emissions)], [_to_excel_value(uptake)]]
    sh.range(f"E{EoL_num}").value = _to_excel_value(ch4)
    sh.range(f"G{EoL_num}").value = _to_excel_value(years)

def _run_macros_and_calc(wb: xw.Book, macros: list[str], calc_mode: str = CALC_MODE):
    for m in macros or []:
        try:
            wb.macro(m)()
        except Exception:
            wb.app.api.Run(f"'{wb.name}'!{m}")

    mode = (calc_mode or "none").lower()
    try:
        if mode == "fullrebuild":
            wb.app.api.CalculateFullRebuild()
        elif mode == "full":
            wb.app.api.CalculateFull()
        elif mode == "calculate":
            wb.app.api.Calculate()
        else:
            pass
    except Exception:
        # Fall back gracefully
        try:
            wb.app.api.CalculateFull()
        except Exception:
            try:
                wb.app.api.Calculate()
            except Exception:
                pass

def _wait_for_status(wb: xw.Book, cell: str, expected, timeout_s: int):
    if not cell: return True
    rng = _resolve_range(wb, cell)
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        val = rng.value
        if (callable(expected) and expected(val)) or (not callable(expected) and val == expected):
            return True
        time.sleep(0.25)
    raise TimeoutError(f"Status cell '{cell}' did not reach expected value within {timeout_s}s (last: {rng.value})")

# Fast results reader for contiguous R7:R19 band
_RESULT_KEYS_IN_ORDER = [s for s, _ in STAGE_ROWS] + ["EoL", "TAWP", "GWP"]
_RESULT_START, _RESULT_END, _RESULT_COL = 7, 19, "R"

def _read_results_fast(wb: xw.Book) -> dict:
    sh = wb.sheets[DEFAULT_SHEET]
    rng = sh.range(f"{_RESULT_COL}{_RESULT_START}:{_RESULT_COL}{_RESULT_END}").options(ndim=2)
    vals2d = rng.value  # list of [val]
    flat = [row[0] if isinstance(row, list) and row else row for row in vals2d]
    out = {}
    for i, key in enumerate(_RESULT_KEYS_IN_ORDER):
        out[key] = flat[i] if i < len(flat) else None
    return out

def _normalize_eol_list(eol_scenarios) -> list[str]:
    if isinstance(eol_scenarios, (int, np.integer, str)):
        eol_scenarios = [eol_scenarios]
    normed = []
    for s in eol_scenarios:
        if isinstance(s, (int, np.integer)):
            normed.append(f"EoL_{int(s)}")
        else:
            s = str(s).strip()
            normed.append(s if s.lower().startswith("eol_") else f"EoL_{s}")
    return normed

def _normalize_rows(out_df: pd.DataFrame, rows=None):
    if rows is None:
        return list(range(len(out_df)))
    if isinstance(rows, int):
        return list(range(min(rows, len(out_df))))
    if isinstance(rows, slice):
        return list(range(*rows.indices(len(out_df))))
    return list(rows)    


import re

def _detect_eol_prefixes(cols) -> list[str]:
    # finds EoL_N that have the required *_emissions input
    pref = sorted(
        {m.group(1) for c in cols if (m := re.match(r'^(EoL_\d+)_emissions$', c))},
        key=lambda s: int(s.split('_')[1])
    )
    return pref



def run_many_realizations_multi_eol(
    out_df: pd.DataFrame,
    rows=None,
    eol_scenarios="auto",          # <-- changed default
    return_style: str = "wide",
    on_error: str = "continue",
    use_tqdm: bool = True,
    tqdm_desc: str = "Realizations",
    tqdm_leave: bool = True
) -> pd.DataFrame:
    """
    Loops through selected realizations, runs chosen EoL scenario(s) per row, and returns one DataFrame.
    Keeps a single Excel instance open for speed. Uses fast block writes & reads.
    """
    src = _find_workbook(FOLDER, BASE_NAME)
    app = wb = tmpdir = None

    row_indexers = _normalize_rows(out_df, rows)
    results_records = []

    try:
        app, wb, tmpdir = _open_copy_with_excel(src)

        # >>> Spyder-friendly tqdm here <<<
        iterator = row_indexers
        if use_tqdm and row_indexers:
            try:
                iterator = _TQDM(row_indexers, total=len(row_indexers),
                                 desc=tqdm_desc, leave=tqdm_leave)
            except Exception:
                iterator = row_indexers

        scenarios = _normalize_eol_list(eol_scenarios)
        if eol_scenarios in (None, "auto"):
            scenarios = _detect_eol_prefixes(out_df.columns)   # e.g., ['EoL_1','EoL_2']
            if len(scenarios) == 0:
                raise ValueError("No EoL_* scenarios detected in out_df columns.")
            else:
                scenarios = _normalize_eol_list(eol_scenarios)

        for i, idx in enumerate(iterator):
            try:
                # Resolve row by label or position
                if isinstance(idx, (int, np.integer)) and idx not in out_df.index:
                    row = out_df.iloc[int(idx)]
                else:
                    try:
                        row = out_df.loc[idx]
                    except Exception:
                        row = out_df.iloc[int(idx)]
                rid = row.name  # keep original index label

                # 1) Write all non-EoL inputs fast
                _write_non_eol_inputs(wb, row)

                # 2) For each selected EoL scenario
                if return_style.lower() == "long":
                    per_row_records = []
                else:
                    row_flat = {}

                for s in scenarios:
                    # Ensure required EoL fields exist
                    for field in ("emissions", "uptake", "CH4", "years"):
                        key = f"{s}_{field}"
                        if key not in row.index:
                            raise KeyError(f"Row {rid}: missing EoL input {key}")

                    # Write EoL values fast
                    _write_eol_fast(
                        wb,
                        emissions=row[f"{s}_emissions"],
                        uptake=row[f"{s}_uptake"],
                        ch4=row[f"{s}_CH4"],
                        years=row[f"{s}_years"],
                    )

                    # Run macro + calc
                    _run_macros_and_calc(wb, MACROS, CALC_MODE)
                    if STATUS_CELL:
                        _wait_for_status(wb, STATUS_CELL, STATUS_EXPECTED, STATUS_TIMEOUT_S)

                    # Read results fast
                    res = _read_results_fast(wb)

                    if return_style.lower() == "long":
                        rec = {"realization": rid, "scenario": s, **res}
                        per_row_records.append(rec)
                    else:
                        for k, v in res.items():
                            row_flat[f"{k}__{s}"] = v

                if return_style.lower() == "long":
                    results_records.extend(per_row_records)
                else:
                    row_flat["realization"] = rid
                    results_records.append(row_flat)

                # Force visible updates in some Spyder setups
                if use_tqdm and hasattr(iterator, "refresh") and (i % 5 == 0):
                    iterator.refresh()

            except Exception as e:
                # Print without breaking the bar
                if use_tqdm and hasattr(iterator, "write"):
                    iterator.write(f"[ERROR] realization={idx}: {e}")
                else:
                    print(f"[ERROR] realization={idx}: {e}")
                if on_error == "raise":
                    raise
                if return_style.lower() != "long":
                    results_records.append({"realization": idx})

        # Build output DataFrame
        if return_style.lower() == "long":
            out = pd.DataFrame.from_records(results_records)
            if not out.empty:
                out = out.set_index(["realization", "scenario"]).sort_index()
            return out
        else:
            out = pd.DataFrame.from_records(results_records).set_index("realization")
            return out

    finally:
        _close_excel(app, wb)
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)