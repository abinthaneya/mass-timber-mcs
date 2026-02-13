################################### SA-MCs Backend Fxns
from __future__ import annotations
import pandas as pd
import numpy as np
import os
import re
import math
from pathlib import Path

################################### Data Reading Fxns
REGIONS = {"PNW", "SE", "NE"}

def _resolve_data_root() -> Path:
    env = os.getenv("MCS_DATA_INPUT_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise FileNotFoundError(
                f"MCS_DATA_INPUT_DIR was set but is not a directory: {p}"
            )
        return p

    try:
        here = Path(__file__).resolve()
    except NameError:
        # e.g., running in a notebook / interactive session
        here = Path.cwd().resolve()

    for parent in [here] + list(here.parents):
        candidate = parent / "MCS_Data_Input"
        if candidate.is_dir():
            return candidate.resolve()

    raise FileNotFoundError()

DATA_ROOT = _resolve_data_root()

def read_input_csv(*relative_parts: str, **kwargs):
    path = DATA_ROOT.joinpath(*relative_parts)
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    return pd.read_csv(path, **kwargs)

def normalize_attribute(col_lower, tokens):
    tset = set(tokens)
    # Standardize Columns - check for any misspellings 
    if "green" in tset and "wood" in tset and "mass" in tset and ("req" in tset or "requirement" in tset):
        return "green_wood_mass_req"
    if "green_wood_mass_req" in col_lower:
        return "green_wood_mass_req"
    if "loss" in tset and "efficiency" in tset:
        return "loss_efficiency"
    if "mass" in tset and ("req" in tset or "requirement" in tset):
        return "mass_req_data"
    # Wood data catch-alls
    if "wood" in tset and "data" in tset:
        return "wood_data"
    if re.search(r"(?:^|_)wood(?:_|$)", col_lower):
        return "wood_data"
    if re.search(r"(?:^|_)data(?:_|$)", col_lower):
        return "wood_data"
    return "wood_data"
def standardize_col(col: str) -> str:
    col_clean = col.strip()
    col_lower = col_clean.lower()
    family = "LVL" if re.search(r"(?:^|_)lvl(?:_|$)", col_lower) else "lumber"
    woodtype = "hardwood" if "hardwood" in col_lower else "softwood"
    region = None
    for r in REGIONS:
        if re.search(fr"(?:^|_){r}(?:_|$)", col_clean):
            region = r
            break
    tokens = re.split(r"_+", col_lower)
    remove = {"lumber", "lvl", "softwood", "hardwood"} | {r.lower() for r in REGIONS}
    core_tokens = [t for t in tokens if t and t not in remove]
    attribute = normalize_attribute(col_lower, core_tokens)
    parts = [family, woodtype]
    if region:
        parts.append(region)
    parts.append(attribute)
    return "_".join(parts)

################################### Slash Accouting Fxns
def _pnw_pile_burn_CO2_CH4(fuel_dry_kg: float, consumption_fraction: float = 1.0):
    pf, ps, pr = 0.70, 0.15, 0.15
    EF_CO2_g_per_kg = pf*1714.62 + ps*1544.94 + pr*1544.94
    EF_CH4_g_per_kg = pf*1.64    + ps*5.52    + pr*5.52
    fuel_consumed_kg = fuel_dry_kg * float(consumption_fraction)
    co2_kg = (EF_CO2_g_per_kg * fuel_consumed_kg) / 1000.0
    ch4_kg = (EF_CH4_g_per_kg * fuel_consumed_kg) / 1000.0
    return co2_kg, ch4_kg


################################### Electrification + Efficiency Fxns
ELECTRIFICATION_CAP_JITTER_REL    = 0.05  
ELECTRIFICATION_PARAM_JITTER_REL  = 0.00  

def _clip(x, lo, hi):
    return max(lo, min(hi, x))

def _u_rel(base, rel, lo=0.0, hi=1.0):
    low  = _clip(base*(1.0 - rel), lo, hi)
    high = _clip(base*(1.0 + rel), lo, hi)
    if high <= low:
        return _clip(base, lo, hi)
    return float(np.random.uniform(low, high))

def logistic_share(year, mid_year, slope, cap=1.0, floor=0.0):
    x = 1.0 / (1.0 + math.exp(-slope * (year - mid_year)))
    return floor + (cap - floor) * x

def electrify_fuel_to_electric(fuel_MJ, share_to_elec, service_ratio):
    share = _clip(float(share_to_elec), 0.0, 1.0)
    shift_fuel_MJ = fuel_MJ * share
    elec_add_MJ   = shift_fuel_MJ * float(service_ratio)
    return fuel_MJ - shift_fuel_MJ, elec_add_MJ

def electrify_fossil_heat_to_HP_or_resistance(fossil_heat_MJ, share_HP, share_RES,
                                              COP_HP=3.0, eta_RES=0.98):
    share_HP  = _clip(float(share_HP), 0.0, 1.0)
    share_RES = _clip(float(share_RES), 0.0, 1.0)
    if share_HP + share_RES > 1.0:
        tot = share_HP + share_RES
        share_HP, share_RES = share_HP/tot, share_RES/tot

    shift_HP_MJ  = fossil_heat_MJ * share_HP
    shift_RES_MJ = fossil_heat_MJ * share_RES
    elec_HP_MJ   = shift_HP_MJ / max(0.1, float(COP_HP))
    elec_RES_MJ  = shift_RES_MJ / max(0.1, float(eta_RES))
    fossil_remaining_MJ = fossil_heat_MJ - (shift_HP_MJ + shift_RES_MJ)
    return fossil_remaining_MJ, elec_HP_MJ, elec_RES_MJ

def get_electrification_params(year, emission_mode):
    fast = (emission_mode == 'low')

    # Baseline adoption caps (fraction of fossil energy that can be electrified)
    cap_kiln_HP   = 0.70 if fast else 0.55   
    cap_kiln_RES  = 0.20 if fast else 0.15   
    cap_clt_HP    = 0.35 if fast else 0.25   
    cap_clt_RES   = 0.35 if fast else 0.30   
    cap_yard_BE   = 0.70 if fast else 0.55   
    cap_harv_BE   = 0.50 if fast else 0.35   
    r = ELECTRIFICATION_CAP_JITTER_REL
    cap_kiln_HP  = _u_rel(cap_kiln_HP,  r, 0.0, 1.0)
    cap_kiln_RES = _u_rel(cap_kiln_RES, r, 0.0, 1.0)
    cap_clt_HP   = _u_rel(cap_clt_HP,   r, 0.0, 1.0)
    cap_clt_RES  = _u_rel(cap_clt_RES,  r, 0.0, 1.0)
    cap_yard_BE  = _u_rel(cap_yard_BE,  r, 0.0, 1.0)
    cap_harv_BE  = _u_rel(cap_harv_BE,  r, 0.0, 1.0)

    # Adoption mid-years and slopes
    mid_kiln = 2042 if fast else 2052
    mid_clt  = 2048 if fast else 2058
    mid_yard = 2050 if fast else 2060
    mid_harv = 2055 if fast else 2065
    slope    = 0.18 if fast else 0.12
    kiln_HP_share  = logistic_share(year, mid_kiln,   slope, cap=cap_kiln_HP,  floor=0.0)
    kiln_RES_share = logistic_share(year, mid_kiln+6, slope, cap=cap_kiln_RES, floor=0.0)
    clt_HP_share   = logistic_share(year, mid_clt,    slope, cap=cap_clt_HP,   floor=0.0)
    clt_RES_share  = logistic_share(year, mid_clt+6,  slope, cap=cap_clt_RES,  floor=0.0)
    yard_BE_share  = logistic_share(year, mid_yard,   slope, cap=cap_yard_BE,  floor=0.0)
    harv_BE_share  = logistic_share(year, mid_harv,   slope, cap=cap_harv_BE,  floor=0.0)
    pr = ELECTRIFICATION_PARAM_JITTER_REL
    COP_kiln_HP = _clip(_u_rel(3.0, pr, 1.5, 5.0), 1.5, 5.0)   
    COP_clt_HP  = _clip(_u_rel(2.8, pr, 1.5, 5.0), 1.5, 5.0)
    eta_res     = _clip(_u_rel(0.98, pr, 0.90, 1.00), 0.90, 1.00)
    EER_yard_BE = _clip(_u_rel(0.45, pr, 0.20, 0.80), 0.20, 0.80)
    EER_harv_BE = _clip(_u_rel(0.50, pr, 0.30, 0.90), 0.30, 0.90)

    return dict(
        # performance
        COP_kiln_HP=COP_kiln_HP, COP_clt_HP=COP_clt_HP, eta_resistance=eta_res,
        EER_yard_BE=EER_yard_BE, EER_harv_BE=EER_harv_BE,
        # shares (fractions of fossil energy shifted)
        kiln_HP_share=kiln_HP_share, kiln_RES_share=kiln_RES_share,
        clt_HP_share=clt_HP_share,   clt_RES_share=clt_RES_share,
        yard_BE_share=yard_BE_share, harv_BE_share=harv_BE_share)

################################### Transportation + Truck Fxns
TRUCK_CAP_JITTER_REL          = 0.07      
TRUCK_PAYLOAD_JITTER_REL      = 0.15      
TRUCK_CHARGING_OVERHEAD_RANGE = (0.05, 0.10)  

# Class-8 kWh/mi ranges 
TRUCK_KWHPM_RANGE = {
    'logs2sawmill': (1.7, 2.3),
    'lumber2prod':  (1.6, 2.2),
    'prod2const':   (1.6, 2.2)}
# Average payload per truck (tonnes)
TRUCK_PAYLOAD_T_MEAN = {
    'logs2sawmill': 22.0,
    'lumber2prod':  22.0,
    'prod2const':   20.0}

def get_trucking_params(year, region, emission_mode):
    # Differs by region
    fast = (emission_mode == 'low')
    cap_bonus   = 0.05 if region == 'PNW' else 0.00
    mid_advance = -3   if region == 'PNW' else 0     
    legs = {}
    
    # Forest 2 Sawmill 
    if fast:
        cap_base, mid_base, slope = 0.85, 2042, 0.20
    else:
        cap_base, mid_base, slope = 0.70, 2052, 0.12
    legs['logs2sawmill'] = dict(
        cap=_u_rel(_clip(cap_base + cap_bonus, 0.0, 1.0), TRUCK_CAP_JITTER_REL, 0.0, 1.0),
        mid=mid_base + mid_advance,
        slope=slope)

    # Sawmill 2 CLT Plant 
    if fast:
        cap_base, mid_base, slope = 0.80, 2045, 0.18
    else:
        cap_base, mid_base, slope = 0.65, 2055, 0.11
    legs['lumber2prod'] = dict(
        cap=_u_rel(_clip(cap_base + cap_bonus, 0.0, 1.0), TRUCK_CAP_JITTER_REL, 0.0, 1.0),
        mid=mid_base + mid_advance,
        slope=slope)

    # Plant 2 Construction site 
    if fast:
        cap_base, mid_base, slope = 0.70, 2050, 0.16
    else:
        cap_base, mid_base, slope = 0.55, 2060, 0.10
        
    legs['prod2const'] = dict(
        cap=_u_rel(_clip(cap_base + cap_bonus, 0.0, 1.0), TRUCK_CAP_JITTER_REL, 0.0, 1.0),
        mid=mid_base + mid_advance,
        slope=slope)

    # Per-leg energy, payload, and charging overhead draws
    for leg in legs.keys():
        k_lo, k_hi = TRUCK_KWHPM_RANGE[leg]
        legs[leg]['kWh_per_mile'] = float(np.random.uniform(k_lo, k_hi))
        # payload per TRUCK (tonnes)
        p_mean = TRUCK_PAYLOAD_T_MEAN[leg]
        legs[leg]['payload_t_truck'] = _u_rel(p_mean, TRUCK_PAYLOAD_JITTER_REL, lo=5.0, hi=40.0)
        # charging overhead
        legs[leg]['charge_overhead'] = float(np.random.uniform(*TRUCK_CHARGING_OVERHEAD_RANGE))
    return dict(legs=legs)

def blended_ef_tkm_for_leg(leg_name, year, diesel_ef_tkm,
                           elec_kgCO2_per_MJ, tr_cfg):

    leg = tr_cfg['legs'][leg_name]
    s_bev = logistic_share(year, leg['mid'], leg['slope'], cap=leg['cap'], floor=0.0)
    # Convert kWh/mi to MJ/t·km using assumed per-truck payload (tonnes)
    ebet_MJ_per_tkm = (leg['kWh_per_mile'] * 3.6) / (1.609 * max(leg['payload_t_truck'], 0.1))
    ef_bev_tkm = elec_kgCO2_per_MJ * ebet_MJ_per_tkm * (1.0 + leg['charge_overhead'])
    ef_blended = diesel_ef_tkm * (1.0 - s_bev) + ef_bev_tkm * s_bev
    return ef_blended

def compute_transport_emissions(distance_km, payload_t_per_m3,
                                empty_backhaul_factor, diesel_ef_tkm,
                                elec_kgCO2_per_MJ, leg_name, tr_cfg,
                                year):
    
    ef_tkm = blended_ef_tkm_for_leg(leg_name, year, diesel_ef_tkm, elec_kgCO2_per_MJ, tr_cfg)
    return ef_tkm * float(distance_km) * float(empty_backhaul_factor) * float(payload_t_per_m3)


################################### Inventory Collect Fxns
def make_out_df(seq_df, harvest_df, sawmill_df, operation_df, const_dem_df, EoL_df, emission_mode,
                *, landfill_ts=None, mill_ts=None, slash_ts=None,
                use_years=50, eol_years=27):

    def get(series_or_df, col, nrows, index):
        if hasattr(series_or_df, '__getitem__') and (col in series_or_df):
            return pd.to_numeric(series_or_df[col], errors='coerce').fillna(0.0)
        return pd.Series(np.zeros(nrows), index=index, dtype=float)

    def sum_ts_window(ts_df, prefix, y_start, y_end, nrows, index):
        if ts_df is None or len(ts_df) == 0:
            return pd.Series(np.zeros(nrows), index=index, dtype=float)
        cols = [f"{prefix}{y}" for y in range(int(y_start), int(y_end) + 1) if f"{prefix}{y}" in ts_df.columns]
        if not cols:
            return pd.Series(np.zeros(nrows), index=index, dtype=float)
        S = pd.DataFrame({c: pd.to_numeric(ts_df[c], errors='coerce').fillna(0.0) for c in cols})
        if len(S) != nrows:
            S = S.iloc[:nrows, :].copy()
            if len(S) < nrows:
                pad = pd.DataFrame(0.0, index=np.arange(len(S), nrows), columns=S.columns)
                S = pd.concat([S, pad], axis=0)
        return pd.Series(S.sum(axis=1).to_numpy(), index=index, dtype=float)

    # Indices and counts
    n = len(seq_df)
    idx = seq_df.index if hasattr(seq_df, 'index') else pd.RangeIndex(n)

    # ----------------------------- RAW MATERIALS -----------------------------
    out_df = pd.DataFrame(index=idx)
    out_df['raw_materials_emissions'] = (
        pd.to_numeric(harvest_df.get('harvest_ops_CO2', 0.0), errors='coerce').fillna(0.0) +
        pd.to_numeric(harvest_df.get('fert_herb_CO2', 0.0), errors='coerce').fillna(0.0)
    )

    # Include product carbon uptake
    out_df['raw_materials_uptake'] = (
        get(seq_df, 'Sequestered_CO2', n, idx) +
        get(seq_df, 'Mill_residue_CO2', n, idx) +           
        get(seq_df, 'Sequestered_CO2_residue', n, idx)
    )
    out_df['raw_materials_CH4'] = 0.0
    out_df['raw_materials_years'] = 21

    # ----------------------------- TRANSPORT 1 -------------------------------
    out_df['transportation_1_emissions'] = get(harvest_df, 'harvest_haul_CO2', n, idx)
    out_df['transportation_1_uptake'] = 0.0
    out_df['transportation_1_CH4'] = 0.0
    out_df['transportation_1_years'] = 1

    # ----------------------------- PROCESSING (SAWMILL) ----------------------
    proc_fossil = get(sawmill_df, 'sawmill_ops_CO2', n, idx)
    out_df['processing_bio_emissions']    = 0.0
    out_df['processing_fossil_emissions'] = proc_fossil
    out_df['processing_emissions']        = out_df['processing_bio_emissions'] + out_df['processing_fossil_emissions']
    out_df['processing_uptake']           = 0.0
    out_df['processing_CH4'] = 0.0
    out_df['processing_years'] = 1

    # ----------------------------- TRANSPORT 2 -------------------------------
    out_df['transportation_2_emissions'] = get(sawmill_df, 'sawmill_haul_CO2', n, idx)
    out_df['transportation_2_uptake'] = 0.0
    out_df['transportation_2_CH4'] = 0.0
    out_df['transportation_2_years'] = 1

    # ----------------------------- MANUFACTURE (CLT PLANT) -------------------
    out_df['operation_emissions'] = get(operation_df, 'total_op_process_CO2_fossil', n, idx)
    out_df['operation_uptake']    = 0.0
    out_df['operation_CH4'] = 0.0
    out_df['operation_years'] = 1

    # ----------------------------- TRANSPORT 3 -------------------------------
    out_df['transportation_3_emissions'] = get(operation_df, 'op_haul_CO2', n, idx)
    out_df['transportation_3_uptake'] = 0.0
    out_df['transportation_3_CH4'] = 0.0
    out_df['transportation_3_years'] = 1

    # ----------------------------- CONSTRUCTION ------------------------------
    out_df['construction_emissions'] = get(const_dem_df, 'construction_CO2', n, idx)
    out_df['construction_uptake'] = 0.0
    out_df['construction_CH4'] = 0.0
    out_df['construction_years'] = 1

    # ----------------------------- Residue Windows  ------------------------------
    mr_use_emit  = sum_ts_window(mill_ts,  "emit_year_", 1,            use_years,            n, idx)
    mr_use_cred  = sum_ts_window(mill_ts,  "cred_year_", 1,            use_years,            n, idx)
    mr_eol_emit  = sum_ts_window(mill_ts,  "emit_year_", use_years+1,  use_years+eol_years,  n, idx)
    mr_eol_cred  = sum_ts_window(mill_ts,  "cred_year_", use_years+1,  use_years+eol_years,  n, idx)

    # Slash time series 
    slash_use = sum_ts_window(slash_ts, "year_", 1,           use_years,           n, idx)
    slash_eol = sum_ts_window(slash_ts, "year_", use_years+1, use_years+eol_years, n, idx)

    # Landfill time series 
    lf_co2_27  = sum_ts_window(landfill_ts, "co2_year_",  1, eol_years, n, idx)
    lf_ch4_27  = sum_ts_window(landfill_ts, "ch4_year_",  1, eol_years, n, idx)
    lf_cred_27 = sum_ts_window(landfill_ts, "cred_year_", 1, eol_years, n, idx)

    # ----------------------------- USE (0–50 years) --------------------------
    if str(emission_mode).lower() == "low":
        # Clean pathway: 
        use_emit_clean   = get(seq_df, 'clean_residue_bio_CO2', n, idx)         
        use_credit_clean = get(seq_df, 'clean_residue_avoided_fossil_CO2', n, idx)
        out_df['use_emissions'] = (use_emit_clean + mr_use_emit)
        out_df['use_uptake']    = (use_credit_clean + mr_use_cred)
        out_df['use_CH4']       = 0.0
        out_df['use_years']     = use_years

    else:  
        # High pathway:
        pnw_pile_co2 = get(seq_df, 'Residue_new_CO2', n, idx)   
        pnw_pile_ch4 = get(seq_df, 'Residue_new_CH4', n, idx)   
        out_df['use_emissions'] = (pnw_pile_co2 + slash_use + mr_use_emit)
        out_df['use_uptake']    = (0.0 + 0.0 + mr_use_cred)     
        out_df['use_CH4']       = pnw_pile_ch4                  
        out_df['use_years']     = use_years

    # ----------------------------- DECONSTRUCTION ----------------------------
    out_df['deconstruction_emissions'] = get(const_dem_df, 'deconstruction_CO2', n, idx)
    out_df['deconstruction_uptake'] = 0.0
    out_df['deconstruction_CH4'] = 0.0
    out_df['deconstruction_years'] = 1

    # ----------------------------- TRANSPORT 4 -------------------------------
    trans_4_scale = np.random.uniform(0.2, 2.0, size=n)
    out_df['transportation_4_emissions'] = trans_4_scale * get(operation_df, 'op_haul_CO2', n, idx)
    out_df['transportation_4_uptake'] = 0.0
    out_df['transportation_4_CH4'] = 0.0
    out_df['transportation_4_years'] = 1

    # ----------------------------- EOL (27 years) ----------------------------
    if str(emission_mode).lower() == "low":
        # EoL_1: Biochar 
        e1_bio = get(EoL_df, 'EoL_3_bio_CO2', n, idx) + mr_eol_emit + slash_eol
        e1_fos = 0.0
        e1_crd = get(EoL_df, 'EoL_3_avoided_fossil_CO2', n, idx) + mr_eol_cred
        out_df['EoL_1_bio_emissions']            = e1_bio
        out_df['EoL_1_fossil_emissions']         = e1_fos
        out_df['EoL_1_avoided_fossil_emissions'] = e1_crd
        out_df['EoL_1_emissions']                = e1_bio + e1_fos
        out_df['EoL_1_uptake']                   = e1_crd
        out_df['EoL_1_CH4']                      = 0.0
        out_df['EoL_1_years']                    = 1 

        # EoL_2: Reuse 
        e2_bio = (0.0 + mr_eol_emit + slash_eol)
        e2_fos = get(EoL_df, 'EoL_4_fossil_CO2', n, idx)
        e2_crd = (0.0 + mr_eol_cred)
        out_df['EoL_2_bio_emissions']            = e2_bio
        out_df['EoL_2_fossil_emissions']         = e2_fos
        out_df['EoL_2_avoided_fossil_emissions'] = e2_crd
        out_df['EoL_2_emissions']                = e2_bio + e2_fos
        out_df['EoL_2_uptake']                   = e2_crd
        out_df['EoL_2_CH4']                      = 0.0
        out_df['EoL_2_years']                    = eol_years

    else:  # "high"
        # EoL_1: Combustion 
        e1_bio = get(EoL_df, 'EoL_1_bio_CO2', n, idx) + mr_eol_emit + slash_eol
        e1_fos = get(EoL_df, 'EoL_1_fossil_CO2', n, idx)
        e1_crd = get(EoL_df, 'EoL_1_avoided_fossil_CO2', n, idx) + mr_eol_cred
        out_df['EoL_1_bio_emissions']            = e1_bio
        out_df['EoL_1_fossil_emissions']         = e1_fos
        out_df['EoL_1_avoided_fossil_emissions'] = e1_crd
        out_df['EoL_1_emissions']                = e1_bio + e1_fos
        out_df['EoL_1_uptake']                   = e1_crd
        out_df['EoL_1_CH4']                      = 0.0
        out_df['EoL_1_years']                    = 1 

        # EoL_2: Landfill 
        e2_bio  = lf_co2_27 + mr_eol_emit + slash_eol
        e2_fos  = 0.0
        e2_crd  = lf_cred_27 + mr_eol_cred
        e2_ch4  = lf_ch4_27
        out_df['EoL_2_bio_emissions']            = e2_bio
        out_df['EoL_2_fossil_emissions']         = e2_fos
        out_df['EoL_2_avoided_fossil_emissions'] = e2_crd
        out_df['EoL_2_emissions']                = e2_bio + e2_fos
        out_df['EoL_2_uptake']                   = e2_crd
        out_df['EoL_2_CH4']                      = e2_ch4
        out_df['EoL_2_years']                    = eol_years

    return out_df

def _read_optional_csv(path, **read_kwargs):
    return pd.read_csv(path, **read_kwargs) if os.path.exists(path) else pd.DataFrame()