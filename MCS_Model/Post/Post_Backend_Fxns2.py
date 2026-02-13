from __future__ import annotations
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Sequence, Mapping

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, FixedLocator, FixedFormatter, MaxNLocator
from scipy.stats import gaussian_kde

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x
#%%
###############################################################################
###############################################################################
# Dynamic Carbon Plot
###############################################################################
###############################################################################
# -------------------- Config --------------------

@dataclass
class DcubeConfig:
    base_year: int = 2020
    growth_years: int = 21

    service_life_sampling: Optional[Dict] = None
    service_life_min_default: int = 45
    service_life_max_default: int = 55

    emission_scenario: str = "low"
    region: str = "PNW"
    production_scenario: str = "S1"

    growth_model: str = "richards"
    growth_params: Optional[Dict] = None

    scale_uptake: bool = True
    unit_in: str = "kg"
    unit_out: str = "t"
    gwp100_ch4: float = 28.0

    landfill_years: int = 30
    landfill_start_offset: int = 1
    landfill_half_life: float = 20.0
    landfill_k: Optional[float] = None

    landfill_use_cdf_target: bool = True
    landfill_cdf_target: float = 0.99
    landfill_assert_mass_balance: bool = False

    landfill_half_life_sampling: Optional[Dict] = None
    landfill_region_climate_hint: Optional[str] = None
    landfill_sampling_seed: Optional[int] = None


LOW_REQUIRED_COLS = [
    "raw_materials_emissions", "raw_materials_uptake",
    "transportation_1_emissions", "processing_bio_emissions", "processing_fossil_emissions",
    "transportation_2_emissions", "operation_emissions", "transportation_3_emissions",
    "construction_emissions", "use_emissions", "use_uptake",
    "deconstruction_emissions", "transportation_4_emissions",
    "EoL_1_emissions", "EoL_1_uptake", "EoL_2_emissions",
]

HIGH_REQUIRED_COLS = [
    "raw_materials_emissions", "raw_materials_uptake",
    "transportation_1_emissions", "processing_bio_emissions", "processing_fossil_emissions",
    "transportation_2_emissions", "operation_emissions", "transportation_3_emissions",
    "construction_emissions", "deconstruction_emissions", "transportation_4_emissions",
    "use_emissions", "use_CH4",
    "EoL_1_bio_emissions", "EoL_1_fossil_emissions", "EoL_1_avoided_fossil_emissions",
    "EoL_2_bio_emissions", "EoL_2_avoided_fossil_emissions", "EoL_2_CH4",
]


FS_AXIS_LABEL = 14
FS_TICK = 12

low_color = "#542788"
high_color = "#E66101"

SC_ORDER = [
    ("low",  2, "LEP — Reuse",      low_color,  "-"),
    ("low",  1, "LEP — Pyrolysis",  low_color,  "-."),
    ("high", 2, "HEP — Landfill",   high_color, "-"),
    ("high", 1, "HEP — Energy",     high_color, "-."),
]

DEFAULT_EPA_BAU_WEIGHTS: Dict[Tuple[str, int], float] = {
    ("high", 2): 0.7,
    ("high", 1): 0.3,
    ("low",  2): 0.0,
    ("low",  1): 0.0,
}


# -------------------- Core helpers --------------------

def _check_required_columns(df: pd.DataFrame, required: List[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _ensure_series(x, like_index: pd.Index, name: str) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] != 1:
            raise ValueError(f"{name} must be Series or single-col DataFrame")
        x = x.iloc[:, 0]
    if isinstance(x, pd.Series):
        if not x.index.equals(like_index):
            x = x.reindex(like_index)
    else:
        x = pd.Series(x, index=like_index)
    x.name = name
    return x


def _replicate_series_from_index(index: pd.Index) -> pd.Series:
    try:
        s = pd.to_numeric(pd.Series(index), errors="raise")
        return pd.Series(s.values, index=index, name="replicate")
    except Exception:
        return pd.Series(range(len(index)), index=index, name="replicate")


def _growth_weights(n: int, model: str = "linear", params: Optional[Dict] = None) -> np.ndarray:
    if n <= 0:
        raise ValueError("growth_years must be > 0")

    params = params or {}
    t = np.linspace(0.0, 1.0, n + 1)

    if model == "linear":
        inc = np.ones(n, dtype=float)

    elif model in ("sigmoid", "logistic", "s-curve"):
        k = float(params.get("k", 6.0))
        x0 = float(params.get("x0", 0.55))
        F = 1.0 / (1.0 + np.exp(-k * (t - x0)))
        inc = np.diff(F)

    elif model == "gompertz":
        b = float(params.get("b", 2.0))
        c = float(params.get("c", 3.0))
        F = np.exp(-b * np.exp(-c * t))
        F = (F - F[0]) / (F[-1] - F[0] + 1e-15)
        inc = np.diff(F)

    elif model in ("richards", "chapman_richards"):
        k = float(params.get("k", 4.0))
        x0 = float(params.get("x0", 0.45))
        nu = float(params.get("nu", 1.6))
        eps = float(params.get("eps", 1e-6))
        a = ((1 - eps) ** (-nu) - 1.0) * np.exp(k * (1 - x0))
        F = (1.0 + a * np.exp(-k * (t - x0))) ** (-1.0 / nu)
        F = (F - F[0]) / (F[-1] - F[0] + 1e-15)
        inc = np.diff(F)

    else:
        raise ValueError(f"unknown growth_model '{model}'")

    w = np.clip(inc, 0.0, None)
    s = w.sum()
    return (w / s) if s > 0 else (np.ones(n, float) / float(n))


def allocate_growth(
    total_uptake: pd.Series,
    base_year: int,
    growth_years: int,
    model="linear",
    params=None,
    include_base_year=True,
) -> pd.DataFrame:
    if include_base_year:
        start = base_year - growth_years + 1
        years = list(range(start, base_year + 1))
    else:
        years = list(range(base_year - growth_years, base_year))
    w = _growth_weights(growth_years, model, params)
    arr = np.outer(w, total_uptake.to_numpy())
    return pd.DataFrame(arr, index=years, columns=total_uptake.index)


def get_region_default_half_life_sampling(
    region: str,
    *,
    climate_hint: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict:
    cat = (climate_hint or "wet").strip().lower()
    if cat not in ("wet", "dry"):
        cat = "wet"
    if cat == "wet":
        spec = {"dist": "triangular", "lo": 17.0, "mode": 23.0, "hi": 35.0}
    else:
        spec = {"dist": "triangular", "lo": 23.0, "mode": 35.0, "hi": 69.0}
    if seed is not None:
        spec["seed"] = int(seed)
    return spec


def _sample_service_lives(
    rep_index: pd.Index,
    spec: Optional[Dict],
    lo_default: int,
    hi_default: int,
) -> pd.Series:
    spec = spec or {"dist": "uniform", "lo": lo_default, "hi": hi_default}
    dist = str(spec.get("dist", "uniform")).lower()
    seed = spec.get("seed", None)
    rng = np.random.default_rng(seed)
    n = len(rep_index)

    if dist == "uniform":
        lo = float(spec.get("lo", lo_default))
        hi = float(spec.get("hi", hi_default))
        if hi <= lo:
            raise ValueError("service life uniform: hi must be > lo")
        sl = rng.uniform(lo, hi, size=n)

    elif dist == "triangular":
        lo = float(spec.get("lo", lo_default))
        mode = float(spec.get("mode", (lo_default + hi_default) / 2.0))
        hi = float(spec.get("hi", hi_default))
        if not (lo <= mode <= hi):
            raise ValueError("service life triangular: lo <= mode <= hi required")
        sl = rng.triangular(lo, mode, hi, size=n)

    elif dist == "discrete":
        vals = spec.get("vals", list(range(lo_default, hi_default + 1)))
        p = spec.get("p", None)
        sl = rng.choice(vals, size=n, p=p)

    else:
        raise ValueError(f"Unknown service life dist '{dist}'")

    return pd.Series(np.round(sl).astype(int), index=rep_index, name="service_life")


def _scale_to_prod_and_tonnes(
    em: pd.DataFrame,
    production: pd.Series,
    cols: List[str],
    scale_uptake: bool,
    unit_in: str,
    unit_out: str,
) -> pd.DataFrame:
    em = em.copy()
    production = _ensure_series(production, em.index, "production")
    to_scale = cols if scale_uptake else [c for c in cols if "uptake" not in c.lower()]
    for c in to_scale:
        em[c] = em.get(c, 0.0)
        em[c] = em[c].multiply(production, axis=0)
    if unit_in.lower().startswith("kg") and unit_out.lower().startswith("t"):
        for c in cols:
            em[c] = em.get(c, 0.0) / 1000.0
    return em


def _scale_ts_cols_to_prod_and_tonnes(
    ts_df: Optional[pd.DataFrame],
    production: pd.Series,
    unit_in: str,
    unit_out: str,
) -> Optional[pd.DataFrame]:
    if ts_df is None or ts_df.empty:
        return ts_df
    ts = ts_df.copy()
    production = _ensure_series(production, ts.index, "production")
    families = ("year_", "emit_year_", "cred_year_", "co2_year_", "ch4_year_")
    for c in list(ts.columns):
        if any(c.startswith(p) for p in families):
            ts[c] = ts[c].multiply(production, axis=0)
            if unit_in.lower().startswith("kg") and unit_out.lower().startswith("t"):
                ts[c] = ts[c] / 1000.0
    for c in ("release_sum_0_50", "unreleased_after_50"):
        if c in ts.columns:
            ts[c] = ts[c].multiply(production, axis=0)
            if unit_in.lower().startswith("kg") and unit_out.lower().startswith("t"):
                ts[c] = ts[c] / 1000.0
    return ts


def _sample_half_lives_for_reps(rep_index: pd.Index, spec: Dict) -> pd.Series:
    dist = str(spec.get("dist", "uniform")).lower()
    seed = spec.get("seed", None)
    rng = np.random.default_rng(seed)
    n = len(rep_index)

    if dist == "uniform":
        lo = float(spec.get("lo", 17.0))
        hi = float(spec.get("hi", 35.0))
        if hi <= lo:
            raise ValueError("uniform: hi must be > lo")
        hl = rng.uniform(lo, hi, size=n)

    elif dist == "triangular":
        lo = float(spec.get("lo", 17.0))
        mode = float(spec.get("mode", 23.0))
        hi = float(spec.get("hi", 35.0))
        if not (lo <= mode <= hi):
            raise ValueError("triangular: lo <= mode <= hi required")
        hl = rng.triangular(lo, mode, hi, size=n)

    elif dist == "lognormal":
        median = float(spec.get("median", 25.0))
        gsd = float(spec.get("gsd", 1.3))
        mu, sigma = np.log(median), np.log(gsd)
        hl = rng.lognormal(mean=mu, sigma=sigma, size=n)
        if "lo" in spec:
            hl = np.maximum(hl, float(spec["lo"]))
        if "hi" in spec:
            hl = np.minimum(hl, float(spec["hi"]))

    else:
        raise ValueError(f"Unknown dist '{dist}' for landfill_half_life_sampling")

    return pd.Series(hl, index=rep_index, name="half_life")


def allocate_fod_variable_k(
    total: pd.Series,
    start_year: int,
    k_per_rep: pd.Series,
    *,
    use_cdf_target: bool = True,
    target: float = 0.99,
    n_years: Optional[int] = None,
    normalize: bool = False,
) -> pd.DataFrame:
    k = _ensure_series(k_per_rep, total.index, "k")
    vals = total.to_numpy()[None, :]

    if use_cdf_target:
        target = float(np.clip(target, 1e-9, 0.999999))
        n_r = np.ceil(-np.log(1.0 - target) / k.to_numpy()).astype(int)
        n_r = np.maximum(n_r, 1)
        n_max = int(n_r.max())
    else:
        if n_years is None or int(n_years) < 1:
            raise ValueError("n_years must be >= 1 if use_cdf_target=False")
        n_max = int(n_years)
        n_r = np.full(len(k), n_max, dtype=int)

    t = np.arange(1, n_max + 1, dtype=float)[:, None]
    k_row = k.to_numpy()[None, :]
    w = np.exp(-k_row * (t - 1.0)) - np.exp(-k_row * t)

    mask = (t <= n_r[None, :])
    w = np.where(mask, w, 0.0)

    if normalize:
        col_sums = w.sum(axis=0, keepdims=True)
        w = np.divide(
            w,
            np.where(col_sums > 0, col_sums, 1.0),
            out=np.zeros_like(w),
            where=col_sums > 0,
        )

    arr = w * vals
    years = list(range(start_year, start_year + n_max))
    return pd.DataFrame(arr, index=years, columns=total.index)


# -------------------- Flow builder --------------------

def compute_flows_for_product(
    emissions_df: pd.DataFrame,
    production_df: pd.DataFrame,
    cfg: DcubeConfig,
    product_label: str,
    use_schedules: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    sc = cfg.emission_scenario.lower()
    use_schedules = use_schedules or {}

    if sc == "low":
        _check_required_columns(emissions_df, LOW_REQUIRED_COLS, f"{cfg.region}-{product_label} LOW")
        req = LOW_REQUIRED_COLS[:]
    elif sc == "high":
        _check_required_columns(emissions_df, HIGH_REQUIRED_COLS, f"{cfg.region}-{product_label} HIGH")
        req = HIGH_REQUIRED_COLS[:]
    else:
        raise ValueError("emission_scenario must be 'low' or 'high'")

    production_df = production_df.copy()
    try:
        production_df.columns = [int(c) if str(c).isdigit() else c for c in production_df.columns]
    except Exception:
        pass

    ycol_int, ycol_str = int(cfg.base_year), str(cfg.base_year)
    if ycol_int in production_df.columns:
        prod_col = production_df[ycol_int]
    elif ycol_str in production_df.columns:
        prod_col = production_df[ycol_str]
    else:
        raise ValueError(
            f"production_df missing base year {cfg.base_year}. "
            f"Columns look like: {list(production_df.columns)[:10]}"
        )

    production = _ensure_series(prod_col, emissions_df.index, "production")
    em = _scale_to_prod_and_tonnes(emissions_df, production, req, cfg.scale_uptake, cfg.unit_in, cfg.unit_out)

    rep_ids = _replicate_series_from_index(em.index)
    service_life = _sample_service_lives(
        em.index,
        cfg.service_life_sampling,
        cfg.service_life_min_default,
        cfg.service_life_max_default,
    )

    mr_ts = _scale_ts_cols_to_prod_and_tonnes(use_schedules.get("mill", None), production, cfg.unit_in, cfg.unit_out)
    sl_ts = _scale_ts_cols_to_prod_and_tonnes(use_schedules.get("slash", None), production, cfg.unit_in, cfg.unit_out)
    lf_ts = _scale_ts_cols_to_prod_and_tonnes(use_schedules.get("landfill", None), production, cfg.unit_in, cfg.unit_out)
    has_mr_ts = (mr_ts is not None) and (not mr_ts.empty)

    rows = []

    def _emit(year: int, stage: str, metric: str, series: pd.Series):
        s = series.fillna(0.0).astype(float)
        rows.append(pd.DataFrame({
            "replicate": rep_ids.values,
            "year": year,
            "region": cfg.region,
            "product": product_label,
            "emission_scenario": sc,
            "production_scenario": cfg.production_scenario,
            "stage": stage,
            "metric": metric,
            "value": s.values,
        }))

    def _emit_at_series_years(stage: str, metric: str, series: pd.Series, year_series: pd.Series):
        s = series.fillna(0.0).astype(float)
        yrs = year_series.astype(int)
        rows.append(pd.DataFrame({
            "replicate": rep_ids.values,
            "year": yrs.values,
            "region": cfg.region,
            "product": product_label,
            "emission_scenario": sc,
            "production_scenario": cfg.production_scenario,
            "stage": stage,
            "metric": metric,
            "value": s.values,
        }))

    def _col(df: Optional[pd.DataFrame], prefix: str, k: int) -> pd.Series:
        if df is None:
            return pd.Series(0.0, index=em.index)
        col = f"{prefix}{k}"
        if col not in df.columns:
            return pd.Series(0.0, index=em.index)
        return df[col].reindex(em.index).fillna(0.0)

    def _max_k_for(df: Optional[pd.DataFrame], prefix: str) -> int:
        if df is None:
            return 0
        ks = []
        for c in df.columns:
            if c.startswith(prefix):
                try:
                    ks.append(int(c.split("_")[-1]))
                except Exception:
                    pass
        return max(ks) if ks else 0

    alloc = allocate_growth(
        em["raw_materials_uptake"],
        cfg.base_year,
        cfg.growth_years,
        cfg.growth_model,
        cfg.growth_params,
        include_base_year=True,
    )

    tot_alloc = alloc.sum(axis=0).astype(float).fillna(0.0)
    tot_true = em["raw_materials_uptake"].astype(float).fillna(0.0)
    diff = (tot_alloc - tot_true).abs()
    ok = diff <= (1e-6 + 1e-10 * tot_true.abs())
    if not ok.all():
        bad = (~ok).to_numpy().nonzero()[0][:10]
        details = pd.DataFrame({
            "replicate": tot_true.index.to_numpy()[bad],
            "alloc_sum_t": tot_alloc.to_numpy()[bad],
            "true_t": tot_true.to_numpy()[bad],
            "diff_t": diff.to_numpy()[bad],
        })
        raise AssertionError(f"Uptake mass-balance failed (showing up to 10):\n{details}")

    for yr in alloc.index:
        _emit(int(yr), "raw_materials", "uptake", alloc.loc[yr])

    y0 = int(cfg.base_year)
    _emit(y0, "raw_materials", "fossil", em["raw_materials_emissions"])
    _emit(y0, "transportation_1", "fossil", em["transportation_1_emissions"])
    if not has_mr_ts:
        _emit(y0, "processing", "biogenic", em["processing_bio_emissions"])
    _emit(y0, "processing", "fossil", em["processing_fossil_emissions"])
    _emit(y0, "transportation_2", "fossil", em["transportation_2_emissions"])
    _emit(y0, "manufacture", "fossil", em["operation_emissions"])
    _emit(y0, "transportation_3", "fossil", em["transportation_3_emissions"])
    _emit(y0, "construction", "fossil", em["construction_emissions"])

    if sc == "low":
        kmax = max(_max_k_for(mr_ts, "emit_year_"), _max_k_for(mr_ts, "cred_year_"))
        for k in range(1, kmax + 1):
            year_global = y0 + (k - 1)
            emit_k = _col(mr_ts, "emit_year_", k)
            cred_k = _col(mr_ts, "cred_year_", k)
            if emit_k.abs().sum() > 0:
                _emit(int(year_global), "use", "biogenic", emit_k)
            if cred_k.abs().sum() > 0:
                _emit(int(year_global), "use", "energy_credit", cred_k)

        if "use_emissions" in em.columns and em["use_emissions"].abs().sum() > 0:
            _emit(y0, "use", "biogenic", em["use_emissions"])
        if "use_uptake" in em.columns and em["use_uptake"].abs().sum() > 0:
            _emit(y0, "use", "energy_credit", em["use_uptake"])

    else:
        if cfg.region.upper() == "SE":
            kmax = max(
                _max_k_for(sl_ts, "emit_year_"),
                _max_k_for(mr_ts, "emit_year_"),
                _max_k_for(mr_ts, "cred_year_"),
            )
            for k in range(1, kmax + 1):
                year_global = y0 + (k - 1)
                emit_k = _col(sl_ts, "emit_year_", k).add(_col(mr_ts, "emit_year_", k), fill_value=0.0)
                cred_k = _col(mr_ts, "cred_year_", k)
                if emit_k.abs().sum() > 0:
                    _emit(int(year_global), "use", "biogenic", emit_k)
                if cred_k.abs().sum() > 0:
                    _emit(int(year_global), "use", "energy_credit", cred_k)
        else:
            pile_burn_now = (em.get("use_emissions", 0.0) + cfg.gwp100_ch4 * em.get("use_CH4", 0.0)).clip(lower=0.0)
            if pile_burn_now.abs().sum() > 0:
                _emit(y0, "use", "biogenic", pile_burn_now)

            kmax = max(_max_k_for(mr_ts, "emit_year_"), _max_k_for(mr_ts, "cred_year_"))
            for k in range(1, kmax + 1):
                year_global = y0 + (k - 1)
                emit_k = _col(mr_ts, "emit_year_", k)
                cred_k = _col(mr_ts, "cred_year_", k)
                if emit_k.abs().sum() > 0:
                    _emit(int(year_global), "use", "biogenic", emit_k)
                if cred_k.abs().sum() > 0:
                    _emit(int(year_global), "use", "energy_credit", cred_k)

    y_eol_series = y0 + service_life
    _emit_at_series_years("deconstruction", "fossil", em["deconstruction_emissions"], y_eol_series)
    _emit_at_series_years("transportation_4", "fossil", em["transportation_4_emissions"], y_eol_series)

    if sc == "low":
        _emit_at_series_years("EoL_1_biochar", "biogenic", em["EoL_1_emissions"], y_eol_series)
        _emit_at_series_years("EoL_1_biochar", "energy_credit", em["EoL_1_uptake"], y_eol_series)
        _emit_at_series_years("EoL_2_reuse", "fossil", em["EoL_2_emissions"], y_eol_series)

    else:
        _emit_at_series_years("EoL_1_combustion", "biogenic", em["EoL_1_bio_emissions"], y_eol_series)
        _emit_at_series_years("EoL_1_combustion", "fossil", em["EoL_1_fossil_emissions"], y_eol_series)
        _emit_at_series_years("EoL_1_combustion", "energy_credit", em["EoL_1_avoided_fossil_emissions"], y_eol_series)

        if lf_ts is not None and not lf_ts.empty:
            start_series = (y_eol_series + 1).astype(int)
            kmax = max(
                _max_k_for(lf_ts, "co2_year_"),
                _max_k_for(lf_ts, "ch4_year_"),
                _max_k_for(lf_ts, "cred_year_"),
            )
            for k in range(1, kmax + 1):
                yr = start_series + (k - 1)
                co2k = _col(lf_ts, "co2_year_", k)
                ch4k = _col(lf_ts, "ch4_year_", k) * cfg.gwp100_ch4
                credk = _col(lf_ts, "cred_year_", k)
                bio = co2k.add(ch4k, fill_value=0.0)
                if bio.abs().sum() > 0:
                    _emit_at_series_years("EoL_2_landfill", "biogenic", bio, yr)
                if credk.abs().sum() > 0:
                    _emit_at_series_years("EoL_2_landfill", "energy_credit", credk, yr)
        else:
            start_series = (y_eol_series + cfg.landfill_start_offset).astype(int)
            co2_total = _ensure_series(emissions_df["EoL_2_bio_emissions"], em.index, "co2_total")
            ch4_total = _ensure_series(emissions_df["EoL_2_CH4"], em.index, "ch4_total") * cfg.gwp100_ch4
            cred_total = _ensure_series(emissions_df["EoL_2_avoided_fossil_emissions"], em.index, "cred_total")

            if cfg.scale_uptake:
                co2_total = co2_total * production
                ch4_total = ch4_total * production
                cred_total = cred_total * production
            if cfg.unit_in.lower().startswith("kg") and cfg.unit_out.lower().startswith("t"):
                co2_total = co2_total / 1000.0
                ch4_total = ch4_total / 1000.0
                cred_total = cred_total / 1000.0

            if cfg.landfill_half_life_sampling is not None:
                sampling_spec = dict(cfg.landfill_half_life_sampling)
            else:
                sampling_spec = get_region_default_half_life_sampling(
                    cfg.region,
                    climate_hint=getattr(cfg, "landfill_region_climate_hint", None),
                    seed=getattr(cfg, "landfill_sampling_seed", None),
                )

            hl = _sample_half_lives_for_reps(em.index, sampling_spec)
            k_series = np.log(2.0) / hl

            start_min = int(start_series.min())
            target = float(getattr(cfg, "landfill_cdf_target", 0.99))

            co2_alloc = allocate_fod_variable_k(co2_total, start_min, k_series, use_cdf_target=True, target=target, normalize=False)
            ch4_alloc = allocate_fod_variable_k(ch4_total, start_min, k_series, use_cdf_target=True, target=target, normalize=False)
            cred_alloc = allocate_fod_variable_k(cred_total, start_min, k_series, use_cdf_target=True, target=target, normalize=False)

            def _shift_alloc(alloc_df: pd.DataFrame, starts: pd.Series) -> pd.DataFrame:
                frames = []
                for rep in alloc_df.columns:
                    shift = int(starts.loc[rep]) - start_min
                    s = alloc_df[rep].copy()
                    s.index = s.index + shift
                    frames.append(s)
                out = pd.concat(frames, axis=1)
                out.index.name = "year"
                return out.fillna(0.0)

            co2_s = _shift_alloc(co2_alloc, start_series)
            ch4_s = _shift_alloc(ch4_alloc, start_series)
            cred_s = _shift_alloc(cred_alloc, start_series)
            bio_s = co2_s.add(ch4_s, fill_value=0.0)

            for yr in bio_s.index:
                _emit(int(yr), "EoL_2_landfill", "biogenic", bio_s.loc[yr].reindex(em.index).fillna(0.0))
                _emit(int(yr), "EoL_2_landfill", "energy_credit", cred_s.loc[yr].reindex(em.index).fillna(0.0))

    flows_long = pd.concat(rows, ignore_index=True)
    wmap = {"fossil": 1.0, "biogenic": 1.0, "uptake": -1.0, "energy_credit": -1.0}
    flows_long["net"] = flows_long["value"] * flows_long["metric"].map(wmap)
    return {"flows_long": flows_long}


def combine_products_aggregated(flows_long_list: List[pd.DataFrame]) -> pd.DataFrame:
    raw = pd.concat(flows_long_list, ignore_index=True)
    keys = ["replicate", "year", "region", "emission_scenario", "production_scenario", "stage", "metric"]
    combined = raw.groupby(keys, as_index=False)["value"].sum()
    wmap = {"fossil": 1.0, "biogenic": 1.0, "uptake": -1.0, "energy_credit": -1.0}
    combined["net"] = combined["value"] * combined["metric"].map(wmap)
    return combined


def make_eol_variant(flows_long: pd.DataFrame, emission_scenario: str, eol_variant: int) -> pd.DataFrame:
    esc = emission_scenario.lower()
    fl = flows_long[flows_long["emission_scenario"] == esc]
    if esc == "low":
        drop_stage = "EoL_2_reuse" if eol_variant == 1 else "EoL_1_biochar"
    elif esc == "high":
        drop_stage = "EoL_2_landfill" if eol_variant == 1 else "EoL_1_combustion"
    else:
        raise ValueError("scenario must be 'low' or 'high'")
    return fl[fl["stage"] != drop_stage].copy()


def net_by_year(flows_long: pd.DataFrame) -> pd.DataFrame:
    return flows_long.groupby(["replicate", "year"], as_index=False)["net"].sum()


# -------------------- Data loaders --------------------

EMISSIONS_CACHE: Dict[Tuple[str, str, str, int], pd.DataFrame] = {}
TS_CACHE: Dict[Tuple[str, str, str, int, str], pd.DataFrame] = {}
SCENARIO_FOLDER_CACHE: dict[tuple[str, str, str, str], str] = {}


def _resolve_path(p: str | os.PathLike) -> Path:
    p = Path(p).expanduser()
    if p.is_absolute():
        return p.resolve()

    cand = (Path.cwd() / p).resolve()
    if cand.exists():
        return cand

    return (Path(__file__).resolve().parent / p).resolve()


def _find_dir_case_insensitive(parent: Path, name: str) -> Path | None:
    if not parent.is_dir():
        return None
    direct = parent / name
    if direct.is_dir():
        return direct
    target = name.casefold()
    for child in parent.iterdir():
        if child.is_dir() and child.name.casefold() == target:
            return child
    return None


# def _scenario_folder(main_path: str, region: str, product: str, scenario: str) -> str:
#     root = _resolve_path(main_path)

#     region = region.upper().strip()
#     product = product.upper().strip()
#     scenario = scenario.lower().strip()

#     if region == "PNW":
#         region_root_names = ["NW", "PNW_softwood_CLT", "PNW"]
#         prefix_names = ["PNW", "NW"]
#     elif region == "SE":
#         region_root_names = ["SE", "SE_softwood_CLT"]
#         prefix_names = ["SE"]
#     else:
#         raise ValueError("region must be 'PNW' or 'SE'")

#     if product in ("GLT", "GLULAM", "GLULAMINATED", "GLULAM"):
#         prod_tokens = ["Glulam", "GLULAM", "GLT"]
#     elif product == "CLT":
#         prod_tokens = ["CLT"]
#     else:
#         raise ValueError("product must be 'CLT' or 'GLT'")

#     parent_candidates: list[Path] = [root, root / "MCS_Results"]
#     for rr in region_root_names:
#         rr_dir = _find_dir_case_insensitive(root, rr) or (root / rr)
#         parent_candidates += [rr_dir / "MCS_Results", rr_dir]

#     sub_candidates: list[str] = []
#     for pref in prefix_names:
#         for pt in prod_tokens:
#             sub_candidates.append(f"{pref}_softwood_{pt}_{scenario}")
#             sub_candidates.append(f"{pref}_softwood_{pt}_{scenario}".lower())

#     tried: list[str] = []
#     for parent in parent_candidates:
#         for sub in sub_candidates:
#             tried.append(str(parent / sub))
#             found = _find_dir_case_insensitive(parent, sub)
#             if found is not None:
#                 return str(found)

#     raise FileNotFoundError(
#         "Could not locate scenario folder.\n"
#         f"main_path resolved to: {root}\n"
#         f"region={region}, product={product}, scenario={scenario}\n"
#         "Tried (first ~12):\n  - " + "\n  - ".join(tried[:12])
#     )

def _scenario_folder(
    main_path: str,
    region: str,
    product: str,
    scenario: str,
    *,
    year: int | None = None,
    require_stage_files: bool = False,
) -> str:

    root = _resolve_path(main_path)

    region_u = region.upper().strip()
    product_u = product.upper().strip()
    scenario_l = scenario.lower().strip()


    if region_u == "PNW":
        region_roots = ["NW", "PNW"]
        prefixes = ["PNW", "NW"]
    elif region_u == "SE":
        region_roots = ["SE"]
        prefixes = ["SE"]
    else:
        raise ValueError("region must be 'PNW' or 'SE'")

    if product_u in ("GLT", "GLULAM", "GLULAMINATED", "GLULAMINATE"):
        prod_tokens = ["GLT", "Glulam", "GLULAM"]
    elif product_u == "CLT":
        prod_tokens = ["CLT"]
    else:
        raise ValueError("product must be 'CLT' or 'GLT'")

    cache_key = (str(root), region_u, product_u, scenario_l)
    cached = SCENARIO_FOLDER_CACHE.get(cache_key)
    if cached:
        p = Path(cached)
        if p.is_dir():
            if not require_stage_files or year is None:
                return str(p)
            req = [
                f"harvest_df_results_{year}.csv",
                f"sawmill_df_results_{year}.csv",
                f"operation_df_results_{year}.csv",
                f"const_dem_df_results_{year}.csv",
                f"EoL_df_results_{year}.csv",
                f"seq_df_results_{year}.csv",
            ]
            if all((p / fn).exists() for fn in req):
                return str(p)
        SCENARIO_FOLDER_CACHE.pop(cache_key, None)

    # helper: check year-specific stage files (used to disambiguate duplicates)
    def _missing_stage_files(folder: Path) -> list[str]:
        if not require_stage_files or year is None:
            return []
        req = [
            f"harvest_df_results_{year}.csv",
            f"sawmill_df_results_{year}.csv",
            f"operation_df_results_{year}.csv",
            f"const_dem_df_results_{year}.csv",
            f"EoL_df_results_{year}.csv",
            f"seq_df_results_{year}.csv",
        ]
        return [fn for fn in req if not (folder / fn).exists()]

    # Build candidate MODEL and scenario folder names
    model_names: list[str] = []
    scenario_names: list[str] = []
    for pref in prefixes:
        for pt in prod_tokens:
            model = f"{pref}_softwood_{pt}"
            model_names.append(model)
            scenario_names.append(f"{model}_{scenario_l}")

    # Candidate base dirs, in priority order: region-root subfolder first, then root
    base_dirs: list[Path] = []

    # If main_path itself is already NW/ or SE/, keep it first
    if root.name.casefold() in {r.casefold() for r in region_roots}:
        base_dirs.append(root)

    for rr in region_roots:
        d = _find_dir_case_insensitive(root, rr)
        if d is not None:
            base_dirs.append(d)

    base_dirs.append(root)

    # de-dupe while preserving order
    seen = set()
    base_dirs2: list[Path] = []
    for d in base_dirs:
        key = str(d.resolve()).casefold()
        if key not in seen:
            seen.add(key)
            base_dirs2.append(d)
    base_dirs = base_dirs2

    tried: list[str] = []
    found_but_missing: list[tuple[str, list[str]]] = []

    def _try_pick(folder: Path) -> Path | None:
        if not folder.is_dir():
            return None
        missing = _missing_stage_files(folder)
        if missing:
            found_but_missing.append((str(folder), missing))
            return None
        return folder

    # Try “base / model / MCS_Results / scenario” 
    for base in base_dirs:
        for model, scen in zip(model_names, scenario_names):
            # 1) base/model/MCS_Results/scen
            model_dir = base if base.name.casefold() == model.casefold() else _find_dir_case_insensitive(base, model)
            if model_dir is not None:
                mcs_dir = _find_dir_case_insensitive(model_dir, "MCS_Results")
                if mcs_dir is not None:
                    scen_dir = _find_dir_case_insensitive(mcs_dir, scen)
                    tried.append(str(Path(mcs_dir) / scen))
                    if scen_dir is not None:
                        picked = _try_pick(scen_dir)
                        if picked is not None:
                            SCENARIO_FOLDER_CACHE[cache_key] = str(picked)
                            return str(picked)

                # 2) base/model/scen (some layouts)
                scen_dir2 = _find_dir_case_insensitive(model_dir, scen)
                tried.append(str(Path(model_dir) / scen))
                if scen_dir2 is not None:
                    picked = _try_pick(scen_dir2)
                    if picked is not None:
                        SCENARIO_FOLDER_CACHE[cache_key] = str(picked)
                        return str(picked)

            # 3) base/MCS_Results/scen
            mcs_base = _find_dir_case_insensitive(base, "MCS_Results")
            if mcs_base is not None:
                scen_dir3 = _find_dir_case_insensitive(mcs_base, scen)
                tried.append(str(Path(mcs_base) / scen))
                if scen_dir3 is not None:
                    picked = _try_pick(scen_dir3)
                    if picked is not None:
                        SCENARIO_FOLDER_CACHE[cache_key] = str(picked)
                        return str(picked)

            # 4) base/scen
            scen_dir4 = _find_dir_case_insensitive(base, scen)
            tried.append(str(Path(base) / scen))
            if scen_dir4 is not None:
                picked = _try_pick(scen_dir4)
                if picked is not None:
                    SCENARIO_FOLDER_CACHE[cache_key] = str(picked)
                    return str(picked)

    # If we found candidates but they were missing files, give a better error
    if found_but_missing and (require_stage_files and year is not None):
        msg = (
            "Found scenario folder candidates, but they are missing required stage CSVs.\n"
            f"Requested: region={region_u}, product={product_u}, scenario={scenario_l}, year={year}\n"
            "Examples (first 3):\n"
        )
        for path, missing in found_but_missing[:3]:
            msg += f"  - {path}\n    missing: {missing}\n"
        raise FileNotFoundError(msg)

    raise FileNotFoundError(
        "Could not locate scenario folder.\n"
        f"main_path resolved to: {root}\n"
        f"region={region_u}, product={product_u}, scenario={scenario_l}\n"
        "Tried (first ~15):\n  - " + "\n  - ".join(tried[:15])
    )



def _build_mill_ts_path(main_path: str, region: str, product: str, scenario: str, year: int) -> str:
    return os.path.join(_scenario_folder(main_path, region, product, scenario),
                        f"mill_residue_co2_timeseries_df_results_{year}.csv")


def _build_slash_ts_path(main_path: str, region: str, product: str, scenario: str, year: int) -> str:
    return os.path.join(_scenario_folder(main_path, region, product, scenario),
                        f"slash_residue_co2_timeseries_{year}.csv")


def _build_landfill_ts_path(main_path: str, region: str, product: str, scenario: str, year: int) -> str:
    return os.path.join(_scenario_folder(main_path, region, product, scenario),
                        f"landfill_timeseries_{year}.csv")


def _maybe_read_csv(path: str, **kwargs) -> Optional[pd.DataFrame]:
    return pd.read_csv(path, **kwargs) if os.path.exists(path) else None


def load_emissions_df(main_path: str, region: str, product: str, scenario: str, year: int) -> pd.DataFrame:
    key = (region.upper(), product.upper(), scenario.lower(), int(year))
    if key in EMISSIONS_CACHE:
        return EMISSIONS_CACHE[key]

    # folder = _scenario_folder(main_path, region, product, scenario)
    folder = _scenario_folder(
    main_path, region, product, scenario,
    year=int(year),
    require_stage_files=True,
    )   

    harvest_df = _maybe_read_csv(os.path.join(folder, f"harvest_df_results_{year}.csv"), index_col=0)
    sawmill_df = _maybe_read_csv(os.path.join(folder, f"sawmill_df_results_{year}.csv"), index_col=0)
    operation_df = _maybe_read_csv(os.path.join(folder, f"operation_df_results_{year}.csv"), index_col=0)
    const_dem_df = _maybe_read_csv(os.path.join(folder, f"const_dem_df_results_{year}.csv"), index_col=0)
    EoL_df = _maybe_read_csv(os.path.join(folder, f"EoL_df_results_{year}.csv"), index_col=0)
    seq_df = _maybe_read_csv(os.path.join(folder, f"seq_df_results_{year}.csv"), index_col=0)

    if any(df is None for df in (harvest_df, sawmill_df, operation_df, const_dem_df, EoL_df, seq_df)):
        missing = [name for name, df in [
            ("harvest_df", harvest_df),
            ("sawmill_df", sawmill_df),
            ("operation_df", operation_df),
            ("const_dem_df", const_dem_df),
            ("EoL_df", EoL_df),
            ("seq_df", seq_df),
        ] if df is None]
        raise FileNotFoundError(f"Missing required MCS stage files in {folder}: {missing}")

    idx = seq_df.index
    out = pd.DataFrame(index=idx)

    out["raw_materials_emissions"] = harvest_df["harvest_ops_CO2"].astype(float) + harvest_df["fert_herb_CO2"].astype(float)
    out["raw_materials_uptake"] = (
        seq_df["Sequestered_CO2"].astype(float)
        + seq_df["Mill_residue_CO2"].astype(float)
        + seq_df["Sequestered_CO2_residue"].astype(float)
    )

    out["transportation_1_emissions"] = harvest_df["harvest_haul_CO2"].astype(float)
    out["processing_bio_emissions"] = sawmill_df.get("sawmill_biomass_CO2_biogenic", 0.0).astype(float)
    out["processing_fossil_emissions"] = sawmill_df["sawmill_ops_CO2"].astype(float)
    out["transportation_2_emissions"] = sawmill_df["sawmill_haul_CO2"].astype(float)
    out["operation_emissions"] = operation_df["total_op_process_CO2_fossil"].astype(float)
    out["transportation_3_emissions"] = operation_df["op_haul_CO2"].astype(float)
    out["construction_emissions"] = const_dem_df["construction_CO2"].astype(float)
    out["deconstruction_emissions"] = const_dem_df["deconstruction_CO2"].astype(float)
    out["transportation_4_emissions"] = 0.2 * operation_df["op_haul_CO2"].astype(float)

    scen = scenario.lower()
    if scen == "low":
        out["use_emissions"] = seq_df.get("clean_residue_bio_CO2", 0.0).astype(float)
        out["use_uptake"] = seq_df.get("clean_residue_avoided_fossil_CO2", 0.0).astype(float)
        out["EoL_1_emissions"] = EoL_df.get("EoL_3_bio_CO2", 0.0).astype(float)
        out["EoL_1_uptake"] = EoL_df.get("EoL_3_avoided_fossil_CO2", 0.0).astype(float)
        out["EoL_2_emissions"] = EoL_df.get("EoL_4_fossil_CO2", 0.0).astype(float)
    else:
        out["use_emissions"] = seq_df.get("Residue_new_CO2", 0.0).astype(float)
        out["use_CH4"] = seq_df.get("Residue_new_CH4", 0.0).astype(float)

        out["EoL_1_bio_emissions"] = EoL_df.get("EoL_1_bio_CO2", 0.0).astype(float)
        out["EoL_1_fossil_emissions"] = EoL_df.get("EoL_1_fossil_CO2", 0.0).astype(float)
        out["EoL_1_avoided_fossil_emissions"] = EoL_df.get("EoL_1_avoided_fossil_CO2", 0.0).astype(float)

        out["EoL_2_bio_emissions"] = EoL_df.get("EoL_2_bio_CO2", 0.0).astype(float)
        out["EoL_2_CH4"] = EoL_df.get("EoL_2_fossil_CH4", 0.0).astype(float)
        out["EoL_2_avoided_fossil_emissions"] = EoL_df.get("EoL_2_avoided_fossil_CO2", 0.0).astype(float)

    EMISSIONS_CACHE[key] = out
    return out


def load_ts_df(main_path: str, region: str, product: str, scenario: str, year: int, kind: str) -> Optional[pd.DataFrame]:
    key = (region.upper(), product.upper(), scenario.lower(), int(year), kind.lower())
    if key in TS_CACHE:
        return TS_CACHE[key]

    reg, prod, scen, yr, k = key

    if k == "mill":
        path = _build_mill_ts_path(main_path, reg, prod, scen, yr)
    elif k == "slash":
        path = _build_slash_ts_path(main_path, reg, prod, scen, yr)
    elif k == "landfill":
        path = _build_landfill_ts_path(main_path, reg, prod, scen, yr)
    else:
        raise ValueError("kind must be 'mill', 'slash', or 'landfill'")

    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if "simulation" in df.columns:
        df = df.set_index("simulation")
    else:
        df.index.name = "simulation"

    cols = list(df.columns)
    if k == "mill":
        rename = {}
        for c in cols:
            rename[c] = c.replace("year_", "emit_year_") if c.startswith("year_") else c
        df = df.rename(columns=rename)

        if not any(c.startswith("cred_year_") for c in df.columns):
            ks = [int(c.split("_")[-1]) for c in df.columns if c.startswith("emit_year_")]
            kmax = max(ks) if ks else 0
            for j in range(1, kmax + 1):
                cj = f"cred_year_{j}"
                if cj not in df.columns:
                    df[cj] = 0.0

    elif k == "slash":
        rename = {c: c.replace("year_", "emit_year_") for c in cols if c.startswith("year_")}
        if rename:
            df = df.rename(columns=rename)

    TS_CACHE[key] = df
    return df


# -------------------- Projections --------------------

def _load_proj_matrix(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv, index_col=0)
    try:
        df.columns = [int(c) if str(c).isdigit() else c for c in df.columns]
    except Exception:
        pass
    return df


def load_all_projections(
    proj_base: str,
    scenarios=("S1", "S2", "S3"),
    products=("CLT", "GLT"),
) -> None:
    for S in scenarios:
        for region in ("PNW", "SE"):
            for prod in products:
                fname_candidates = [
                    f"{region.lower()}_{prod}_{S}.csv",
                    f"{region.lower()}_{('Glulam' if prod.upper()=='GLT' else prod)}_{S}.csv",
                    f"{region.lower()}_{prod.lower()}_{S}.csv",
                ]
                found = None
                for fn in fname_candidates:
                    path = os.path.join(proj_base, fn)
                    if os.path.exists(path):
                        found = path
                        break
                if not found:
                    print(f"WARNING: missing projection CSV for {region}-{prod}-{S}")
                    continue
                df = _load_proj_matrix(found)
                globals()[f"{region.lower()}_{prod.lower()}_{S}"] = df


def _get_prod_df(region: str, product: str, prod_scenario: str) -> pd.DataFrame:
    region_key = region.lower()
    scen_key = prod_scenario.upper()
    product_candidates = [product, product.upper(), product.lower(), "glulam" if product.upper() == "GLT" else product]

    tried = []
    for prod_key in product_candidates:
        name = f"{region_key}_{prod_key.lower()}_{scen_key}"
        if name in globals():
            df = globals()[name].copy()
            try:
                df.columns = [int(c) if str(c).isdigit() else c for c in df.columns]
            except Exception:
                pass
            return df

        name_uc = f"{region_key}_{prod_key.upper()}_{scen_key}"
        tried.append(name_uc)
        if name_uc in globals():
            df = globals()[name_uc].copy()
            try:
                df.columns = [int(c) if str(c).isdigit() else c for c in df.columns]
            except Exception:
                pass
            return df

    raise KeyError(f"Production DataFrame not found for ({region},{product},{prod_scenario}). Tried: {tried}")


# -------------------- Multi-cohort engine --------------------

def run_multi_cohort(
    main_path: str,
    cohort_end: int = 2050,
    prod_scenario: str = "S1",
    include_regions: Tuple[str, ...] = ("PNW", "SE"),
    include_scenarios: Tuple[str, ...] = ("low", "high"),
    gwp100_ch4: float = 28.0,
    compute_audit: bool = True,
):
    earliest = 2000

    LANDIFLL_TAIL_YEARS_99_SLOW = 460
    RESIDUE_TS_YEARS = 200
    SL_MAX = 55

    end_landfill = cohort_end + SL_MAX + LANDIFLL_TAIL_YEARS_99_SLOW
    end_residue = cohort_end + SL_MAX + RESIDUE_TS_YEARS
    years_full = np.arange(earliest, max(end_landfill, end_residue) + 1, dtype=int)

    variants = []
    if "low" in include_scenarios:
        variants += [("low", 1), ("low", 2)]
    if "high" in include_scenarios:
        variants += [("high", 1), ("high", 2)]

    rep_index_source = None
    for reg in include_regions:
        try:
            rep_index_source = _get_prod_df(reg, "CLT", prod_scenario).index
            break
        except KeyError:
            continue
    if rep_index_source is None:
        raise RuntimeError("No production DF found to infer replicates.")

    reps = pd.Index(rep_index_source)
    rep_to_i = pd.Series(range(len(reps)), index=reps)
    year_to_j = pd.Series(range(len(years_full)), index=years_full)

    acc = {(sc, e): np.zeros((len(reps), len(years_full)), dtype=np.float32) for sc, e in variants}
    audit_rows = []

    for cohort in tqdm(range(2020, cohort_end + 1)):
        region_flows_by_scenario = {"low": [], "high": []}

        for region in include_regions:
            prod_CLT = _get_prod_df(region, "CLT", prod_scenario)
            prod_GLT = _get_prod_df(region, "GLT", prod_scenario)

            if "low" in include_scenarios:
                cfg_low = DcubeConfig(
                    base_year=cohort,
                    emission_scenario="low",
                    region=region,
                    production_scenario=prod_scenario,
                )
                em_CLT_low = load_emissions_df(main_path, region, "CLT", "low", cohort)
                em_GLT_low = load_emissions_df(main_path, region, "GLT", "low", cohort)

                mr_CLT = load_ts_df(main_path, region, "CLT", "low", cohort, kind="mill")
                mr_GLT = load_ts_df(main_path, region, "GLT", "low", cohort, kind="mill")

                fl_CLT_low = compute_flows_for_product(em_CLT_low, prod_CLT, cfg_low, "CLT", use_schedules={"mill": mr_CLT})["flows_long"]
                fl_GLT_low = compute_flows_for_product(em_GLT_low, prod_GLT, cfg_low, "GLT", use_schedules={"mill": mr_GLT})["flows_long"]
                region_flows_by_scenario["low"].append(combine_products_aggregated([fl_CLT_low, fl_GLT_low]))

            if "high" in include_scenarios:
                cfg_high = DcubeConfig(
                    base_year=cohort,
                    emission_scenario="high",
                    region=region,
                    production_scenario=prod_scenario,
                    gwp100_ch4=gwp100_ch4,
                )
                em_CLT_high = load_emissions_df(main_path, region, "CLT", "high", cohort)
                em_GLT_high = load_emissions_df(main_path, region, "GLT", "high", cohort)

                mr_CLT_h = load_ts_df(main_path, region, "CLT", "high", cohort, kind="mill")
                mr_GLT_h = load_ts_df(main_path, region, "GLT", "high", cohort, kind="mill")
                lf_CLT_h = load_ts_df(main_path, region, "CLT", "high", cohort, kind="landfill")
                lf_GLT_h = load_ts_df(main_path, region, "GLT", "high", cohort, kind="landfill")

                if region.upper() == "SE":
                    sl_CLT_h = load_ts_df(main_path, region, "CLT", "high", cohort, kind="slash")
                    sl_GLT_h = load_ts_df(main_path, region, "GLT", "high", cohort, kind="slash")

                    fl_CLT_high = compute_flows_for_product(
                        em_CLT_high, prod_CLT, cfg_high, "CLT",
                        use_schedules={"mill": mr_CLT_h, "slash": sl_CLT_h, "landfill": lf_CLT_h},
                    )["flows_long"]
                    fl_GLT_high = compute_flows_for_product(
                        em_GLT_high, prod_GLT, cfg_high, "GLT",
                        use_schedules={"mill": mr_GLT_h, "slash": sl_GLT_h, "landfill": lf_GLT_h},
                    )["flows_long"]
                else:
                    fl_CLT_high = compute_flows_for_product(
                        em_CLT_high, prod_CLT, cfg_high, "CLT",
                        use_schedules={"mill": mr_CLT_h, "landfill": lf_CLT_h},
                    )["flows_long"]
                    fl_GLT_high = compute_flows_for_product(
                        em_GLT_high, prod_GLT, cfg_high, "GLT",
                        use_schedules={"mill": mr_GLT_h, "landfill": lf_GLT_h},
                    )["flows_long"]

                region_flows_by_scenario["high"].append(combine_products_aggregated([fl_CLT_high, fl_GLT_high]))

        for sc in include_scenarios:
            if not region_flows_by_scenario[sc]:
                continue
            both_regions = pd.concat(region_flows_by_scenario[sc], ignore_index=True)
            for eol in (1, 2):
                variant_fl = make_eol_variant(both_regions, sc, eol)
                ny = net_by_year(variant_fl)
                ny = ny[ny["replicate"].isin(reps) & ny["year"].isin(years_full)]
                if len(ny) == 0:
                    continue
                ri = ny["replicate"].map(rep_to_i).to_numpy()
                yj = ny["year"].map(year_to_j).to_numpy()
                acc[(sc, eol)][ri, yj] += ny["net"].to_numpy(dtype=np.float32)

                if compute_audit:
                    lifetime = ny.groupby("replicate")["net"].sum()
                    audit_rows.append({
                        "cohort": cohort,
                        "scenario": sc,
                        "eol": eol,
                        "mean": float(lifetime.mean()),
                        "median": float(lifetime.median()),
                        "p05": float(lifetime.quantile(0.05)),
                        "p95": float(lifetime.quantile(0.95)),
                    })

    rows = []
    cum_arrays: Dict[Tuple[str, int], np.ndarray] = {}
    for (sc, eol), A in acc.items():
        A_cum = A.cumsum(axis=1)
        cum_arrays[(sc, eol)] = A_cum

        med = np.median(A_cum, axis=0)
        p05 = np.percentile(A_cum, 5, axis=0)
        p95 = np.percentile(A_cum, 95, axis=0)
        mean = A_cum.mean(axis=0)

        for j, yr in enumerate(years_full):
            rows.append({
                "scenario": sc,
                "eol": int(eol),
                "year": int(yr),
                "median": float(med[j]),
                "p05": float(p05[j]),
                "p95": float(p95[j]),
                "mean": float(mean[j]),
            })

    cum_stats = pd.DataFrame(rows).sort_values(["scenario", "eol", "year"]).reset_index(drop=True)
    audit_df = pd.DataFrame(audit_rows) if compute_audit else None
    return cum_stats, audit_df, years_full, cum_arrays


# -------------------- BAU mix --------------------

def compute_bau_median_mix_from_stats(
    cum_stats: pd.DataFrame,
    *,
    weights: Optional[Mapping[Tuple[str, int], float]] = None,
    normalize: bool = True,
    normalize_to_available: bool = True,
) -> pd.DataFrame:
    if weights is None:
        weights = DEFAULT_EPA_BAU_WEIGHTS

    w_all = {(str(k[0]).lower(), int(k[1])): float(max(0.0, v)) for k, v in weights.items()}
    present_pairs = set(zip(cum_stats["scenario"].astype(str), cum_stats["eol"].astype(int)))

    if normalize_to_available:
        w = {k: v for k, v in w_all.items() if k in present_pairs and v > 0}
    else:
        w = {k: v for k, v in w_all.items() if v > 0}

    years = np.sort(cum_stats["year"].unique())
    if not w:
        return pd.DataFrame({"year": years, "bau_median": np.zeros_like(years, dtype=float)})

    if normalize:
        s = sum(w.values())
        if s > 0:
            w = {k: v / s for k, v in w.items()}

    bau = pd.Series(0.0, index=years, name="bau_median")
    for (sc, eol), wt in w.items():
        sub = cum_stats[(cum_stats["scenario"] == sc) & (cum_stats["eol"] == int(eol))]
        if sub.empty:
            continue
        med = sub[["year", "median"]].set_index("year").reindex(years)["median"].ffill().fillna(0.0)
        bau = bau.add(wt * med, fill_value=0.0)

    return pd.DataFrame({"year": years, "bau_median": bau.values})


# -------------------- Bundle IO --------------------

def build_cum_plot_bundle(
    PROJ_BASE: str,
    MAIN_PATH: str,
    *,
    prod_scenarios: Sequence[str] = ("S1", "S2", "S3"),
    include_regions: Sequence[str] = ("PNW", "SE"),
    include_scenarios: Sequence[str] = ("low", "high"),
    cohort_end: int = 2120,
    gwp100_ch4: float = 28.0,
) -> Dict:
    panels = {}
    for S in prod_scenarios:
        load_all_projections(PROJ_BASE, scenarios=(S,), products=("CLT", "GLT"))
        cum_stats, _audit, years_full, cum_arrays = run_multi_cohort(
            MAIN_PATH,
            cohort_end=cohort_end,
            prod_scenario=S,
            include_regions=tuple(include_regions),
            include_scenarios=tuple(include_scenarios),
            gwp100_ch4=gwp100_ch4,
            compute_audit=False,
        )
        panels[S] = {
            "cum_stats": cum_stats.reset_index(drop=True),
            "cum_arrays": cum_arrays,
            "years_full": np.asarray(years_full, dtype=int),
        }
    return {
        "panels": panels,
        "meta": {
            "prod_scenarios": list(prod_scenarios),
            "include_regions": list(include_regions),
            "include_scenarios": list(include_scenarios),
            "cohort_end": int(cohort_end),
            "gwp100_ch4": float(gwp100_ch4),
        },
    }


def save_bundle(bundle: Dict, folder="Results/PlotBundles", name="cum_net_bundle"):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{name}__meta.json"), "w") as f:
        json.dump(bundle["meta"], f)

    for S, panel in bundle["panels"].items():
        panel_folder = os.path.join(folder, f"{name}__{S}")
        os.makedirs(panel_folder, exist_ok=True)
        panel["cum_stats"].to_parquet(os.path.join(panel_folder, "cum_stats.parquet"))
        np.save(os.path.join(panel_folder, "years_full.npy"), panel["years_full"])
        np.savez_compressed(
            os.path.join(panel_folder, "cum_arrays.npz"),
            **{f"{k[0]}__{k[1]}": v for k, v in panel["cum_arrays"].items()},
        )


def load_bundle(folder="Results/PlotBundles", name="cum_net_bundle") -> Dict:
    with open(os.path.join(folder, f"{name}__meta.json"), "r") as f:
        meta = json.load(f)

    panels = {}
    for S in meta.get("prod_scenarios", ["S1", "S2", "S3"]):
        panel_folder = os.path.join(folder, f"{name}__{S}")
        if not os.path.exists(panel_folder):
            continue
        cum_stats = pd.read_parquet(os.path.join(panel_folder, "cum_stats.parquet"))
        years_full = np.load(os.path.join(panel_folder, "years_full.npy"))
        z = np.load(os.path.join(panel_folder, "cum_arrays.npz"))
        cum_arrays = {}
        for key in z.files:
            sc, eol = key.split("__")
            cum_arrays[(sc, int(eol))] = z[key]
        panels[S] = {"cum_stats": cum_stats, "cum_arrays": cum_arrays, "years_full": years_full}

    return {"panels": panels, "meta": meta}


# -------------------- Plotting --------------------

def plot_cumulative_net_with_pdfs_on_ax(
    ax,
    cum_stats,
    cum_arrays,
    years_full,
    *,
    cut_year=2120,
    to_units="Mt",
    style="line",
    band_quantiles=(0.05, 0.95),
    face_gray=0.975,
    decade_lw=0.8,
    decade_color="0.90",
    pdf_width_years=6.0,
    right_pad_frac=0.18,
    band_alpha=0.18,
    pdf_alpha=0.18,
    grid_alpha=0.25,
    fs_axis_label=FS_AXIS_LABEL,
    fs_tick=FS_TICK,
    y_major_step=None,
    legend=False,
    title=None,
    show_end_markers=False,
    show_end_values=False,
    main_lw=2.8,
    pdf_lw=1.6,
    zero_lw=1.1,
    cut_lw=1.6,
    grid_lw=0.9,
    spine_lw=1.2,
    show_bau: bool = True,
    bau_weights: Optional[Dict[Tuple[str, int], float]] = None,
    bau_label: str = "EPA mix (BAU)",
    bau_color: str = "k",
    bau_lw: Optional[float] = None,
    bau_ls: str = "--",
    anchor_years=(2020, 2040, 2060, 2080, 2100, 2120),
):
    u = str(to_units).lower()
    unit_scale = {"t": 1.0, "kt": 1e-3, "mt": 1e-6, "gt": 1e-9}
    scale = unit_scale.get(u, 1.0)

    years_full = np.asarray(years_full, dtype=int)
    if cut_year not in set(years_full):
        raise ValueError(f"cut_year {cut_year} not in model horizon {years_full.min()}…{years_full.max()}")
    j_cut = int(np.where(years_full == cut_year)[0][0])

    def qband(A, qlo, qhi, j_end):
        lo = np.quantile(A[:, :j_end + 1], qlo, axis=0)
        md = np.quantile(A[:, :j_end + 1], 0.5, axis=0)
        hi = np.quantile(A[:, :j_end + 1], qhi, axis=0)
        return lo, md, hi

    ax.set_facecolor(str(face_gray))
    decades = list(range(int(years_full.min()), int(cut_year) + 1, 10))
    for d in decades[1:]:
        ax.axvline(d, color=decade_color, lw=decade_lw, zorder=0)

    all_line_vals = []
    all_pdf_vals = []
    plotted = []

    for sc, eol, label, color, ls in SC_ORDER:
        key = (sc, eol)
        sub = cum_stats[(cum_stats["scenario"] == sc) & (cum_stats["eol"] == eol) & (cum_stats["year"] <= cut_year)]
        if key not in cum_arrays or sub.empty:
            continue

        x = sub["year"].to_numpy()
        A = cum_arrays[key]
        lo, md, hi = qband(A, band_quantiles[0], band_quantiles[1], j_cut)

        y_med = md * scale
        y_lo = lo * scale
        y_hi = hi * scale

        if len(x):
            all_line_vals.extend([float(np.nanmin(y_lo[:len(x)])), float(np.nanmax(y_hi[:len(x)]))])

        if style == "step":
            ax.step(x, y_med[:len(x)], where="post", color=color, lw=main_lw, ls=ls, zorder=3, label=label)
            ax.fill_between(x, y_lo[:len(x)], y_hi[:len(x)], step="post", color=color, alpha=band_alpha, zorder=1)
        else:
            ax.plot(x, y_med[:len(x)], color=color, lw=main_lw, ls=ls, zorder=3, label=label)
            ax.fill_between(x, y_lo[:len(x)], y_hi[:len(x)], color=color, alpha=band_alpha, zorder=1)

        if show_end_markers:
            ax.plot([cut_year], [y_med[j_cut]], marker="o", ms=3.0, color=color, zorder=4)
        if show_end_values:
            ax.text(cut_year + 1.0, y_med[j_cut], f"{y_med[j_cut]:.3f}", va="center", fontsize=fs_tick, color=color)

        all_pdf_vals.append((A[:, j_cut] * scale).astype(float))
        plotted.append((sc, eol, label, color, ls))

    if show_bau:
        bau_df = compute_bau_median_mix_from_stats(cum_stats, weights=bau_weights)
        xb = bau_df["year"].to_numpy()
        yb = bau_df["bau_median"].to_numpy() * scale
        m = xb <= cut_year
        xb, yb = xb[m], yb[m]
        if bau_lw is None:
            bau_lw = 1.35 * main_lw
        if style == "step":
            ax.step(xb, yb, where="post", color=bau_color, lw=bau_lw, ls=bau_ls, zorder=5, label=bau_label)
        else:
            ax.plot(xb, yb, color=bau_color, lw=bau_lw, ls=bau_ls, zorder=5, label=bau_label)
        if len(yb):
            all_line_vals.extend([float(np.nanmin(yb)), float(np.nanmax(yb))])

    all_pdf_flat = np.concatenate([v for v in all_pdf_vals if v.size]) if all_pdf_vals else np.array([])
    if all_pdf_flat.size:
        qlo_val, qhi_val = np.quantile(all_pdf_flat, [0.005, 0.995])
        span = max(qhi_val - qlo_val, 1e-9)
        y_lo_pdf = qlo_val - 0.06 * span
        y_hi_pdf = qhi_val + 0.06 * span
    else:
        y_lo_pdf, y_hi_pdf = -1.0, 1.0

    y_lo_line = min(all_line_vals) if all_line_vals else -1.0
    y_hi_line = max(all_line_vals) if all_line_vals else 1.0
    ymin, ymax = min(y_lo_pdf, y_lo_line), max(y_hi_pdf, y_hi_line)

    ax.axvspan(
        cut_year,
        cut_year + pdf_width_years + right_pad_frac * pdf_width_years,
        facecolor=ax.get_facecolor(),
        edgecolor="none",
        zorder=0.5,
    )

    y_grid = np.linspace(ymin, ymax, 1400)
    dmax = 1e-12
    dens_list = []
    for samples in all_pdf_vals:
        a = samples[np.isfinite(samples)]
        dens = np.zeros_like(y_grid) if (a.size < 2 or np.allclose(a.std(), 0.0)) else gaussian_kde(a)(y_grid)
        dens_list.append(dens)
        dmax = max(dmax, float(dens.max()))

    for (_sc, _eol, _label, color, ls), dens in zip(plotted, dens_list):
        x_right = cut_year + (dens / dmax) * pdf_width_years if dmax > 0 else np.full_like(y_grid, cut_year)
        ax.plot(x_right, y_grid, color=color, lw=pdf_lw, ls=ls, zorder=4)
        ax.fill_betweenx(y_grid, cut_year, x_right, color=color, alpha=pdf_alpha, zorder=2)

    ax.axhline(0, color="k", lw=zero_lw, alpha=0.85)
    ax.axvline(cut_year, color="k", lw=cut_lw, alpha=1.0, zorder=7)
    ax.grid(True, axis="y", alpha=grid_alpha, linewidth=grid_lw)
    ax.grid(axis="x", visible=False)

    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(spine_lw)
        ax.spines[side].set_color("0.2")

    ax.tick_params(axis="both", labelsize=fs_tick, pad=1)
    if y_major_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major_step))

    ax.set_xlim(years_full.min(), cut_year + pdf_width_years + right_pad_frac * pdf_width_years)

    ax.xaxis.set_major_locator(FixedLocator(list(anchor_years)))
    ax.xaxis.set_major_formatter(FixedFormatter([str(y) for y in anchor_years]))
    ax.tick_params(axis="x", which="major", labelsize=fs_tick, length=5.5, width=1.0)

    if y_major_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_major_step))
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="y", which="major", labelsize=fs_tick, length=5.5, width=1.0)

    if title:
        ax.set_title(title, fontsize=fs_axis_label)
    if legend:
        ax.legend(ncols=2, frameon=False, fontsize=max(fs_tick - 1, 12), loc="lower left")

    return ymin, ymax


def plot_two_panel_from_bundle(
    bundle: Dict,
    *,
    cut_year: int = 2120,
    to_units: str = "Mt",
    style: str = "line",
    fig_size=(12, 4),
    dpi: int = 300,
    pdf_width_years: float = 6.0,
    right_pad_frac: float = 0.18,
    band_alpha: float = 0.18,
    pdf_alpha: float = 0.18,
    grid_alpha: float = 0.25,
    fs_axis_label: int = FS_AXIS_LABEL,
    fs_tick: int = FS_TICK,
    main_lw=2.8,
    pdf_lw=1.6,
    zero_lw=1.1,
    cut_lw=1.6,
    grid_lw=0.9,
    spine_lw=1.2,
    decade_lw: float = 0.8,
    panel_letters: Optional[Sequence[str]] = None,
    letter_fs: int = FS_TICK,
    letter_xy: Tuple[float, float] = (0.985, 0.965),
    y_major_step: float | None = None,
    save_path: Optional[str] = None,
    show_bau: bool = True,
    bau_weights: Optional[Mapping[Tuple[str, int], float]] = None,
    bau_label: str = "BAU",
    bau_color: str = "k",
    bau_lw: Optional[float] = None,
    bau_ls: str = "--",
    anchor_years=(2020, 2040, 2060, 2080, 2100, 2120),
):
    panels = bundle["panels"]
    keys = [k for k in ("S1", "S3") if k in panels] or list(panels.keys())
    if panel_letters is None:
        panel_letters = ["A", "B"][:len(keys)]

    u = str(to_units).lower()
    unit_label = {"t": "t", "kt": "kt", "mt": "Mt", "gt": "Gt"}.get(u, to_units)

    fig, axes = plt.subplots(1, len(keys), figsize=fig_size, sharey=True, dpi=dpi, constrained_layout=True)
    if len(keys) == 1:
        axes = [axes]

    global_lo, global_hi = [], []
    for i, S in enumerate(keys):
        p = panels[S]
        ymin, ymax = plot_cumulative_net_with_pdfs_on_ax(
            axes[i],
            p["cum_stats"],
            p["cum_arrays"],
            p["years_full"],
            cut_year=cut_year,
            to_units=to_units,
            style=style,
            pdf_width_years=pdf_width_years,
            right_pad_frac=right_pad_frac,
            band_alpha=band_alpha,
            pdf_alpha=pdf_alpha,
            grid_alpha=grid_alpha,
            fs_axis_label=fs_axis_label,
            fs_tick=fs_tick,
            y_major_step=y_major_step,
            legend=False,
            title=None,
            main_lw=main_lw,
            pdf_lw=pdf_lw,
            zero_lw=zero_lw,
            cut_lw=cut_lw,
            grid_lw=grid_lw,
            spine_lw=spine_lw,
            decade_lw=decade_lw,
            show_bau=show_bau,
            bau_weights=bau_weights,
            bau_label=bau_label,
            bau_color=bau_color,
            bau_lw=bau_lw,
            bau_ls=bau_ls,
            anchor_years=anchor_years,
        )

        if i > 0:
            axes[i].set_ylabel("")

        if i < len(panel_letters) and panel_letters[i]:
            axes[i].text(
                *letter_xy,
                panel_letters[i],
                transform=axes[i].transAxes,
                ha="right",
                va="top",
                fontsize=letter_fs,
                fontweight="bold",
            )

        global_lo.append(ymin)
        global_hi.append(ymax)

    y_min, y_max = min(global_lo), max(global_hi)
    for ax in axes:
        ax.set_ylim(y_min - 0.35, 0)

    axes[0].set_ylabel(rf"Cumulative Carbon Storage ({unit_label} CO$_2$e)", fontsize=fs_axis_label)

    handles = [Line2D([0], [0], color=c, ls=ls, lw=main_lw) for (_sc, _eol, _lab, c, ls) in SC_ORDER]
    labels = [lab for (_sc, _eol, lab, _c, _ls) in SC_ORDER]
    if show_bau:
        handles.append(Line2D([0], [0], color=bau_color, ls=bau_ls, lw=(bau_lw or 1.35 * main_lw)))
        labels.append(bau_label)

    fig.legend(
        handles,
        labels,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        frameon=False,
        handlelength=2.8,
        handletextpad=0.8,
        borderaxespad=0.0,
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    return fig, axes


# -------------------- Future-only uptake patch --------------------

def _total_uptake_by_rep_for_cohort(
    MAIN_PATH: str,
    prod_scenario: str,
    year: int,
    regions: Sequence[str] = ("PNW", "SE"),
) -> pd.Series:
    pieces = []
    for region in regions:
        prod_CLT = _get_prod_df(region, "CLT", prod_scenario)
        prod_GLT = _get_prod_df(region, "GLT", prod_scenario)

        cfg = DcubeConfig(base_year=year, emission_scenario="low", region=region, production_scenario=prod_scenario)

        em_CLT = load_emissions_df(MAIN_PATH, region, "CLT", "low", year)
        em_GLT = load_emissions_df(MAIN_PATH, region, "GLT", "low", year)

        mr_CLT = load_ts_df(MAIN_PATH, region, "CLT", "low", year, kind="mill")
        mr_GLT = load_ts_df(MAIN_PATH, region, "GLT", "low", year, kind="mill")

        fl_CLT = compute_flows_for_product(em_CLT, prod_CLT, cfg, "CLT", use_schedules={"mill": mr_CLT})["flows_long"]
        fl_GLT = compute_flows_for_product(em_GLT, prod_GLT, cfg, "GLT", use_schedules={"mill": mr_GLT})["flows_long"]

        for fl in (fl_CLT, fl_GLT):
            rm = fl[(fl["stage"] == "raw_materials") & (fl["metric"] == "uptake")]
            pieces.append(rm.groupby("replicate")["value"].sum())

    tot = sum(pieces).sort_index()
    tot.name = f"uptake_{year}"
    return tot


def _build_future_only_uptake_matrix(
    uptake_tot_2120: pd.Series,
    years_full: np.ndarray,
    last_real_cohort: int = 2120,
    future_last: int = 2150,
    growth_years: int = 21,
    growth_model: str = "richards",
    growth_params: Optional[Mapping[str, float]] = None,
) -> np.ndarray:
    reps = uptake_tot_2120.index
    R, Y = len(reps), len(years_full)
    year_to_j = pd.Series(range(Y), index=years_full)
    w = _growth_weights(growth_years, growth_model, growth_params)

    B = np.zeros((R, Y), dtype=np.float64)
    base = int(last_real_cohort)
    u = uptake_tot_2120.reindex(reps).fillna(0.0).to_numpy()

    for coh in range(base + 1, int(future_last) + 1):
        start = coh - growth_years + 1
        for k, wk in enumerate(w):
            yr = start + k
            if yr > base:
                continue
            if yr not in year_to_j:
                continue
            j = int(year_to_j[yr])
            B[:, j] += wk * u

    return B


def extend_bundle_with_future_uptake(
    bundle: Dict,
    MAIN_PATH: str,
    *,
    last_real_cohort: int = 2120,
    future_last: int = 2150,
    growth_years: int = 21,
    growth_model: str = "richards",
    growth_params: Optional[Mapping[str, float]] = None,
    save_as: Optional[str] = None,
) -> Tuple[Dict, Dict[str, pd.DataFrame]]:
    diags: Dict[str, pd.DataFrame] = {}
    panels = bundle["panels"]
    meta = bundle.get("meta", {})
    regions = tuple(meta.get("include_regions", ("PNW", "SE")))
    prod_scenarios = tuple(meta.get("prod_scenarios", panels.keys()))

    for S in tqdm(prod_scenarios):
        if S not in panels:
            continue
        p = panels[S]
        years_full = p["years_full"]

        uptake_2120 = _total_uptake_by_rep_for_cohort(MAIN_PATH, S, last_real_cohort, regions)
        B = _build_future_only_uptake_matrix(
            uptake_2120,
            years_full,
            last_real_cohort=last_real_cohort,
            future_last=future_last,
            growth_years=growth_years,
            growth_model=growth_model,
            growth_params=growth_params,
        )

        add_cum = -np.cumsum(B.astype(np.float32), axis=1)

        new_arrays = {}
        rows = []
        for key, A_cum in p["cum_arrays"].items():
            A_new = A_cum + add_cum
            new_arrays[key] = A_new

            med = np.median(A_new, axis=0)
            p05 = np.percentile(A_new, 5, axis=0)
            p95 = np.percentile(A_new, 95, axis=0)
            mean = A_new.mean(axis=0)

            sc, eol = key
            for j, yr in enumerate(years_full):
                rows.append({
                    "scenario": sc,
                    "eol": int(eol),
                    "year": int(yr),
                    "median": float(med[j]),
                    "p05": float(p05[j]),
                    "p95": float(p95[j]),
                    "mean": float(mean[j]),
                })

        p["cum_arrays"] = new_arrays
        p["cum_stats"] = pd.DataFrame(rows).sort_values(["scenario", "eol", "year"]).reset_index(drop=True)

        w = _growth_weights(growth_years, growth_model, growth_params)
        yrs_2120 = list(range(last_real_cohort - growth_years + 1, last_real_cohort + 1))
        u_by_year = pd.DataFrame({
            "year": yrs_2120,
            "uptake_2120_sum_t": (uptake_2120.sum() * pd.Series(w, index=yrs_2120)).values,
        })
        add_by_year = pd.DataFrame({
            "year": years_full,
            "added_future_uptake_sum_t": B.sum(axis=0),
        })

        check = (pd.DataFrame({"year": range(2100, last_real_cohort + 1)})
                 .merge(u_by_year, on="year", how="left")
                 .merge(add_by_year, on="year", how="left")
                 .fillna(0.0))

        diags[S] = check

    if save_as:
        save_bundle(bundle, folder="Results/PlotBundles", name=save_as)

    return bundle, diags

###############################################################################
###############################################################################
# Dynamic Stats
###############################################################################
###############################################################################
import numpy as np
import pandas as pd

DEFAULT_EPA_BAU_WEIGHTS = {
    ("high", 2): 0.7,   # landfill
    ("high", 1): 0.3,   # combustion
    ("low",  2): 0.0,   # reuse
    ("low",  1): 0.0,   # pyrolysis/biochar
}


def _build_bau_array(cum_arrays, weights=None):
    w = (weights or DEFAULT_EPA_BAU_WEIGHTS).copy()
    w = {k: float(v) for k, v in w.items() if v > 0 and k in cum_arrays}
    if not w:
        return None

    s = sum(w.values())
    if s <= 0:
        return None

    keys = list(w.keys())
    A = np.zeros_like(cum_arrays[keys[0]], dtype=np.float64)
    for k, v in w.items():
        A += (v / s) * cum_arrays[k].astype(np.float64)
    return A


def compute_case_deltas(
    bundle,
    cut_year,
    *,
    percentiles=("median", "p05", "p95"),
    to_units="t",
    reference=("low", 2),
    variants=None,
    include_bau=False,
    bau_weights=None,
    eps=1e-12,
):
    if variants is None:
        variants = [("low", 1), ("high", 2), ("high", 1)]

    unit_scale = {"t": 1.0, "kt": 1e-3, "mt": 1e-6, "gt": 1e-9}
    u = str(to_units).strip().lower()
    scale = unit_scale.get(u, 1.0)
    units = {"t": "t", "kt": "kt", "mt": "Mt", "gt": "Gt"}.get(u, to_units)

    labels = {
        ("low", 2): "reuse",
        ("low", 1): "biochar",
        ("high", 2): "landfill",
        ("high", 1): "combustion",
        ("bau", 0): "BAU",
    }

    def _stat_label(p):
        if isinstance(p, str):
            return p.strip().lower()
        q = float(p)
        return f"q{int(round(100*q)):02d}"

    def _value_from_array(A, j, p):
        x = A[:, j]
        if isinstance(p, str):
            s = p.strip().lower()
            if s in ("median", "p50"):
                return float(np.nanquantile(x, 0.5))
            if s in ("p05", "p5"):
                return float(np.nanquantile(x, 0.05))
            if s == "p95":
                return float(np.nanquantile(x, 0.95))
            if s == "mean":
                return float(np.nanmean(x))
            raise ValueError(f"Unknown stat '{p}'")
        q = float(p)
        if not (0.0 < q < 1.0):
            raise ValueError(f"Quantile must be in (0,1), got {q}")
        return float(np.nanquantile(x, q))

    rows = []
    for panel_name, panel in (bundle.get("panels", {}) or {}).items():
        years_full = np.asarray(panel["years_full"], dtype=int)
        idx = np.where(years_full == int(cut_year))[0]
        if idx.size == 0:
            raise ValueError(
                f"cut_year {cut_year} not in horizon [{years_full.min()}..{years_full.max()}] for panel {panel_name}"
            )
        j = int(idx[0])

        cum_arrays = dict(panel["cum_arrays"])

        if include_bau:
            A_bau = _build_bau_array(cum_arrays, bau_weights)
            if A_bau is not None:
                cum_arrays[("bau", 0)] = A_bau
                variants_use = list(variants) + [("bau", 0)]
            else:
                variants_use = list(variants)
        else:
            variants_use = list(variants)

        ref_key = (str(reference[0]).lower(), int(reference[1]))
        if ref_key not in cum_arrays:
            raise KeyError(f"Reference {ref_key} missing in panel {panel_name}")

        for p in percentiles:
            ref_t = _value_from_array(cum_arrays[ref_key], j, p)
            ref_val = ref_t * scale

            for case_key in variants_use:
                case_key = (str(case_key[0]).lower(), int(case_key[1]))
                A = cum_arrays.get(case_key)
                if A is None:
                    continue

                case_t = _value_from_array(A, j, p)
                case_val = case_t * scale

                abs_delta = case_val - ref_val
                denom = abs(ref_val)
                rel_delta = abs_delta / denom if (np.isfinite(denom) and denom > eps) else np.nan

                rows.append(
                    {
                        "panel": panel_name,
                        "cut_year": int(cut_year),
                        "stat": _stat_label(p),
                        "variant": labels.get(case_key, f"{case_key[0]}__{case_key[1]}"),
                        "variant_key": f"{case_key[0]}__{case_key[1]}",
                        "ref_value": ref_val,
                        "case_value": case_val,
                        "abs_delta": abs_delta,
                        "rel_delta": rel_delta,
                        "units": units,
                    }
                )

    out = pd.DataFrame(rows)
    col_order = [
        "panel",
        "cut_year",
        "variant",
        "variant_key",
        "stat",
        "ref_value",
        "case_value",
        "abs_delta",
        "rel_delta",
        "units",
    ]
    return out[col_order].sort_values(["panel", "variant", "stat"]).reset_index(drop=True)

#%%
###############################################################################
###############################################################################
# Lever Analysis
###############################################################################
###############################################################################

from typing import Dict, Tuple, Optional, Sequence, Union, List

import os
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================================
# Internal helpers used by bundle construction
# =============================================================================

def _as_series(x, like_index: pd.Index) -> pd.Series:
    """Coerce x into a float Series aligned to like_index."""
    s = x if isinstance(x, pd.Series) else pd.Series(x, index=like_index)
    return s.reindex(like_index).astype(float).fillna(0.0)


def _net_from_ts(
    ts_df: Optional[pd.DataFrame],
    prod_series: pd.Series,
    unit_in: str,
    unit_out: str,
    *,
    emit_prefix: str = "emit_year_",
    cred_prefix: str = "cred_year_",
    start_year: int,
) -> pd.DataFrame:
    """
    Scale a TS (per-rep) to production & units; return long [replicate, year, net] where:
      net(year) = emit(year) - cred(year)
    """
    if ts_df is None or ts_df.empty:
        return pd.DataFrame(columns=["replicate", "year", "net"])

    ts = _scale_ts_cols_to_prod_and_tonnes(ts_df, prod_series, unit_in, unit_out)

    ks = sorted(int(c.split("_")[-1]) for c in ts.columns if c.startswith(emit_prefix))
    rows: List[pd.DataFrame] = []
    for k in ks:
        y = int(start_year + (k - 1))
        emit_k = ts.get(f"{emit_prefix}{k}", 0.0)
        cred_k = ts.get(f"{cred_prefix}{k}", 0.0)
        net = _as_series(emit_k, ts.index) - _as_series(cred_k, ts.index)
        rows.append(pd.DataFrame({"replicate": net.index.values, "year": y, "net": net.values}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["replicate", "year", "net"])


def _accumulate_long(long_df: pd.DataFrame, arr: np.ndarray, rep_to_i: pd.Series, year_to_j: pd.Series) -> None:
    """Accumulate long-form rows into arr[replicate_i, year_j] in-place."""
    if long_df is None or long_df.empty:
        return

    ri = long_df["replicate"].map(rep_to_i).to_numpy()
    yj = long_df["year"].map(year_to_j).to_numpy()
    v = long_df["net"].to_numpy(dtype=np.float32)

    m = (~np.isnan(ri)) & (~np.isnan(yj))
    if not np.any(m):
        return
    ri = ri[m].astype(int)
    yj = yj[m].astype(int)
    v = v[m]

    np.add.at(arr, (ri, yj), v)


def _long_from_flows(fl: pd.DataFrame, stages: Sequence[str], metrics: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """
    Filter flows_long to specific stages (and optional metrics) and return
    [replicate, year, net]. If net isn't present, construct it using a metric sign map.
    """
    sub = fl[fl["stage"].isin(stages)]
    if metrics is not None:
        sub = sub[sub["metric"].isin(metrics)]

    if "net" not in sub.columns:
        # positive = release; negative = uptake/credit
        w = {"fossil": 1.0, "biogenic": 1.0, "uptake": -1.0, "energy_credit": -1.0}
        sub = sub.assign(net=sub["value"] * sub["metric"].map(w))

    return sub[["replicate", "year", "net"]].copy()


# =============================================================================
# Bundle build / save / load / summarize
# =============================================================================

def build_lever_bundle(
    PROJ_BASE: str,
    MAIN_PATH: str,
    *,
    prod_scenarios: Sequence[str] = ("S1", "S2", "S3"),
    include_regions: Sequence[str] = ("PNW", "SE"),
    include_scenarios: Sequence[str] = ("low", "high"),
    cohort_end: int = 2120,
    gwp100_ch4: float = 28.0,
) -> Dict:
    """
    Build the lever bundle used by the waterfall plot and lever stats.

    Returns:
      bundle["panels"][S]["lever_arrays"][(scenario, eol, lever)] -> np.ndarray [n_reps, n_years] (tonnes)
      bundle["panels"][S]["years_full"]  -> np.ndarray of modeled years
      bundle["panels"][S]["rep_ids"]     -> list of replicate labels
    """
    earliest = 2000
    LANDIFLL_TAIL_YEARS_99_SLOW = 460
    RESIDUE_TS_YEARS = 200
    SL_MAX = 55

    end_landfill = cohort_end + SL_MAX + LANDIFLL_TAIL_YEARS_99_SLOW
    end_residue = cohort_end + SL_MAX + RESIDUE_TS_YEARS

    years_full = np.arange(earliest, max(end_landfill, end_residue) + 1, dtype=int)
    year_to_j = pd.Series(range(len(years_full)), index=years_full)

    panels: Dict[str, Dict] = {}

    for S in prod_scenarios:
        load_all_projections(PROJ_BASE, scenarios=(S,), products=("CLT", "GLT"))

        rep_index_source = None
        for reg in include_regions:
            try:
                rep_index_source = _get_prod_df(reg, "CLT", S).index
                break
            except KeyError:
                continue
        if rep_index_source is None:
            raise RuntimeError(f"No production DF found to infer replicates for {S}.")

        reps = pd.Index(rep_index_source)
        rep_to_i = pd.Series(range(len(reps)), index=reps)

        lever_arrays: Dict[Tuple[str, int, str], np.ndarray] = {}

        def _arr(sc: str, eol: int, lever: str) -> np.ndarray:
            key = (sc, int(eol), lever)
            if key not in lever_arrays:
                lever_arrays[key] = np.zeros((len(reps), len(years_full)), dtype=np.float32)
            return lever_arrays[key]

        for cohort in tqdm(range(2020, cohort_end + 1), desc=f"Build levers {S}"):
            y0 = int(cohort)

            for region in include_regions:
                prod_CLT = _get_prod_df(region, "CLT", S)
                prod_GLT = _get_prod_df(region, "GLT", S)

                prod_CLT_col = prod_CLT[y0] if y0 in prod_CLT.columns else prod_CLT[str(y0)]
                prod_GLT_col = prod_GLT[y0] if y0 in prod_GLT.columns else prod_GLT[str(y0)]

                cfg_unit_in, cfg_unit_out = "kg", "t"

                # ---------- LOW (LEP) ----------
                if "low" in include_scenarios:
                    em_CLT_low = load_emissions_df(MAIN_PATH, region, "CLT", "low", y0)
                    em_GLT_low = load_emissions_df(MAIN_PATH, region, "GLT", "low", y0)
                    mr_CLT_low = load_ts_df(MAIN_PATH, region, "CLT", "low", y0, kind="mill")
                    mr_GLT_low = load_ts_df(MAIN_PATH, region, "GLT", "low", y0, kind="mill")

                    fl_CLT_low = compute_flows_for_product(
                        em_CLT_low,
                        prod_CLT,
                        DcubeConfig(base_year=y0, emission_scenario="low", region=region, production_scenario=S),
                        "CLT",
                        use_schedules={"mill": mr_CLT_low},
                    )["flows_long"]

                    fl_GLT_low = compute_flows_for_product(
                        em_GLT_low,
                        prod_GLT,
                        DcubeConfig(base_year=y0, emission_scenario="low", region=region, production_scenario=S),
                        "GLT",
                        use_schedules={"mill": mr_GLT_low},
                    )["flows_long"]

                    fl_low = combine_products_aggregated([fl_CLT_low, fl_GLT_low])

                    uptake_long = _long_from_flows(fl_low, stages=["raw_materials"], metrics=["uptake"])
                    _accumulate_long(uptake_long, _arr("low", 1, "uptake"), rep_to_i, year_to_j)
                    _accumulate_long(uptake_long, _arr("low", 2, "uptake"), rep_to_i, year_to_j)

                    cols = ["use_emissions", "use_uptake"]
                    eC = _scale_to_prod_and_tonnes(em_CLT_low[cols].copy(), prod_CLT_col, cols, True, cfg_unit_in, cfg_unit_out)
                    eG = _scale_to_prod_and_tonnes(em_GLT_low[cols].copy(), prod_GLT_col, cols, True, cfg_unit_in, cfg_unit_out)
                    clean_net = (eC["use_emissions"] - eC["use_uptake"] + eG["use_emissions"] - eG["use_uptake"])
                    slash_low_long = pd.DataFrame({"replicate": clean_net.index.values, "year": y0, "net": clean_net.values})
                    _accumulate_long(slash_low_long, _arr("low", 1, "slash"), rep_to_i, year_to_j)
                    _accumulate_long(slash_low_long, _arr("low", 2, "slash"), rep_to_i, year_to_j)

                    mr_CLT_long = _net_from_ts(mr_CLT_low, prod_CLT_col, cfg_unit_in, cfg_unit_out, start_year=y0)
                    mr_GLT_long = _net_from_ts(mr_GLT_low, prod_GLT_col, cfg_unit_in, cfg_unit_out, start_year=y0)
                    mr_low_long = pd.concat([mr_CLT_long, mr_GLT_long], ignore_index=True)
                    _accumulate_long(mr_low_long, _arr("low", 1, "mill_residue"), rep_to_i, year_to_j)
                    _accumulate_long(mr_low_long, _arr("low", 2, "mill_residue"), rep_to_i, year_to_j)

                    proc_stages = [
                        "raw_materials","transportation_1","processing","transportation_2",
                        "manufacture","transportation_3","construction","deconstruction","transportation_4",
                    ]
                    proc = fl_low[fl_low["stage"].isin(proc_stages)].copy()
                    # Fossil-only process+logistics emissions (exclude uptake & any credits)
                    proc = proc[proc["metric"] == "fossil"]
                    proc_long = proc[["replicate", "year", "net"]]
                    _accumulate_long(proc_long, _arr("low", 1, "process_logistics"), rep_to_i, year_to_j)
                    _accumulate_long(proc_long, _arr("low", 2, "process_logistics"), rep_to_i, year_to_j)

                    _accumulate_long(_long_from_flows(fl_low, stages=["EoL_1_biochar"]), _arr("low", 1, "biochar"), rep_to_i, year_to_j)
                    _accumulate_long(_long_from_flows(fl_low, stages=["EoL_2_reuse"]), _arr("low", 2, "reuse"), rep_to_i, year_to_j)

                # ---------- HIGH (HEP) ----------
                if "high" in include_scenarios:
                    em_CLT_high = load_emissions_df(MAIN_PATH, region, "CLT", "high", y0)
                    em_GLT_high = load_emissions_df(MAIN_PATH, region, "GLT", "high", y0)
                    mr_CLT_high = load_ts_df(MAIN_PATH, region, "CLT", "high", y0, kind="mill")
                    mr_GLT_high = load_ts_df(MAIN_PATH, region, "GLT", "high", y0, kind="mill")
                    lf_CLT_high = load_ts_df(MAIN_PATH, region, "CLT", "high", y0, kind="landfill")
                    lf_GLT_high = load_ts_df(MAIN_PATH, region, "GLT", "high", y0, kind="landfill")

                    sl_CLT_high = load_ts_df(MAIN_PATH, region, "CLT", "high", y0, kind="slash") if region.upper() == "SE" else None
                    sl_GLT_high = load_ts_df(MAIN_PATH, region, "GLT", "high", y0, kind="slash") if region.upper() == "SE" else None

                    fl_CLT_high = compute_flows_for_product(
                        em_CLT_high,
                        prod_CLT,
                        DcubeConfig(base_year=y0, emission_scenario="high", region=region, production_scenario=S, gwp100_ch4=gwp100_ch4),
                        "CLT",
                        use_schedules={"mill": mr_CLT_high, "slash": sl_CLT_high, "landfill": lf_CLT_high},
                    )["flows_long"]

                    fl_GLT_high = compute_flows_for_product(
                        em_GLT_high,
                        prod_GLT,
                        DcubeConfig(base_year=y0, emission_scenario="high", region=region, production_scenario=S, gwp100_ch4=gwp100_ch4),
                        "GLT",
                        use_schedules={"mill": mr_GLT_high, "slash": sl_GLT_high, "landfill": lf_GLT_high},
                    )["flows_long"]

                    fl_high = combine_products_aggregated([fl_CLT_high, fl_GLT_high])

                    uptake_long = _long_from_flows(fl_high, stages=["raw_materials"], metrics=["uptake"])
                    _accumulate_long(uptake_long, _arr("high", 1, "uptake"), rep_to_i, year_to_j)
                    _accumulate_long(uptake_long, _arr("high", 2, "uptake"), rep_to_i, year_to_j)

                    if region.upper() == "SE":
                        se_sl_CLT = _net_from_ts(sl_CLT_high, prod_CLT_col, cfg_unit_in, cfg_unit_out,
                                                 emit_prefix="emit_year_", cred_prefix="nope_", start_year=y0)
                        se_sl_GLT = _net_from_ts(sl_GLT_high, prod_GLT_col, cfg_unit_in, cfg_unit_out,
                                                 emit_prefix="emit_year_", cred_prefix="nope_", start_year=y0)
                        slash_high_long = pd.concat([se_sl_CLT, se_sl_GLT], ignore_index=True)
                    else:
                        cols = ["use_emissions", "use_CH4"]
                        eC = _scale_to_prod_and_tonnes(em_CLT_high[cols].copy(), prod_CLT_col, cols, True, cfg_unit_in, cfg_unit_out)
                        eG = _scale_to_prod_and_tonnes(em_GLT_high[cols].copy(), prod_GLT_col, cols, True, cfg_unit_in, cfg_unit_out)
                        pile = eC["use_emissions"] + eG["use_emissions"] + gwp100_ch4 * (eC["use_CH4"] + eG["use_CH4"])
                        slash_high_long = pd.DataFrame({"replicate": pile.index.values, "year": y0, "net": pile.values})

                    _accumulate_long(slash_high_long, _arr("high", 1, "slash"), rep_to_i, year_to_j)
                    _accumulate_long(slash_high_long, _arr("high", 2, "slash"), rep_to_i, year_to_j)

                    mr_CLT_long = _net_from_ts(mr_CLT_high, prod_CLT_col, cfg_unit_in, cfg_unit_out, start_year=y0)
                    mr_GLT_long = _net_from_ts(mr_GLT_high, prod_GLT_col, cfg_unit_in, cfg_unit_out, start_year=y0)
                    mr_high_long = pd.concat([mr_CLT_long, mr_GLT_long], ignore_index=True)
                    _accumulate_long(mr_high_long, _arr("high", 1, "mill_residue"), rep_to_i, year_to_j)
                    _accumulate_long(mr_high_long, _arr("high", 2, "mill_residue"), rep_to_i, year_to_j)

                    proc_stages = [
                        "raw_materials","transportation_1","processing","transportation_2",
                        "manufacture","transportation_3","construction","deconstruction","transportation_4",
                    ]
                    proc = fl_high[fl_high["stage"].isin(proc_stages)].copy()
                    # Fossil-only process+logistics emissions (exclude uptake & any credits)
                    proc = proc[proc["metric"] == "fossil"]
                    proc_long = proc[["replicate", "year", "net"]]
                    _accumulate_long(proc_long, _arr("high", 1, "process_logistics"), rep_to_i, year_to_j)
                    _accumulate_long(proc_long, _arr("high", 2, "process_logistics"), rep_to_i, year_to_j)

                    _accumulate_long(_long_from_flows(fl_high, stages=["EoL_1_combustion"]), _arr("high", 1, "combustion"), rep_to_i, year_to_j)
                    _accumulate_long(_long_from_flows(fl_high, stages=["EoL_2_landfill"]), _arr("high", 2, "landfill"), rep_to_i, year_to_j)

        panels[S] = {"years_full": years_full, "rep_ids": list(reps), "lever_arrays": lever_arrays}

    meta = {
        "prod_scenarios": list(prod_scenarios),
        "include_regions": list(include_regions),
        "include_scenarios": list(include_scenarios),
        "cohort_end": int(cohort_end),
        "gwp100_ch4": float(gwp100_ch4),
        "note": "Slash explicit; MR from TS; process fossil only; units = tonnes (t) in arrays.",
    }
    return {"panels": panels, "meta": meta}


def save_lever_bundle(bundle: Dict, folder: str = "Results/PlotBundles", name: str = "lever_bundle") -> None:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, f"{name}__meta.json"), "w") as f:
        json.dump(bundle["meta"], f)

    for S, panel in bundle["panels"].items():
        pathS = os.path.join(folder, f"{name}__{S}")
        os.makedirs(pathS, exist_ok=True)
        np.save(os.path.join(pathS, "years_full.npy"), np.asarray(panel["years_full"], dtype=int))
        np.savez_compressed(
            os.path.join(pathS, "lever_arrays.npz"),
            **{f"{sc}__{eol}__{lev}": arr for (sc, eol, lev), arr in panel["lever_arrays"].items()},
        )
        pd.Series(panel.get("rep_ids", [])).to_csv(os.path.join(pathS, "rep_ids.csv"), index=False, header=False)


def load_lever_bundle(folder: str = "Results/PlotBundles", name: str = "lever_bundle") -> Dict:
    with open(os.path.join(folder, f"{name}__meta.json"), "r") as f:
        meta = json.load(f)

    panels: Dict[str, Dict] = {}
    for S in meta.get("prod_scenarios", []):
        pathS = os.path.join(folder, f"{name}__{S}")
        if not os.path.exists(pathS):
            continue

        years_full = np.load(os.path.join(pathS, "years_full.npy"))
        rep_path = os.path.join(pathS, "rep_ids.csv")
        rep_ids = list(pd.read_csv(rep_path, header=None)[0]) if os.path.exists(rep_path) else None

        z = np.load(os.path.join(pathS, "lever_arrays.npz"))
        lever_arrays: Dict[Tuple[str, int, str], np.ndarray] = {}
        for k in z.files:
            sc, eol, lev = k.split("__")
            lever_arrays[(sc, int(eol), lev)] = z[k]

        panels[S] = {"years_full": years_full, "rep_ids": rep_ids, "lever_arrays": lever_arrays}

    return {"panels": panels, "meta": meta}


def summarize_lever_bundle(
    bundle: Dict,
    cut_year: int,
    *,
    to_units: str = "Mt",
    stats: Sequence[str] = ("mean", "median", "p05", "p95"),
) -> pd.DataFrame:
    u = str(to_units).lower()
    scale = 1.0
    if u == "kt":
        scale = 1e-3
    elif u == "mt":
        scale = 1e-6
    elif u == "gt":
        scale = 1e-9
    units = {"t": "t", "kt": "kt", "mt": "Mt", "gt": "Gt"}.get(u, to_units)

    rows = []
    for S, panel in bundle["panels"].items():
        years_full = np.asarray(panel["years_full"], dtype=int)
        if cut_year not in set(years_full):
            raise ValueError(f"cut_year {cut_year} not in horizon {years_full.min()}–{years_full.max()} for panel {S}")
        j_end = int(np.where(years_full == cut_year)[0][0])

        for (sc, eol, lev), A in panel["lever_arrays"].items():
            s = A[:, : j_end + 1].sum(axis=1)
            if lev == "uptake":
                s = -s
            s = s * scale

            vals = {
                "mean": float(np.nanmean(s)),
                "median": float(np.nanmedian(s)),
                "p05": float(np.nanpercentile(s, 5)),
                "p95": float(np.nanpercentile(s, 95)),
            }
            for st in stats:
                rows.append({"panel": S, "scenario": sc, "eol": int(eol), "lever": lev, "stat": st, "value": vals[st], "units": units})

    return pd.DataFrame(rows).sort_values(["panel","scenario","eol","lever","stat"]).reset_index(drop=True)


def patch_process_logistics_in_bundle(
    bundle: Dict,
    *,
    force: bool = False,
    negative_tol: float = -1e-12,
) -> Tuple[Dict, float]:
 
    max_delta = 0.0
    for _, panel in bundle.get("panels", {}).items():
        lever_arrays = panel.get("lever_arrays", {})
        for sc in ("low", "high"):
            for eol in (1, 2):
                key_proc = (sc, eol, "process_logistics")
                key_upt = (sc, eol, "uptake")
                if key_proc not in lever_arrays or key_upt not in lever_arrays:
                    continue

                A_proc = lever_arrays[key_proc]
                A_upt = lever_arrays[key_upt]

                if not force:
                    # Fossil-only process/logistics should never be negative; negativity is a strong
                    # signal that uptake got included by accident.
                    if float(np.nanmin(A_proc)) >= negative_tol:
                        continue

                new_proc = A_proc - A_upt
                max_delta = max(max_delta, float(np.nanmax(np.abs(new_proc - A_proc))))
                lever_arrays[key_proc] = new_proc

    return bundle, max_delta
def _j_end(years_full: np.ndarray, cut_year: int) -> int:
    idx = np.where(np.asarray(years_full, dtype=int) == int(cut_year))[0]
    if len(idx) == 0:
        raise ValueError(f"cut_year {cut_year} not in horizon {years_full.min()}–{years_full.max()}")
    return int(idx[0])


def _sum_to_year(panel: Dict, scenario: str, eol: int, lever: str, cut_year: int) -> np.ndarray:
    years_full = np.asarray(panel["years_full"], dtype=int)
    je = _j_end(years_full, cut_year)
    key = (scenario, int(eol), lever)
    if key not in panel["lever_arrays"]:
        R = next(iter(panel["lever_arrays"].values())).shape[0]
        return np.zeros(R, dtype=float)
    A = panel["lever_arrays"][key]
    s = A[:, : je + 1].sum(axis=1)
    if lever == "uptake":
        s = -s
    return s.astype(float)


def _q(v: np.ndarray, p: float) -> float:
    return float(np.nanpercentile(np.asarray(v, float), p))


def _med(v: np.ndarray) -> float:
    return float(np.nanmedian(np.asarray(v, float)))


def _unit_scale(to_units: str = "Mt") -> Tuple[float, str]:
    u = str(to_units).lower()
    if u in ("mt","megaton","megatonne","megatonnes"): return 1e-6, "Mt CO2e"
    if u == "gt": return 1e-9, "Gt CO2e"
    if u == "kt": return 1e-3, "kt CO2e"
    return 1.0, "t CO2e"


FS_AXIS_LABEL = 18
FS_TICK = 18

SEG_ABBR = {
    "slash":"SL","mill_residue":"MR","process_logistics":"PR",
    "landfill":"LF","combustion":"EN","reuse":"RE","biochar":"BC",
}

def _label_stack(
    ax, x: float, start: float, heights: Sequence[float], keys: Sequence[str], *,
    bar_width: float, fs: int = 12, color: str = "0.15",
    outside_thresh: float = 0.35,
    force_outside_keys: Sequence[str] = (),
    bottom_align_keys: Sequence[str] = (),
    outside_side_map: Optional[Dict[str, str]] = None,
    y_offset_map_frac: Optional[Dict[str, float]] = None,
    outside_x_pad_frac: float = 0.12,
    label_z: int = 13,
):
    outside_side_map = outside_side_map or {}
    y_offset_map_frac = y_offset_map_frac or {}

    y0, y1 = ax.get_ylim()
    span = y1 - y0

    base = start
    for h, k in zip(heights, keys):
        if h <= 0:
            base -= h
            continue

        txt = SEG_ABBR.get(k, str(k)[:2].upper())
        y_center = base - 0.5*h
        y_bottom = base - h
        dy = float(y_offset_map_frac.get(k, 0.0)) * span

        must_outside = (h < outside_thresh) or (k in force_outside_keys)
        if must_outside:
            side = outside_side_map.get(k, "right")
            y_anchor = (y_bottom if (k in bottom_align_keys) else y_center) + dy
            if side == "left":
                x_edge = x - bar_width/2.0
                x_text = x_edge - outside_x_pad_frac * bar_width
                ha = "right"
            else:
                x_edge = x + bar_width/2.0
                x_text = x_edge + outside_x_pad_frac * bar_width
                ha = "left"

            ax.annotate(
                txt, xy=(x_edge, y_anchor), xytext=(x_text, y_anchor),
                ha=ha, va="center", fontsize=fs, fontweight="bold", color=color,
                arrowprops=dict(arrowstyle="-", lw=1.0, color="0.35"),
                zorder=label_z, clip_on=False
            )
        else:
            ax.text(x, y_center + dy, txt, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=color,
                    zorder=label_z, clip_on=False)
        base -= h


def plot_two_levers_waterfall_pub(
    bundle: Dict,
    *,
    panels: Sequence[str] = ("S1","S2","S3"),
    cut_year: int = 2120,
    to_units: str = "Gt",
    fig_size: Tuple[float,float] = (12, 4),
    dpi: int = 600,
    panel_letters: Sequence[str] = ("A","B","C"),
    letter_xy: Tuple[float,float] = (0.98, 0.96),
    bar_width: float = 0.55,
    group_gap: float = 1.05,
    uptake_color: str = "#5b5b5b",
    hep_cols: Optional[Dict[str,str]] = None,
    lep_cols: Optional[Dict[str,str]] = None,
    show_legend: bool = False,
    save_path: Optional[str] = None,
    en_yoff_frac: float = -0.020,
    bc_yoff_frac: float = -0.010,
    pl_yoff_frac: float =  0.000,
    mr_yoff_frac: float =  0.010,
    ann_decimals: int = 1,
) -> pd.DataFrame:
    plt.rcParams.update({
        "font.size": FS_TICK,
        "axes.labelsize": FS_AXIS_LABEL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "axes.linewidth": 1.2,
    })

    if hep_cols is None:
        hep_cols = {
            "slash":"#E66101","mill_residue":"#E28B3B","process_logistics":"#F4A259",
            "landfill":"#7A3A00","combustion":"#7A3A00",
        }
    if lep_cols is None:
        lep_cols = {
            "slash":"#6A51A3","mill_residue":"#7B74BD","process_logistics":"#9E9AC8",
            "reuse":"#542788","biochar":"#542788",
        }

    scale, _ = _unit_scale(to_units)

    fig, axes = plt.subplots(1, len(panels), figsize=fig_size, dpi=dpi, sharey=True, constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]

    x_u = 0.0
    x_h = x_u + group_gap
    x_l = x_h + group_gap

    global_ymax = []
    ann_stash = []
    table_rows = []

    legend_handles = [
        mpatches.Patch(facecolor=uptake_color, edgecolor="black", label="Uptake"),
        mpatches.Patch(facecolor=hep_cols["slash"], edgecolor="black", label="Slash (HEP)"),
        mpatches.Patch(facecolor=hep_cols["mill_residue"], edgecolor="black", label="Mill residue (HEP)"),
        mpatches.Patch(facecolor=hep_cols["process_logistics"], edgecolor="black", label="Process + logistics (HEP)"),
        mpatches.Patch(facecolor=hep_cols["landfill"], edgecolor="black", label="EoL landfill (front)"),
        mpatches.Patch(facecolor=hep_cols["combustion"], edgecolor="black", alpha=0.35, label="EoL combustion (behind)"),
        mpatches.Patch(facecolor=lep_cols["slash"], edgecolor="black", label="Slash (LEP)"),
        mpatches.Patch(facecolor=lep_cols["mill_residue"], edgecolor="black", label="Mill residue (LEP)"),
        mpatches.Patch(facecolor=lep_cols["process_logistics"], edgecolor="black", label="Process + logistics (LEP)"),
        mpatches.Patch(facecolor=lep_cols["reuse"], edgecolor="black", label="EoL reuse (front)"),
        mpatches.Patch(facecolor=lep_cols["biochar"], edgecolor="black", alpha=0.35, label="EoL biochar (behind)"),
    ]

    for i, S in enumerate(panels):
        if S not in bundle["panels"]:
            continue

        ax = axes[i]
        panel = bundle["panels"][S]

        u_vec = _sum_to_year(panel, "low", 1, "uptake", cut_year)
        u_med = _med(u_vec) * scale
        u_p05 = _q(u_vec, 5) * scale
        u_p95 = _q(u_vec, 95) * scale

        sl_h = _sum_to_year(panel, "high", 2, "slash", cut_year)
        mr_h = _sum_to_year(panel, "high", 2, "mill_residue", cut_year)
        pr_h = _sum_to_year(panel, "high", 2, "process_logistics", cut_year)
        lf   = _sum_to_year(panel, "high", 2, "landfill", cut_year)
        en   = _sum_to_year(panel, "high", 1, "combustion", cut_year)

        sl_l = _sum_to_year(panel, "low", 2, "slash", cut_year)
        mr_l = _sum_to_year(panel, "low", 2, "mill_residue", cut_year)
        pr_l = _sum_to_year(panel, "low", 2, "process_logistics", cut_year)
        re   = _sum_to_year(panel, "low", 2, "reuse", cut_year)
        bc   = _sum_to_year(panel, "low", 1, "biochar", cut_year)

        h_meds = [max(0.0, _med(sl_h)*scale), max(0.0, _med(mr_h)*scale), max(0.0, _med(pr_h)*scale)]
        lf_med = max(0.0, _med(lf)*scale)
        en_med = max(0.0, _med(en)*scale)

        l_meds = [max(0.0, _med(sl_l)*scale), max(0.0, _med(mr_l)*scale), max(0.0, _med(pr_l)*scale)]
        re_med = max(0.0, _med(re)*scale)
        bc_med = max(0.0, _med(bc)*scale)

        ax.bar(x_u, u_med, width=bar_width, color=uptake_color, edgecolor="black", linewidth=1.0, zorder=3)
        ax.vlines(x_u, u_p05, u_p95, color="black", linewidth=2.1, zorder=10)
        ax.hlines([u_p05, u_p95], x_u - bar_width*0.25, x_u + bar_width*0.25, color="black", linewidth=2.1, zorder=10)

        def draw_waterfall(x, start, heights, colors, *, alpha=1.0, z=2):
            base = start
            for h, c in zip(heights, colors):
                ax.bar(x, -h, width=bar_width, bottom=base, color=c, edgecolor="black",
                       linewidth=1.0, alpha=alpha, zorder=z)
                base -= h
            return base

        draw_waterfall(x_h, u_med, h_meds + [en_med],
                       [hep_cols["slash"], hep_cols["mill_residue"], hep_cols["process_logistics"], hep_cols["combustion"]],
                       alpha=0.35, z=1)
        draw_waterfall(x_h, u_med, h_meds + [lf_med],
                       [hep_cols["slash"], hep_cols["mill_residue"], hep_cols["process_logistics"], hep_cols["landfill"]],
                       alpha=1.0, z=4)

        draw_waterfall(x_l, u_med, l_meds + [bc_med],
                       [lep_cols["slash"], lep_cols["mill_residue"], lep_cols["process_logistics"], lep_cols["biochar"]],
                       alpha=0.35, z=1)
        draw_waterfall(x_l, u_med, l_meds + [re_med],
                       [lep_cols["slash"], lep_cols["mill_residue"], lep_cols["process_logistics"], lep_cols["reuse"]],
                       alpha=1.0, z=4)

        HATCH_FACE = (1,1,1,0.0)
        HATCH = "////"
        boundary_hep = u_med - sum(h_meds)
        boundary_lep = u_med - sum(l_meds)

        ax.bar(x_h, -en_med, width=bar_width, bottom=boundary_hep,
               facecolor=HATCH_FACE, edgecolor=hep_cols["landfill"], hatch=HATCH, linewidth=1.2, zorder=12)
        ax.bar(x_l, -bc_med, width=bar_width, bottom=boundary_lep,
               facecolor=HATCH_FACE, edgecolor=lep_cols["reuse"], hatch=HATCH, linewidth=1.2, zorder=12)

        _label_stack(ax, x_h, u_med, h_meds + [lf_med],
                     ["slash","mill_residue","process_logistics","landfill"],
                     bar_width=bar_width, fs=12, outside_thresh=0.35,
                     outside_side_map={"process_logistics":"left"},
                     force_outside_keys=("process_logistics",),
                     y_offset_map_frac={"process_logistics": pl_yoff_frac, "mill_residue": mr_yoff_frac})

        start_en = boundary_hep + en_med
        _label_stack(ax, x_h, start_en, [en_med], ["combustion"],
                     bar_width=bar_width, fs=12, outside_thresh=1e9,
                     force_outside_keys=("combustion",),
                     bottom_align_keys=("combustion",),
                     outside_side_map={"combustion":"left"},
                     y_offset_map_frac={"combustion": en_yoff_frac})

        _label_stack(ax, x_l, u_med, l_meds + [re_med],
                     ["slash","mill_residue","process_logistics","reuse"],
                     bar_width=bar_width, fs=12, outside_thresh=0.35,
                     outside_side_map={"process_logistics":"left"},
                     force_outside_keys=("process_logistics",),
                     y_offset_map_frac={"process_logistics": pl_yoff_frac, "mill_residue": mr_yoff_frac})

        start_bc = boundary_lep + bc_med
        _label_stack(ax, x_l, start_bc, [bc_med], ["biochar"],
                     bar_width=bar_width, fs=12, outside_thresh=1e9,
                     force_outside_keys=("biochar",),
                     bottom_align_keys=("biochar",),
                     outside_side_map={"biochar":"left"},
                     y_offset_map_frac={"biochar": bc_yoff_frac})

        rem_hep_lf = u_vec - (sl_h + mr_h + pr_h + lf)
        rem_hep_en = u_vec - (sl_h + mr_h + pr_h + en)
        rem_lep_re = u_vec - (sl_l + mr_l + pr_l + re)
        rem_lep_bc = u_vec - (sl_l + mr_l + pr_l + bc)

        hep_low  = min(_q(rem_hep_lf,5),  _q(rem_hep_en,5))  * scale
        hep_high = max(_q(rem_hep_lf,95), _q(rem_hep_en,95)) * scale
        lep_low  = min(_q(rem_lep_re,5),  _q(rem_lep_bc,5))  * scale
        lep_high = max(_q(rem_lep_re,95), _q(rem_lep_bc,95)) * scale

        kw = dict(color="black", linewidth=2.1, zorder=10)
        ax.vlines(x_h, hep_low, hep_high, **kw)
        ax.hlines([hep_low, hep_high], x_h - bar_width*0.26, x_h + bar_width*0.26, **kw)
        ax.vlines(x_l, lep_low, lep_high, **kw)
        ax.hlines([lep_low, lep_high], x_l - bar_width*0.26, x_l + bar_width*0.26, **kw)

        rel_hep_lf = sl_h + mr_h + pr_h + lf
        rel_hep_en = sl_h + mr_h + pr_h + en
        rel_lep_re = sl_l + mr_l + pr_l + re
        rel_lep_bc = sl_l + mr_l + pr_l + bc

        hep_med_lf = _med(rel_hep_lf) * scale
        hep_med_en = _med(rel_hep_en) * scale
        lep_med_re = _med(rel_lep_re) * scale
        lep_med_bc = _med(rel_lep_bc) * scale

        def _fmt_range(a: float, b: float) -> str:
            lo, hi = (a, b) if a <= b else (b, a)
            return f"{lo:.{ann_decimals}f} – {hi:.{ann_decimals}f} {to_units}"

        ann_stash.append({"ax": ax, "x": x_h, "anchor": u_med, "txt": _fmt_range(hep_med_lf, hep_med_en)})
        ann_stash.append({"ax": ax, "x": x_l, "anchor": u_med, "txt": _fmt_range(lep_med_re, lep_med_bc)})

        ax.grid(True, axis="y", linestyle="-", alpha=0.15, linewidth=0.9, zorder=0)
        for side in ("top","right","left","bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.2)
            ax.spines[side].set_color("0.2")

        ax.set_xticks([x_u, x_h, x_l])
        ax.set_xticklabels(["Uptake","HEP Release","LEP Release"])
        if i == 0:
            ax.set_ylabel(f"Carbon Storage & Release ({to_units} CO$_2e$)", fontsize=FS_AXIS_LABEL)

        if i < len(panel_letters):
            ax.text(*letter_xy, panel_letters[i], transform=ax.transAxes,
                    ha="right", va="top", fontsize=FS_TICK, fontweight="bold")

        global_ymax.append(max(u_p95, u_med, hep_high, lep_high))
        ax.set_xlim(x_u - 0.45, x_l + 0.65)

        table_rows.append({
            "panel": S,
            "uptake_med": u_med,
            "HEP_med_total_LF": float(np.nanmedian(rel_hep_lf) * scale),
            "HEP_med_total_EN": float(np.nanmedian(rel_hep_en) * scale),
            "LEP_med_total_RE": float(np.nanmedian(rel_lep_re) * scale),
            "LEP_med_total_BC": float(np.nanmedian(rel_lep_bc) * scale),
            "HEP_env_low": hep_low, "HEP_env_high": hep_high,
            "LEP_env_low": lep_low, "LEP_env_high": lep_high,
        })

        if show_legend and i == 0:
            ax.legend(handles=legend_handles, ncol=3, frameon=False, fontsize=max(FS_TICK-1,11),
                      loc="upper left")

    Ymax = max(global_ymax) * 1.06 if global_ymax else 1.0
    for ax in axes:
        ax.set_ylim(0, Ymax)

    for it in ann_stash:
        ax = it["ax"]
        y0, y1 = ax.get_ylim()
        span = y1 - y0
        y_text = min(y1 - 0.02*span, it["anchor"] + 0.01*span)
        ax.text(it["x"], y_text, it["txt"], ha="center", va="bottom", fontsize=14, fontweight="bold")

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
    return pd.DataFrame(table_rows)


# =============================================================================
# Future-only uptake extension
# =============================================================================

def _growth_weights(n: int, model: str = "richards", params: Optional[Dict] = None) -> np.ndarray:
    params = params or {}
    t = np.linspace(0.0, 1.0, n + 1)

    if model == "linear":
        inc = np.ones(n)
    elif model in ("sigmoid","logistic","s-curve"):
        k  = float(params.get("k", 6.0))
        x0 = float(params.get("x0", 0.55))
        F = 1.0 / (1.0 + np.exp(-k * (t - x0)))
        inc = np.diff(F)
    elif model == "gompertz":
        b = float(params.get("b", 2.0))
        c = float(params.get("c", 3.0))
        F = np.exp(-b * np.exp(-c * t))
        F = (F - F[0]) / (F[-1] - F[0] + 1e-15)
        inc = np.diff(F)
    elif model in ("richards","chapman_richards"):
        k   = float(params.get("k", 4.0))
        x0  = float(params.get("x0", 0.45))
        nu  = float(params.get("nu", 1.6))
        eps = float(params.get("eps", 1e-6))
        a = ((1 - eps) ** (-nu) - 1.0) * np.exp(k * (1 - x0))
        F = (1.0 + a * np.exp(-k * (t - x0))) ** (-1.0 / nu)
        F = (F - F[0]) / (F[-1] - F[0] + 1e-15)
        inc = np.diff(F)
    else:
        raise ValueError(f"unknown growth model '{model}'")

    w = np.clip(inc, 0.0, None)
    s = w.sum()
    return (w / s) if s > 0 else (np.ones(n) / float(n))


def _total_uptake_by_rep_for_cohort(MAIN_PATH: str, prod_scenario: str, year: int,
                                   regions: Sequence[str] = ("PNW","SE")) -> pd.Series:
    pieces = []
    for region in regions:
        prod_CLT = _get_prod_df(region, "CLT", prod_scenario)
        prod_GLT = _get_prod_df(region, "GLT", prod_scenario)
        ycol = year if year in prod_CLT.columns else str(year)
        pCLT = prod_CLT[ycol]
        pGLT = prod_GLT[ycol]

        eCLT = load_emissions_df(MAIN_PATH, region, "CLT", "low", year)[["raw_materials_uptake"]].copy()
        eGLT = load_emissions_df(MAIN_PATH, region, "GLT", "low", year)[["raw_materials_uptake"]].copy()

        eCLT = _scale_to_prod_and_tonnes(eCLT, pCLT, ["raw_materials_uptake"], True, "kg", "t")
        eGLT = _scale_to_prod_and_tonnes(eGLT, pGLT, ["raw_materials_uptake"], True, "kg", "t")

        pieces.append(eCLT["raw_materials_uptake"] + eGLT["raw_materials_uptake"])
    tot = sum(pieces)
    tot.name = "uptake_template_t"
    return tot


def _future_uptake_matrix(
    uptake_template: pd.Series,
    years_full: np.ndarray,
    *,
    last_real_cohort: int = 2120,
    future_last: int = 2150,
    growth_years: int = 21,
    growth_model: str = "richards",
    growth_params: Optional[Dict] = None,
    rep_order: Optional[Sequence] = None,
) -> np.ndarray:
    rep_order = list(rep_order) if rep_order is not None else list(uptake_template.index)
    amp = uptake_template.reindex(rep_order).fillna(0.0).to_numpy(dtype=float)

    R, Y = len(rep_order), len(years_full)
    year_to_j = {int(y): j for j, y in enumerate(years_full)}

    B = np.zeros((R, Y), dtype=np.float64)
    w = _growth_weights(int(growth_years), model=growth_model, params=(growth_params or {}))

    for cohort in range(int(last_real_cohort) + 1, int(future_last) + 1):
        start = cohort - int(growth_years) + 1
        for k in range(int(growth_years)):
            yr = start + k
            if yr > last_real_cohort:
                break
            j = year_to_j.get(int(yr))
            if j is not None:
                B[:, j] += amp * w[k]
    return B


def extend_lever_bundle_with_future_uptake(
    lever_bundle: Dict,
    MAIN_PATH: str,
    *,
    PROJ_BASE: Optional[str] = None,
    last_real_cohort: int = 2120,
    future_last: int = 2150,
    growth_years: int = 21,
    growth_model: str = "richards",
    growth_params: Optional[Dict] = None,
    save_as: Optional[str] = None,
    save_folder: str = "Results/PlotBundles",
) -> Tuple[Dict, pd.DataFrame]:
    import copy

    meta = lever_bundle.get("meta", {})
    regions = tuple(meta.get("include_regions", ("PNW","SE")))

    out = {"panels": {}, "meta": copy.deepcopy(meta)}
    out["meta"]["note"] = meta.get("note","") + f" | uptake patched with cohorts {last_real_cohort+1}–{future_last}"

    checks = []

    for S, panel in lever_bundle["panels"].items():
        if PROJ_BASE is not None:
            load_all_projections(PROJ_BASE, scenarios=(S,), products=("CLT","GLT"))

        years_full = np.asarray(panel["years_full"], dtype=int)
        rep_ids = list(panel.get("rep_ids") or range(next(iter(panel["lever_arrays"].values())).shape[0]))

        uptake_template = _total_uptake_by_rep_for_cohort(MAIN_PATH, S, last_real_cohort, regions)

        B_pos = _future_uptake_matrix(
            uptake_template, years_full,
            last_real_cohort=last_real_cohort, future_last=future_last,
            growth_years=growth_years, growth_model=growth_model, growth_params=growth_params,
            rep_order=rep_ids,
        )
        B_net = (-B_pos).astype(np.float32)

        lever_arrays = {k: v.copy() for k, v in panel["lever_arrays"].items()}
        for sc in ("low","high"):
            for eol in (1,2):
                key = (sc, int(eol), "uptake")
                if key in lever_arrays:
                    if lever_arrays[key].shape != B_net.shape:
                        raise ValueError(f"Shape mismatch for {S} {key}: {lever_arrays[key].shape} vs {B_net.shape}")
                    lever_arrays[key] = lever_arrays[key] + B_net

        year_to_j = {int(y): j for j, y in enumerate(years_full)}
        for yr in range(max(2000, last_real_cohort - growth_years + 1), last_real_cohort + 1):
            j = year_to_j.get(int(yr))
            if j is None:
                continue
            checks.append({"panel": S, "year": int(yr), "added_uptake_Gt": float(-B_net[:, j].sum()) * 1e-9})

        out["panels"][S] = {"years_full": years_full, "rep_ids": rep_ids, "lever_arrays": lever_arrays}

    checks_df = pd.DataFrame(checks).sort_values(["panel","year"]).reset_index(drop=True)
    if save_as:
        save_lever_bundle(out, folder=save_folder, name=save_as)
    return out, checks_df


# =============================================================================
# Deltas from levers + breakdown table
# =============================================================================

DEFAULT_EPA_BAU_WEIGHTS: Dict[Tuple[str, int], float] = {
    ("high", 2): 0.7,
    ("high", 1): 0.3,
}

def compute_case_deltas_from_levers(
    lever_bundle: Dict,
    cut_year: int,
    *,
    percentiles: Sequence[Union[str, float]] = ("median", "p05", "p95"),
    to_units: str = "Mt",
    variants: Optional[Sequence[Tuple[str,int]]] = None,
    reference: Tuple[str,int] = ("low", 2),
    include_bau: bool = False,
    bau_weights: Optional[Dict[Tuple[str,int], float]] = None,
    eps: float = 1e-12,
) -> pd.DataFrame:
    EOL_LEVER = {("high",1):"combustion", ("high",2):"landfill", ("low",1):"biochar", ("low",2):"reuse"}

    if variants is None:
        variants = [("low",1),("high",2),("high",1)]

    VAR_LABELS = {
        ("low",2):"reuse", ("low",1):"biochar", ("high",2):"landfill", ("high",1):"combustion", ("bau",0):"BAU mix",
    }

    u = str(to_units).lower()
    if u == "gt": scale, units = 1e-9, "Gt"
    elif u == "mt": scale, units = 1e-6, "Mt"
    elif u == "kt": scale, units = 1e-3, "kt"
    else: scale, units = 1.0, "t"

    if include_bau:
        if bau_weights is None:
            bau_weights = DEFAULT_EPA_BAU_WEIGHTS
        w = {k: float(v) for k,v in bau_weights.items() if k in {("high",2),("high",1)} and v>0}
        s = sum(w.values())
        bau_w_land = w.get(("high",2), 0.0)/s if s>0 else 0.73
        bau_w_comb = w.get(("high",1), 0.0)/s if s>0 else 0.27

    rows = []
    for panel_key, panel in lever_bundle["panels"].items():
        years = np.asarray(panel["years_full"], dtype=int)
        if cut_year not in set(years):
            raise ValueError(f"cut_year {cut_year} not in panel {panel_key} horizon {years.min()}–{years.max()}")
        j = int(np.where(years == cut_year)[0][0])

        def vec(key):
            A = panel["lever_arrays"][key]
            return A[:, :j+1].sum(axis=1)

        def net(sc, eol):
            eol_lev = EOL_LEVER[(sc,eol)]
            U = vec((sc,eol,"uptake"))        # net (negative)
            SL = vec((sc,eol,"slash"))
            MR = vec((sc,eol,"mill_residue"))
            PR = vec((sc,eol,"process_logistics"))
            E = vec((sc,eol,eol_lev))
            return (-U) - (SL + MR + PR + E)

        ref_sc, ref_eol = reference
        ref_vec = net(ref_sc, ref_eol)

        if include_bau:
            net_bau = bau_w_land*net("high",2) + bau_w_comb*net("high",1)

        for p in percentiles:
            if isinstance(p, str):
                pl = p.lower()
                if pl in ("median","p50"): fstat = lambda v: float(np.nanmedian(v))
                elif pl in ("p05","p5"):  fstat = lambda v: float(np.nanpercentile(v,5))
                elif pl == "p95":         fstat = lambda v: float(np.nanpercentile(v,95))
                elif pl == "mean":        fstat = lambda v: float(np.nanmean(v))
                else: raise ValueError(f"Unknown stat '{p}'")
                stat_label = pl
            else:
                q = float(p)
                if not (0.0 < q < 1.0): raise ValueError("Numeric percentiles must be in (0,1)")
                fstat = lambda v, q=q: float(np.nanquantile(v, q))
                stat_label = f"q{int(round(100*q))}"

            ref_val = fstat(ref_vec) * scale

            run_variants = list(variants) + ([("bau",0)] if include_bau else [])
            for case_sc, case_eol in run_variants:
                case_vec = net_bau if (case_sc,case_eol)==("bau",0) else net(case_sc,case_eol)
                case_val = fstat(case_vec) * scale
                abs_delta = case_val - ref_val
                denom = abs(ref_val)
                rel_delta = (abs_delta/denom) if (np.isfinite(denom) and denom>eps) else np.nan

                rows.append({
                    "panel": panel_key, "cut_year": int(cut_year),
                    "variant": VAR_LABELS.get((case_sc,case_eol), f"{case_sc}__{case_eol}"),
                    "variant_key": f"{case_sc}__{case_eol}",
                    "stat": stat_label,
                    "ref_value": ref_val, "case_value": case_val,
                    "abs_delta": abs_delta, "rel_delta": rel_delta,
                    "units": units,
                })

    out = pd.DataFrame(rows)
    cols = ["panel","cut_year","variant","variant_key","stat","ref_value","case_value","abs_delta","rel_delta","units"]
    return out[cols].sort_values(["panel","variant","stat"]).reset_index(drop=True)


def build_lever_breakdown_table(
    bundle: Dict,
    *,
    panels: Sequence[str] = ("S1","S2","S3"),
    cut_year: int = 2120,
    to_units: str = "Gt",
    stats: Sequence[str] = ("median","p05","p95"),
    round_to: Optional[int] = 2,
) -> pd.DataFrame:
    scale, units_label = _unit_scale(to_units)
    rows = []

    def _summ(v: np.ndarray) -> Dict[str, float]:
        v = np.asarray(v, float) * scale
        out = {}
        for s in stats:
            s_l = str(s).lower()
            if s_l in ("median","p50"): out["median"] = _med(v)
            elif s_l in ("p05","p5"):  out["p05"] = _q(v, 5)
            elif s_l == "p95":         out["p95"] = _q(v,95)
            elif s_l == "mean":        out["mean"] = float(np.nanmean(v))
        return out

    for S in panels:
        if S not in bundle["panels"]:
            continue
        panel = bundle["panels"][S]

        uptake = _sum_to_year(panel, "low", 1, "uptake", cut_year)

        sl_h = _sum_to_year(panel, "high", 2, "slash", cut_year)
        mr_h = _sum_to_year(panel, "high", 2, "mill_residue", cut_year)
        pl_h = _sum_to_year(panel, "high", 2, "process_logistics", cut_year)
        lf   = _sum_to_year(panel, "high", 2, "landfill", cut_year)
        en   = _sum_to_year(panel, "high", 1, "combustion", cut_year)

        sl_l = _sum_to_year(panel, "low", 2, "slash", cut_year)
        mr_l = _sum_to_year(panel, "low", 2, "mill_residue", cut_year)
        pl_l = _sum_to_year(panel, "low", 2, "process_logistics", cut_year)
        re   = _sum_to_year(panel, "low", 2, "reuse", cut_year)
        bc   = _sum_to_year(panel, "low", 1, "biochar", cut_year)

        hep_lf = {"slash": sl_h, "mill_residue": mr_h, "process_logistics": pl_h, "EoL": lf}
        hep_en = {"slash": sl_h, "mill_residue": mr_h, "process_logistics": pl_h, "EoL": en}
        lep_re = {"slash": sl_l, "mill_residue": mr_l, "process_logistics": pl_l, "EoL": re}
        lep_bc = {"slash": sl_l, "mill_residue": mr_l, "process_logistics": pl_l, "EoL": bc}

        def _emit_case(group: str, eol_case: str, comp: Dict[str, np.ndarray]):
            for k, v in comp.items():
                st = _summ(v)
                st.update(panel=S, group=group, eol_case=eol_case, component=k)
                rows.append(st)

            total_rel = sum(comp.values())
            net_store = uptake - total_rel

            st_rel = _summ(total_rel); st_rel.update(panel=S, group=group, eol_case=eol_case, component="total_release")
            st_upt = _summ(uptake);    st_upt.update(panel=S, group=group, eol_case=eol_case, component="uptake")
            st_net = _summ(net_store); st_net.update(panel=S, group=group, eol_case=eol_case, component="net_storage")

            rows.extend([st_rel, st_upt, st_net])

        _emit_case("HEP","landfill", hep_lf)
        _emit_case("HEP","energy",   hep_en)
        _emit_case("LEP","reuse",    lep_re)
        _emit_case("LEP","biochar",  lep_bc)

    out = pd.DataFrame(rows)
    stat_cols = [c for c in ("median","mean","p05","p95") if c in out.columns]
    out["units"] = units_label
    out = out[["panel","group","eol_case","component"] + stat_cols + ["units"]]

    if round_to is not None:
        for c in stat_cols:
            out[c] = out[c].round(round_to)

    comp_order = pd.CategoricalDtype(
        ["slash","mill_residue","process_logistics","EoL","total_release","uptake","net_storage"],
        ordered=True
    )
    out["component"] = out["component"].astype(comp_order)
    return out.sort_values(["panel","group","eol_case","component"]).reset_index(drop=True)

#%%
###############################################################################
###############################################################################
# Switch Timing Functions 
###############################################################################
###############################################################################

#from __future__ import annotations

from typing import Dict, Tuple, Optional, Sequence, Mapping, Any, Union
import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# tqdm is optional (keeps code usable even if tqdm isn't installed)
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
SWITCH_DEFAULT_HEP_BAU_WEIGHTS: Dict[Tuple[str, int], float] = {
    ("high", 2): 0.70,  # landfill
    ("high", 1): 0.30,  # combustion
}

SWITCH_DEFAULT_LEP_WEIGHTS: Dict[Tuple[str, int], float] = {
    ("low", 2): 0.60,  # reuse
    ("low", 1): 0.40,  # biochar
}

# Base (pre‑EoL) stages 
SWITCH_PROC_STAGES: Tuple[str, ...] = (
    "raw_materials",
    "transportation_1",
    "processing",
    "transportation_2",
    "manufacture",
    "transportation_3",
    "construction",
    "use",
    "deconstruction",
    "transportation_4",
)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _switch_normalize_weights(
    weights: Mapping[Tuple[str, int], float],
    keys: Sequence[Tuple[str, int]],
) -> Dict[Tuple[str, int], float]:
    """Normalize positive weights over 'keys'. If all are zero/missing, use equal weights."""
    w = {k: float(weights.get(k, 0.0)) for k in keys}
    w = {k: (v if v > 0 else 0.0) for k, v in w.items()}
    s = float(sum(w.values()))
    if s <= 0.0:
        # equal split fallback
        return {k: 1.0 / float(len(keys)) for k in keys}
    return {k: v / s for k, v in w.items()}


def _switch_long_from_flows(
    flows_long: pd.DataFrame,
    *,
    stages: Sequence[str] = (),
    metrics: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Return long DF: [replicate, year, net].
    Uses flows_long['net'] if present, otherwise builds it from (metric,value).
    """
    if flows_long is None or flows_long.empty:
        return pd.DataFrame(columns=["replicate", "year", "net"])

    sub = flows_long
    if stages:
        sub = sub[sub["stage"].isin(stages)]
    if metrics is not None:
        sub = sub[sub["metric"].isin(metrics)]

    if sub.empty:
        return pd.DataFrame(columns=["replicate", "year", "net"])

    if "net" not in sub.columns:
        # fossil +, biogenic +, uptake -, energy_credit -
        w = {"fossil": 1.0, "biogenic": 1.0, "uptake": -1.0, "energy_credit": -1.0}
        sub = sub.assign(net=sub["value"] * sub["metric"].map(w).fillna(0.0))

    return sub[["replicate", "year", "net"]].copy()


def _switch_accumulate_long(
    long_df: pd.DataFrame,
    arr: np.ndarray,
    rep_to_i: pd.Series,
    year_to_j: pd.Series,
) -> None:
    """
    Accumulate long [replicate, year, net] into annual NET array [R,Y].
    Rows outside horizon are silently dropped (year not in year_to_j).
    """
    if long_df is None or long_df.empty:
        return

    ri = long_df["replicate"].map(rep_to_i).to_numpy()
    yj = long_df["year"].map(year_to_j).to_numpy()
    v = long_df["net"].to_numpy(dtype=np.float32)

    m = (~pd.isna(ri)) & (~pd.isna(yj))
    if not np.any(m):
        return

    np.add.at(
        arr,
        (ri[m].astype(int), yj[m].astype(int)),
        v[m],
    )


def _switch_scenario_folder_for_year(main_path: str, region: str, product: str, scenario: str, year: int) -> str:

    try:
        # Preferred 
        return _scenario_folder(main_path, region, product, scenario, year=year, require_stage_files=True)
    except TypeError:
        # Back-compat fallback (older _scenario_folder signature)
        folder = _scenario_folder(main_path, region, product, scenario)
        req = [
            f"harvest_df_results_{year}.csv",
            f"sawmill_df_results_{year}.csv",
            f"operation_df_results_{year}.csv",
            f"const_dem_df_results_{year}.csv",
            f"EoL_df_results_{year}.csv",
            f"seq_df_results_{year}.csv",
        ]
        missing = [fn for fn in req if not os.path.exists(os.path.join(folder, fn))]
        if missing:
            raise FileNotFoundError(f"Missing required MCS stage files in {folder}: {missing}")
        return folder


def detect_max_available_cohort_year(
    main_path: str,
    *,
    regions: Sequence[str] = ("PNW", "SE"),
    products: Sequence[str] = ("CLT", "GLT"),
    scenarios: Sequence[str] = ("low", "high"),
    start_year: int = 2020,
    stop_year: int = 2300,
) -> int:

    max_ok = start_year - 1
    for y in range(int(start_year), int(stop_year) + 1):
        ok = True
        for reg in regions:
            for prod in products:
                for sc in scenarios:
                    try:
                        _switch_scenario_folder_for_year(main_path, reg, prod, sc, y)
                    except FileNotFoundError:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break

        if ok:
            max_ok = y
        else:
            break

    return int(max_ok)


# -----------------------------------------------------------------------------
# Main builder
# -----------------------------------------------------------------------------
def build_switch_paths_fast(
    PROJ_BASE: str,
    MAIN_PATH: str,
    *,
    switch_years: Sequence[int] = (2025, 2050, 2075, 2100),
    cut_year: int = 2120,
    start_cohort: int = 2020,
    prod_scenarios: Sequence[str] = ("S1", "S2", "S3"),
    include_regions: Sequence[str] = ("PNW", "SE"),
    eol_policy: str = "follow_lep",  # "as_produced" or "follow_lep"
    hep_bau_weights: Optional[Mapping[Tuple[str, int], float]] = None,
    lep_weights: Optional[Mapping[Tuple[str, int], float]] = None,
    gwp100_ch4: float = 28.0,
    include_baselines: bool = True,
) -> Dict[str, Any]:

    eol_policy = str(eol_policy).lower().strip()
    if eol_policy not in ("as_produced", "follow_lep"):
        raise ValueError("eol_policy must be 'as_produced' or 'follow_lep'")

    switch_years = [int(y) for y in switch_years]
    if any(y > int(cut_year) for y in switch_years):
        raise ValueError("All switch_years must be <= cut_year")

    hep_bau_weights = hep_bau_weights or SWITCH_DEFAULT_HEP_BAU_WEIGHTS
    lep_weights = lep_weights or SWITCH_DEFAULT_LEP_WEIGHTS

    # normalize weights
    hep_w = _switch_normalize_weights(hep_bau_weights, keys=[("high", 1), ("high", 2)])
    lep_w = _switch_normalize_weights(lep_weights, keys=[("low", 1), ("low", 2)])

    years_full = np.arange(2000, int(cut_year) + 1, dtype=int)
    year_to_j = pd.Series(range(len(years_full)), index=years_full)

    # load projections once (rep index + scaling)
    load_all_projections(PROJ_BASE, scenarios=tuple(prod_scenarios), products=("CLT", "GLT"))

    meta = {
        "switch_years": list(map(int, switch_years)),
        "cut_year": int(cut_year),
        "start_cohort": int(start_cohort),
        "prod_scenarios": list(prod_scenarios),
        "include_regions": list(include_regions),
        "eol_policy": eol_policy,
        "hep_bau_weights": dict(hep_bau_weights),
        "lep_weights": dict(lep_weights),
        "gwp100_ch4": float(gwp100_ch4),
        "note": "cum_arrays store cumulative NET (t CO2e). Storage = -NET.",
    }

    out_panels: Dict[str, Dict[str, Any]] = {}
    summary_rows: list[dict] = []

    for S in prod_scenarios:
        # infer replicate index from any available production DF
        rep_index_source = None
        for reg in include_regions:
            try:
                rep_index_source = _get_prod_df(reg, "CLT", S).index
                break
            except Exception:
                continue
        if rep_index_source is None:
            raise RuntimeError(f"No production DF found to infer replicates for {S}.")

        reps = pd.Index(rep_index_source)
        rep_to_i = pd.Series(range(len(reps)), index=reps)

        # detect how far we have complete MCS stage files
        max_avail = detect_max_available_cohort_year(
            MAIN_PATH,
            regions=include_regions,
            products=("CLT", "GLT"),
            scenarios=("low", "high"),
            start_year=int(start_cohort),
            stop_year=int(cut_year),
        )
        cohort_end = min(int(max_avail), int(cut_year))

        if cohort_end < max(switch_years):
            print(
                f"[{S}] WARNING: complete MCS cohorts available only through {cohort_end}. "
                f"Switch years beyond that will have no post-switch cohorts."
            )

        # annual NET arrays for each switch case
        nets_by_switch = {
            ys: np.zeros((len(reps), len(years_full)), dtype=np.float32)
            for ys in switch_years
        }

        # optional baselines
        nets_all_hep = np.zeros((len(reps), len(years_full)), dtype=np.float32)
        nets_all_lep = np.zeros((len(reps), len(years_full)), dtype=np.float32)

        for y0 in tqdm(range(int(start_cohort), int(cohort_end) + 1), desc=f"Switch paths {S}"):
            for region in include_regions:
                # -------------------------
                # HIGH (HEP ingredients)
                # -------------------------
                em_CLT_high = load_emissions_df(MAIN_PATH, region, "CLT", "high", y0)
                em_GLT_high = load_emissions_df(MAIN_PATH, region, "GLT", "high", y0)

                mr_CLT_high = load_ts_df(MAIN_PATH, region, "CLT", "high", y0, kind="mill")
                mr_GLT_high = load_ts_df(MAIN_PATH, region, "GLT", "high", y0, kind="mill")
                lf_CLT_high = load_ts_df(MAIN_PATH, region, "CLT", "high", y0, kind="landfill")
                lf_GLT_high = load_ts_df(MAIN_PATH, region, "GLT", "high", y0, kind="landfill")

                # slash schedules only in SE-high 
                if str(region).upper() == "SE":
                    sl_CLT_high = load_ts_df(MAIN_PATH, region, "CLT", "high", y0, kind="slash")
                    sl_GLT_high = load_ts_df(MAIN_PATH, region, "GLT", "high", y0, kind="slash")
                else:
                    sl_CLT_high = None
                    sl_GLT_high = None

                fl_CLT_high = compute_flows_for_product(
                    em_CLT_high,
                    _get_prod_df(region, "CLT", S),
                    DcubeConfig(
                        base_year=y0,
                        emission_scenario="high",
                        region=region,
                        production_scenario=S,
                        gwp100_ch4=gwp100_ch4,
                    ),
                    "CLT",
                    use_schedules={"mill": mr_CLT_high, "landfill": lf_CLT_high, "slash": sl_CLT_high},
                )["flows_long"]

                fl_GLT_high = compute_flows_for_product(
                    em_GLT_high,
                    _get_prod_df(region, "GLT", S),
                    DcubeConfig(
                        base_year=y0,
                        emission_scenario="high",
                        region=region,
                        production_scenario=S,
                        gwp100_ch4=gwp100_ch4,
                    ),
                    "GLT",
                    use_schedules={"mill": mr_GLT_high, "landfill": lf_GLT_high, "slash": sl_GLT_high},
                )["flows_long"]

                fl_high = combine_products_aggregated([fl_CLT_high, fl_GLT_high])

                # -------------------------
                # LOW (LEP ingredients)
                # -------------------------
                em_CLT_low = load_emissions_df(MAIN_PATH, region, "CLT", "low", y0)
                em_GLT_low = load_emissions_df(MAIN_PATH, region, "GLT", "low", y0)
                mr_CLT_low = load_ts_df(MAIN_PATH, region, "CLT", "low", y0, kind="mill")
                mr_GLT_low = load_ts_df(MAIN_PATH, region, "GLT", "low", y0, kind="mill")

                fl_CLT_low = compute_flows_for_product(
                    em_CLT_low,
                    _get_prod_df(region, "CLT", S),
                    DcubeConfig(
                        base_year=y0,
                        emission_scenario="low",
                        region=region,
                        production_scenario=S,
                    ),
                    "CLT",
                    use_schedules={"mill": mr_CLT_low},
                )["flows_long"]

                fl_GLT_low = compute_flows_for_product(
                    em_GLT_low,
                    _get_prod_df(region, "GLT", S),
                    DcubeConfig(
                        base_year=y0,
                        emission_scenario="low",
                        region=region,
                        production_scenario=S,
                    ),
                    "GLT",
                    use_schedules={"mill": mr_GLT_low},
                )["flows_long"]

                fl_low = combine_products_aggregated([fl_CLT_low, fl_GLT_low])

                # -------------------------
                # base (pre‑EoL) includes "use"
                # -------------------------
                base_high = _switch_long_from_flows(fl_high, stages=SWITCH_PROC_STAGES)
                base_low = _switch_long_from_flows(fl_low, stages=SWITCH_PROC_STAGES)

                # -------------------------
                # EoL mixes
                # -------------------------
                eol_high_comb = _switch_long_from_flows(fl_high, stages=["EoL_1_combustion"])
                eol_high_lf = _switch_long_from_flows(fl_high, stages=["EoL_2_landfill"])
                eol_high_mix = pd.concat(
                    [
                        eol_high_comb.assign(net=eol_high_comb["net"] * float(hep_w[("high", 1)])),
                        eol_high_lf.assign(net=eol_high_lf["net"] * float(hep_w[("high", 2)])),
                    ],
                    ignore_index=True,
                )

                eol_low_re = _switch_long_from_flows(fl_low, stages=["EoL_2_reuse"])
                eol_low_bio = _switch_long_from_flows(fl_low, stages=["EoL_1_biochar"])
                eol_low_mix = pd.concat(
                    [
                        eol_low_re.assign(net=eol_low_re["net"] * float(lep_w[("low", 2)])),
                        eol_low_bio.assign(net=eol_low_bio["net"] * float(lep_w[("low", 1)])),
                    ],
                    ignore_index=True,
                )

                # Optional: accumulate baselines for verification
                if include_baselines:
                    _switch_accumulate_long(base_high, nets_all_hep, rep_to_i, year_to_j)
                    _switch_accumulate_long(eol_high_mix, nets_all_hep, rep_to_i, year_to_j)
                    _switch_accumulate_long(base_low, nets_all_lep, rep_to_i, year_to_j)
                    _switch_accumulate_long(eol_low_mix, nets_all_lep, rep_to_i, year_to_j)

                # -------------------------
                # push into switch cases
                # -------------------------
                for ys in switch_years:
                    A = nets_by_switch[int(ys)]

                    if int(y0) < int(ys):
                        # cohort produced under HEP base
                        _switch_accumulate_long(base_high, A, rep_to_i, year_to_j)

                        if eol_policy == "as_produced":
                            _switch_accumulate_long(eol_high_mix, A, rep_to_i, year_to_j)
                        else:
                            # follow_lep: EoL years < switch use high EoL,
                            # EoL years >= switch use low EoL
                            _switch_accumulate_long(eol_high_mix[eol_high_mix["year"] < int(ys)], A, rep_to_i, year_to_j)
                            _switch_accumulate_long(eol_low_mix[eol_low_mix["year"] >= int(ys)], A, rep_to_i, year_to_j)

                    else:
                        # cohort produced under LEP base
                        _switch_accumulate_long(base_low, A, rep_to_i, year_to_j)
                        _switch_accumulate_long(eol_low_mix, A, rep_to_i, year_to_j)

        # Convert annual NET to cumulative NET
        cum_arrays: Dict[Tuple[str, Union[int, str]], np.ndarray] = {
            ("switch", int(ys)): nets_by_switch[int(ys)].cumsum(axis=1).astype(np.float32)
            for ys in switch_years
        }
        if include_baselines:
            cum_arrays[("baseline", "all_hep")] = nets_all_hep.cumsum(axis=1).astype(np.float32)
            cum_arrays[("baseline", "all_lep")] = nets_all_lep.cumsum(axis=1).astype(np.float32)

        out_panels[str(S)] = {
            "years_full": years_full,
            "rep_ids": list(reps),
            "cum_arrays": cum_arrays,
        }

        # summary at cut_year (storage in Gt = -NET * 1e-9)
        jcut = int(np.where(years_full == int(cut_year))[0][0])
        for ys in switch_years:
            v_net = cum_arrays[("switch", int(ys))][:, jcut]  # NET (t)
            stor_gt = -(v_net * 1e-9)  # Gt CO2e

            summary_rows += [
                {"panel": str(S), "switch_year": int(ys), "stat": "median", "value": float(np.nanmedian(stor_gt)), "units": "Gt"},
                {"panel": str(S), "switch_year": int(ys), "stat": "p05",    "value": float(np.nanpercentile(stor_gt, 5)), "units": "Gt"},
                {"panel": str(S), "switch_year": int(ys), "stat": "p95",    "value": float(np.nanpercentile(stor_gt, 95)), "units": "Gt"},
            ]

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["panel", "switch_year", "stat"])
        .reset_index(drop=True)
    )

    return {"meta": meta, "panels": out_panels, "summary": summary_df}


# -----------------------------------------------------------------------------
# Save / load (disk)
# -----------------------------------------------------------------------------
def _switch_key_to_str(k: Any) -> str:
    if isinstance(k, tuple) and len(k) == 2:
        return f"{k[0]}__{k[1]}"
    return str(k)


def _switch_str_to_key(s: Any) -> Any:
    if isinstance(s, str) and "__" in s:
        a, b = s.split("__", 1)
        try:
            b = int(b)
        except Exception:
            pass
        return (a, b)
    return s


def save_switch_bundle(bundle: Dict[str, Any], folder: str, name: str) -> None:
    os.makedirs(folder, exist_ok=True)

    # meta: stringify tuple keys inside weights dicts
    meta = dict(bundle.get("meta", {}))
    for k in ("hep_bau_weights", "lep_weights"):
        if isinstance(meta.get(k), dict):
            meta[k] = {_switch_key_to_str(kk): float(vv) for kk, vv in meta[k].items()}

    with open(os.path.join(folder, f"{name}__meta.json"), "w") as f:
        json.dump(meta, f)

    # summary
    summ = bundle.get("summary", None)
    if isinstance(summ, pd.DataFrame):
        summ.to_csv(os.path.join(folder, f"{name}__summary.csv"), index=False)

    # panels
    for S, panel in bundle.get("panels", {}).items():
        pathS = os.path.join(folder, f"{name}__{S}")
        os.makedirs(pathS, exist_ok=True)

        np.save(os.path.join(pathS, "years_full.npy"), np.asarray(panel["years_full"], dtype=int))

        # rep ids (optional but recommended)
        rep_ids = panel.get("rep_ids", None)
        if rep_ids is not None:
            pd.Series(list(rep_ids)).to_csv(os.path.join(pathS, "rep_ids.csv"), index=False, header=False)

        arrays = panel.get("cum_arrays") or {}
        np.savez_compressed(
            os.path.join(pathS, "arrays.npz"),
            **{_switch_key_to_str(k): v for k, v in arrays.items()},
        )


def load_switch_bundle(folder: str, name: str) -> Dict[str, Any]:
    meta_path = os.path.join(folder, f"{name}__meta.json")
    meta = json.load(open(meta_path, "r")) if os.path.exists(meta_path) else {}

    # parse weight keys back to tuples
    for k in ("hep_bau_weights", "lep_weights"):
        if isinstance(meta.get(k), dict):
            meta[k] = {_switch_str_to_key(kk): vv for kk, vv in meta[k].items()}

    summ_path = os.path.join(folder, f"{name}__summary.csv")
    summary = pd.read_csv(summ_path) if os.path.exists(summ_path) else None

    panels: Dict[str, Any] = {}
    prefix = f"{name}__"
    for item in os.listdir(folder):
        if not item.startswith(prefix):
            continue

        panel_key = item.split(prefix, 1)[-1]
        pathS = os.path.join(folder, item)

        yf = os.path.join(pathS, "years_full.npy")
        arrf = os.path.join(pathS, "arrays.npz")
        if not (os.path.exists(yf) and os.path.exists(arrf)):
            continue

        years_full = np.load(yf)
        z = np.load(arrf)

        arrays = {_switch_str_to_key(k): z[k] for k in z.files}

        rep_path = os.path.join(pathS, "rep_ids.csv")
        rep_ids = None
        if os.path.exists(rep_path):
            rep_ids = list(pd.read_csv(rep_path, header=None)[0])

        panels[str(panel_key)] = {"years_full": years_full, "rep_ids": rep_ids, "cum_arrays": arrays}

    return {"meta": meta, "summary": summary, "panels": panels}


# -----------------------------------------------------------------------------
# Patch: extend *saved* switch bundle with future-cohort uptake (2121–2150) cut at 2120
# -----------------------------------------------------------------------------
def extend_switch_bundle_with_future_uptake(
    switch_bundle: Dict[str, Any],
    MAIN_PATH: str,
    *,
    PROJ_BASE: Optional[str] = None,
    last_real_cohort: int = 2120,
    future_last: int = 2150,
    growth_years: int = 21,
    growth_model: str = "richards",
    growth_params: Optional[Dict[str, Any]] = None,
    save_as: Optional[str] = None,
    save_folder: str = "Results/PlotBundles",
) -> Tuple[Dict[str, Any], pd.DataFrame]:

    import copy

    meta_in = switch_bundle.get("meta", {})
    cut_year = int(meta_in.get("cut_year", 2120))
    switch_years = [int(y) for y in meta_in.get("switch_years", [])]
    regions = tuple(meta_in.get("include_regions", ("PNW", "SE")))

    out = {"meta": copy.deepcopy(meta_in), "panels": {}, "summary": None}
    out["meta"]["note"] = meta_in.get("note", "") + f" | uptake patched with cohorts {last_real_cohort+1}–{future_last}"

    checks_rows: list[dict] = []
    summary_rows: list[dict] = []

    for S, P in switch_bundle.get("panels", {}).items():
        # Ensure projections are loaded if we need production replicate IDs
        if PROJ_BASE is not None:
            load_all_projections(PROJ_BASE, scenarios=(S,), products=("CLT", "GLT"))

        years_full = np.asarray(P["years_full"], dtype=int)
        if cut_year not in set(years_full):
            raise ValueError(f"[{S}] cut_year {cut_year} not in years_full horizon {years_full.min()}..{years_full.max()}")
        jcut = int(np.where(years_full == cut_year)[0][0])

        arrays_in: Dict[Any, np.ndarray] = P.get("cum_arrays") or {}
        if not arrays_in:
            raise ValueError(f"[{S}] panel has no cum_arrays")

        # replicate order
        rep_ids = P.get("rep_ids", None)
        if rep_ids is None:
            if PROJ_BASE is None:
                raise ValueError(
                    f"[{S}] rep_ids missing in bundle and PROJ_BASE not provided; "
                    "cannot safely align uptake patch to replicate dimension."
                )
            # infer from production
            rep_ids = list(_get_prod_df("PNW", "CLT", S).index) if "PNW" in regions else list(_get_prod_df("SE", "CLT", S).index)

        # Build amplitude template from *actual* 2120 cohort uptake (tonnes/replicate)
        uptake_template = _total_uptake_by_rep_for_cohort(MAIN_PATH, S, last_real_cohort, regions=regions)

        # Build future-only uptake flow matrix (positive magnitudes), then convert to NET (negative)
        B_pos = _future_uptake_matrix(
            uptake_template,
            years_full,
            last_real_cohort=last_real_cohort,
            future_last=future_last,
            growth_years=growth_years,
            growth_model=growth_model,
            growth_params=growth_params,
            rep_order=rep_ids,
        )
        B_net = (-B_pos).astype(np.float32)
        B_cum = np.cumsum(B_net, axis=1).astype(np.float32)

        # shape checks
        Rpanel, Ypanel = next(iter(arrays_in.values())).shape
        if (Rpanel, Ypanel) != B_cum.shape:
            raise ValueError(f"[{S}] shape mismatch: panel {Rpanel}x{Ypanel} vs patch {B_cum.shape}")

        arrays_out = {k: (A + B_cum).astype(np.float32) for k, A in arrays_in.items()}

        # checks: added uptake flow per year (Gt) in the last growth window up to 2120
        year_to_j = {int(y): j for j, y in enumerate(years_full)}
        for yr in range(max(2000, last_real_cohort - int(growth_years) + 1), last_real_cohort + 1):
            j = year_to_j.get(int(yr), None)
            if j is None:
                continue
            checks_rows.append(
                {"panel": str(S), "year": int(yr), "added_uptake_Gt": float(-B_net[:, j].sum()) * 1e-9}
            )

        # recompute summary (storage = -NET * 1e-9 at cut_year)
        for ys in switch_years:
            key = ("switch", int(ys))
            if key not in arrays_out:
                continue
            v_net = arrays_out[key][:, jcut]
            stor_gt = -(v_net * 1e-9)
            summary_rows += [
                {"panel": str(S), "switch_year": int(ys), "stat": "median", "value": float(np.nanmedian(stor_gt)), "units": "Gt"},
                {"panel": str(S), "switch_year": int(ys), "stat": "p05",    "value": float(np.nanpercentile(stor_gt, 5)), "units": "Gt"},
                {"panel": str(S), "switch_year": int(ys), "stat": "p95",    "value": float(np.nanpercentile(stor_gt, 95)), "units": "Gt"},
            ]

        out["panels"][str(S)] = {"years_full": years_full, "rep_ids": list(rep_ids), "cum_arrays": arrays_out}

    out["summary"] = (
        pd.DataFrame(summary_rows)
        .sort_values(["panel", "switch_year", "stat"])
        .reset_index(drop=True)
    )
    checks_df = (
        pd.DataFrame(checks_rows)
        .sort_values(["panel", "year"])
        .reset_index(drop=True)
    )

    if save_as:
        save_switch_bundle(out, folder=save_folder, name=save_as)

    return out, checks_df


# -----------------------------------------------------------------------------
# Plot: forgone storage bars from summaries (paper-matched style)
# -----------------------------------------------------------------------------
def plot_switch_storage_bars_from_summaries(
    *,
    bundle_for_median: Dict[str, Any],
    bundle_for_p05: Dict[str, Any],
    bundle_for_p95: Dict[str, Any],
    panels: Sequence[str] = ("S1", "S2", "S3"),
    baseline_year: int = 2025,
    fig_size: Tuple[float, float] = (12, 4),
    dpi: int = 600,
    bar_width: float = 0.55,
    letters: Sequence[str] = ("C", "D", "E"),
    letter_xy: Tuple[float, float] = (0.98, 0.96),
    bar_color: str = "#0072B2",
    save_path: Optional[str] = None,
) -> None:

    def _prep(bundle: Dict[str, Any]) -> pd.DataFrame:
        if "summary" not in bundle or bundle["summary"] is None:
            raise ValueError("Bundle is missing a 'summary' DataFrame.")
        df = bundle["summary"].copy()
        df["panel"] = df["panel"].astype(str)
        df["switch_year"] = df["switch_year"].astype(int)
        return df.pivot_table(index=["panel", "switch_year"], columns="stat", values="value", aggfunc="first")

    med_w = _prep(bundle_for_median)
    p05_w = _prep(bundle_for_p05)
    p95_w = _prep(bundle_for_p95)

    # x years: prefer median bundle meta
    meta = bundle_for_median.get("meta", {}) or {}
    years_all = meta.get("switch_years", None)
    if years_all:
        years_all = [int(y) for y in years_all]
    else:
        # fallback: infer from med summary
        years_all = sorted(med_w.index.get_level_values(1).unique().astype(int).tolist())

    # drop baseline year from x axis
    years_all = [y for y in years_all if y != int(baseline_year)]

    # units label (default Gt)
    units = "Gt"
    try:
        df0 = bundle_for_median["summary"]
        if "units" in df0.columns and not df0["units"].isna().all():
            units = str(df0["units"].dropna().iloc[0])
    except Exception:
        pass

    # Fonts 
    FS_AXIS_LABEL = plt.rcParams.get("axes.labelsize", 16)
    FS_TICK = plt.rcParams.get("xtick.labelsize", 14)

    fig, axes = plt.subplots(1, len(panels), figsize=fig_size, dpi=dpi, sharey=True, constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]

    global_hi = 0.0
    global_lo = 0.0

    for i, S in enumerate(panels):
        ax = axes[i]
        S = str(S)

        # slice by panel
        if S not in set(med_w.index.get_level_values(0)):
            raise KeyError(f"Panel {S} not found in median bundle.")
        if S not in set(p05_w.index.get_level_values(0)):
            raise KeyError(f"Panel {S} not found in p05 bundle.")
        if S not in set(p95_w.index.get_level_values(0)):
            raise KeyError(f"Panel {S} not found in p95 bundle.")

        med_S = med_w.xs(S, level=0, drop_level=True)
        p05_S = p05_w.xs(S, level=0, drop_level=True)
        p95_S = p95_w.xs(S, level=0, drop_level=True)

        # baseline values must exist
        if int(baseline_year) not in med_S.index or int(baseline_year) not in p05_S.index or int(baseline_year) not in p95_S.index:
            raise KeyError(
                f"Baseline {baseline_year} missing for {S}. "
                f"Available med years: {sorted(map(int, med_S.index))}"
            )

        b_med = float(med_S.loc[int(baseline_year), "median"])
        b_p05 = float(p05_S.loc[int(baseline_year), "p05"])
        b_p95 = float(p95_S.loc[int(baseline_year), "p95"])

        # intersect years present in all
        years = [y for y in years_all if (y in med_S.index) and (y in p05_S.index) and (y in p95_S.index)]
        if not years:
            raise ValueError(f"No non-baseline years available for {S} after intersecting the three bundles.")

        meds, los, his = [], [], []
        for y in years:
            m = float(med_S.loc[int(y), "median"])
            lo = float(p05_S.loc[int(y), "p05"])
            hi = float(p95_S.loc[int(y), "p95"])

            # Forgone storage vs baseline
            meds.append(b_med - m)
            los.append(b_p05 - lo)
            his.append(b_p95 - hi)

        x = np.arange(len(years))

        # bars
        ax.bar(x, meds, width=bar_width, color=bar_color, edgecolor="black", linewidth=1.0, zorder=3)

        # whiskers
        cap = bar_width * 0.26
        for xi, lo, hi in zip(x, los, his):
            ax.vlines(xi, lo, hi, color="black", linewidth=2.1, zorder=5)
            ax.hlines([lo, hi], xi - cap, xi + cap, color="black", linewidth=2.1, zorder=5)

        # cosmetics (paper-like)
        ax.axhline(0, color="k", lw=1.1, alpha=0.9, zorder=2)
        ax.grid(True, axis="y", linestyle="-", alpha=0.15, linewidth=0.9, zorder=0)

        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(1.2)
            ax.spines[side].set_color("0.2")

        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in years], fontsize=FS_TICK)
        ax.tick_params(axis="y", which="major", labelsize=FS_TICK, length=5.5, width=1.0)
        ax.tick_params(axis="x", which="major", labelsize=FS_TICK, length=5.5, width=1.0)

        ax.set_xlabel("LEP Implementation Year", fontsize=FS_AXIS_LABEL)
        if i == 0:
            ax.set_ylabel(f"Forgone Carbon Storage ({units} CO$_2$e)", fontsize=FS_AXIS_LABEL)
        else:
            ax.set_ylabel("")

        if i < len(letters):
            ax.text(
                *letter_xy,
                letters[i],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=FS_TICK,
                fontweight="bold",
                color="0.15",
            )

        local_max = max(0.0, *(meds + his))
        local_min = min(0.0, *(meds + los))
        global_hi = max(global_hi, local_max)
        global_lo = min(global_lo, local_min)

    # shared y-lims 
    span = max(1e-9, global_hi - global_lo)
    pad_top = 0.06 * span
    for ax in axes:
        ax.set_ylim(0, global_hi + pad_top)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()
