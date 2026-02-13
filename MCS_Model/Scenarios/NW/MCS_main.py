################################### SA-MCs main 
import pandas as pd
import numpy as np
import os
from scipy import stats
from tqdm import tqdm
import random
from MCS_backend_fxns import (
    _resolve_data_root,
    read_input_csv,
    normalize_attribute,
    standardize_col,
    _pnw_pile_burn_CO2_CH4,
    _clip,
    _u_rel,
    logistic_share,
    electrify_fuel_to_electric,
    electrify_fossil_heat_to_HP_or_resistance,
    get_electrification_params,
    get_trucking_params,
    blended_ef_tkm_for_leg,
    compute_transport_emissions,
    make_out_df,
    _read_optional_csv,
)
################################### Reading Data  
DATA_ROOT = _resolve_data_root()

# Wood Properties Data:
prop_data = read_input_csv("Properties", "Results", "properties_variables.csv", index_col=0)
# Apply to dataframe
rename_map = {c: standardize_col(c) for c in prop_data.columns}
prop_data = prop_data.rename(columns={c: standardize_col(c) for c in prop_data.columns})
new_cols = [standardize_col(c) for c in prop_data.columns]
assert len(new_cols) == len(set(new_cols)), "Duplicate column names created—inspect your inputs or refine rules."

# Emission Factor Data:
EF_data_PNW = read_input_csv("EFs", "Results", "EF_var_PNW.csv", index_col=0)
EF_data_SE  = read_input_csv("EFs", "Results", "EF_var_SE.csv",  index_col=0)

# Harvesting Data:
harvesting_data_MA_pnw = read_input_csv("Forestry_Harvesting", "Results", "harvesting_MA_PNW.csv", index_col=0)
harvesting_data_ME_pnw = read_input_csv("Forestry_Harvesting", "Results", "harvesting_ME_PNW.csv", index_col=0)
harvesting_data_MA_se  = read_input_csv("Forestry_Harvesting", "Results", "harvesting_MA_SE.csv",  index_col=0)
harvesting_data_ME_se  = read_input_csv("Forestry_Harvesting", "Results", "harvesting_ME_SE.csv",  index_col=0)

harvest_data_dic = {
    "harvesting_data_MA_PNW": harvesting_data_MA_pnw,
    "harvesting_data_ME_PNW": harvesting_data_ME_pnw,
    "harvesting_data_MA_SE":  harvesting_data_MA_se,
    "harvesting_data_ME_SE":  harvesting_data_ME_se,
}

# Transportation Data:
transport_dir = DATA_ROOT / "Transportation" / "Results"
if not transport_dir.is_dir():
    raise FileNotFoundError(f"Missing directory: {transport_dir}")

# Deterministic ordering + ignore hidden/non-csv files
transportation_files = sorted(
    p.name for p in transport_dir.iterdir()
    if p.is_file() and p.suffix.lower() == ".csv" and not p.name.startswith(".")
)

trans_pd_lst = {
    fname: pd.read_csv(transport_dir / fname, index_col=0)
    for fname in transportation_files
}

# Sawmill Data:
sawmill_data_pnw = read_input_csv("Sawmill", "Results", "sawmill_operations_PNW.csv", index_col=0)
sawmill_data_se  = read_input_csv("Sawmill", "Results", "sawmill_operations_SE.csv",  index_col=0)
sawmill_data_ne  = read_input_csv("Sawmill", "Results", "sawmill_operations_NE.csv",  index_col=0)
sawmill_data_se_hardwood = read_input_csv("Sawmill", "Results", "sawmill_operations_SE_hardwood.csv", index_col=0)

sawmill_data_dic = {
    "sawmill_data_PNW":          sawmill_data_pnw,
    "sawmill_data_SE":           sawmill_data_se,
    "sawmill_data_NE":           sawmill_data_ne,
    "sawmill_data_SE_hardwood":  sawmill_data_se_hardwood,
}

# Product Manufacturing Data:
op_data_clt_pnw    = read_input_csv("Product_Manufacturing", "Results", "product_prod_CLT_pnw.csv", index_col=0)
op_data_clt_se     = read_input_csv("Product_Manufacturing", "Results", "product_prod_CLT_se.csv",  index_col=0)
op_data_clt_ne     = read_input_csv("Product_Manufacturing", "Results", "product_prod_CLT_ne.csv",  index_col=0)

op_data_glulam_pnw = read_input_csv("Product_Manufacturing", "Results", "product_prod_glulam_pnw.csv", index_col=0)
op_data_glulam_se  = read_input_csv("Product_Manufacturing", "Results", "product_prod_glulam_se.csv",  index_col=0)
op_data_glulam_ne  = read_input_csv("Product_Manufacturing", "Results", "product_prod_glulam_ne.csv",  index_col=0)

# NOTE: These two lines in your original snippet appear to be copy/paste mistakes:
# op_data_lvl_pnw = ...glulam_ne.csv
# op_data_lvl_se  = ...glulam_ne.csv
# If you actually have LVL files, point them to those filenames instead:
op_data_lvl_pnw    = read_input_csv("Product_Manufacturing", "Results", "product_prod_lvl_pnw.csv", index_col=0)
op_data_lvl_se     = read_input_csv("Product_Manufacturing", "Results", "product_prod_lvl_se.csv",  index_col=0)

op_data_dic = {
    "op_data_CLT_PNW":    op_data_clt_pnw,
    "op_data_CLT_SE":     op_data_clt_se,
    "op_data_CLT_NE":     op_data_clt_ne,
    "op_data_Glulam_PNW": op_data_glulam_pnw,
    "op_data_Glulam_SE":  op_data_glulam_se,
    "op_data_Glulam_NE":  op_data_glulam_ne,
    "op_data_LVL_PNW":    op_data_lvl_pnw,
    "op_data_LVL_SE":     op_data_lvl_se,
}

# EoL Data:
EoL_1_data = read_input_csv("EoL", "Results", "EoL_scen_1.csv", index_col=0)
EoL_2_data = read_input_csv("EoL", "Results", "EoL_scen_2.csv", index_col=0)

# Biogenic CO2 Data:
bio_data = read_input_csv("Biomass_species", "Results", "biomass_cc.csv", index_col=0)


################################### Delcaration of Model Paramaters
sim_nums = 10000
region_   = "PNW" # options: {'PNW', 'SE'} - PNW (old naming convention) refers to NW 
pre_wood_ = "lumber_" # options: {'lumber_'}
pre_type_ = "softwood_" # options: {'softwood_'}
prod_type_ = 'CLT' # options: {'CLT','Glulam'}
emission_mode = 'low' #options: {'low', 'high'}
emission_mode_lst = ['low','high']
analysis_years = np.arange(2020,2121,1)  

# Run
# ==============================================================================================================
for analysis_year in tqdm(analysis_years):
    
    for jj in (range(len(emission_mode_lst))):
        emission_mode = emission_mode_lst[jj]
    
       ################################ Generate Collection lists
        contrib_rows_harvest = []
        contrib_rows_sawmill = []
        contrib_rows_product = []
        contrib_rows_EoL = []
        contrib_rows_bio = []
        residue_co2_ts_rows = []
        landfill_ts_rows = []   

        
        ################################ Choose Harvesting Data:
        pre_all_  = pre_wood_ + pre_type_ + region_ + '_'
        if emission_mode == 'low':
            harvest_method = "ME"
        elif emission_mode == 'high':
            harvest_method = "MA"
        harvest_data  = harvest_data_dic['harvesting_data_'+harvest_method+'_'+region_]
        
        ################################ Choose Sawmill Data:
        if pre_type_ == 'hardwood_':
            sawmill_data = sawmill_data_dic['sawmill_data_SE_hardwood']
        else:
            sawmill_data = sawmill_data_dic['sawmill_data_'+region_]
        
        # Designate Drying Scenarios
        if emission_mode == 'low':
            biomass_drying = "high"
        elif emission_mode == 'high':
            biomass_drying = "low"
        
        ################################ Choose Operation Data:
        op_data = op_data_dic['op_data_'+prod_type_+'_'+region_]
        
        ################################ Choose Transportation Data: 
        if pre_type_ == 'softwood_':
            wood_type = 'S'
        elif pre_type_ == 'hardwood_':
            wood_type = 'H'
        
        if ((region_+'_'+prod_type_ == "PNW_CLT") or (region_+'_'+prod_type_ == "PNW_LVL") or (region_+'_'+prod_type_ == "PNW_Glulam")):
            trans_name = "trans_"+region_+'_'+prod_type_+".csv"
        else:
            trans_name = "trans_"+region_+'_'+prod_type_+"_"+wood_type+".csv"
        
        trans_data = trans_pd_lst[trans_name]
        
        ################################ Choose Biogenic Data:
        if pre_type_ == 'softwood_':
            bio_data_col = bio_data[region_+'_'+'final_cc']
        elif pre_type_ == 'hardwood_':
            bio_data_col = bio_data[region_+'_hardwood_'+'final_cc']
    
        ################################ Choose Biogenic Data:
        if region_ == 'PNW':
            EF_data = EF_data_PNW.copy()
        elif region_ == 'SE':
            EF_data = EF_data_SE.copy()
        
        for i in (range(sim_nums)):
            ############################################ Harvesting ############################################
            CLT_wood = prop_data[pre_all_ + "wood_data"][i]                
            wood_mass_req = prop_data[pre_all_ + "mass_req_data"][i]       
            wood_vol_req = wood_mass_req / CLT_wood                        
            green_wood_mass_req = prop_data[pre_all_ + "green_wood_mass_req"][i]   
            wood_loss_efficiency = prop_data[pre_all_ + "loss_efficiency"][i]      

            # ----  sawmill yield 
            if region_ == 'PNW':
                Y_saw = float(np.random.triangular(0.44, 0.46, 0.52))
            elif region_ == 'SE':
                Y_saw = float(np.random.triangular(0.46, 0.48, 0.5))

            # Harvesting inputs per tonne of green wood
            gasoline_L = harvest_data[f'gasoline_harvesting_L_per_t_greenwood_{harvest_method}'][i]
            diesel_planting_L   = harvest_data['diesel_planting_L_per_t_greenwood'][i]
            diesel_harvesting_L = harvest_data[f'diesel_harvesting_L_per_t_greenwood_{harvest_method}'][i]
            diesel_L = diesel_planting_L + diesel_harvesting_L
        
            lubricant_planting_L   = harvest_data['lubricants_planting_L_per_t_greenwood'][i]
            lubricant_harvesting_L = harvest_data[f'lubricants_harvesting_L_per_t_greenwood_{harvest_method}'][i]
            lubricant_L = lubricant_planting_L + lubricant_harvesting_L
        
            harvesting_equipment_kg = harvest_data[f'harvesting_equipment_harvesting_kg_per_t_greenwood_{harvest_method}'][i]
            electric_planting_MJ    = harvest_data['electric_planting_MJ_per_t_greenwood'][i]
        
            # Fertilizer & herbicide per tonne of green wood
            nitrogen_kg  = harvest_data['nitrogen_kg_per_t_greenwood'][i]
            phosphate_kg = harvest_data['phosphate_kg_per_t_greenwood'][i]
            potassium_kg = harvest_data['potassium_kg_per_t_greenwood'][i]
            herbicide_kg = harvest_data['herbicide_kg_per_t_greenwood'][i]
        
            # Scale to per m3 CLT via your reqs/efficiency
            scale = (green_wood_mass_req / wood_vol_req) * (1 / wood_loss_efficiency)
        
            # Densities
            gasoline_kg_L  = 0.7429
            diesel_kg_L    = 0.85
            lubricant_kg_L = 0.825
        
            gasoline_kg = gasoline_L * gasoline_kg_L * scale
            diesel_kg   = diesel_L   * diesel_kg_L   * scale
            lubricant_kg = lubricant_L * lubricant_kg_L * scale
            harvesting_equipment_kg = harvesting_equipment_kg * scale
            electric_planting_MJ    = electric_planting_MJ    * scale
        
            nitrogen_kg  = nitrogen_kg  * scale
            phosphate_kg = phosphate_kg * scale
            potassium_kg = potassium_kg * scale
            herbicide_kg = herbicide_kg * scale
        
            # Heating values (MJ/kg)
            gasoline_MJ_kg = 45.8
            diesel_MJ_kg   = 45.5
            gasoline_MJ = gasoline_MJ_kg * gasoline_kg
            diesel_MJ   = diesel_MJ_kg   * diesel_kg
        
            # Emission factors 
            gasoline_kgCO2_per_MJ = EF_data.gasoline_kg_CO2_MJ[i]
            diesel_kgCO2_per_MJ   = EF_data.diesel_kg_CO2_MJ[i]
            lubricant_kgCO2_per_kg = EF_data.lubricant_kgCO2_per_kg[i]
            equipment_kgCO2_per_kg = 5.01 
            ################################################### Fix Based on Year
            elec_kgCO2_per_MJ      = EF_data['EF_kgCO2e_per_MJ_'+emission_mode+'_'+region_][i] * EF_data['grid_factor_'+str(analysis_year)][i]
            # --- EoL grid EF adjustment: use year + 50 (capped at 2120) ---
            eol_year = min(2120, int(analysis_year) + 50)
            elec_kgCO2_per_MJ_EoL = EF_data['EF_kgCO2e_per_MJ_'+emission_mode+'_'+region_][i] * EF_data['grid_factor_'+str(eol_year)][i]
            # ---------------------------------------------------------------
            propane_kgCO2_per_gallon = EF_data.propane_kgCO2_per_gallon[i]
            biomass_kg_CO2_MJ = EF_data.biomass_kg_CO2_MJ[i]
            NG_kg_CO2_MJ = EF_data.NG_kg_CO2_MJ[i]
        
            # fert/herbicide EF 
            herbicide_kgCO2_per_kg = EF_data.herbicide_kgCO2_per_kg[i]
            nitrogen_kgCO2_per_kg  = EF_data.nitrogen_kgCO2_per_kg[i]
            phosphate_kgCO2_per_kg = EF_data.phosphate_kgCO2_per_kg[i]
            potassium_kgCO2_per_kg = EF_data.potassium_kgCO2_per_kg[i]
    
            # Electrification Paramaters 
            elec_cfg = get_electrification_params(analysis_year, emission_mode)
            trucking_cfg = get_trucking_params(analysis_year, region_, emission_mode)
    
            diesel_MJ, elec_add_harv_diesel_MJ = electrify_fuel_to_electric(
                fuel_MJ=diesel_MJ, share_to_elec=elec_cfg['harv_BE_share'], service_ratio=elec_cfg['EER_harv_BE']
            )
            gasoline_MJ, elec_add_harv_gas_MJ = electrify_fuel_to_electric(
                fuel_MJ=gasoline_MJ, share_to_elec=min(0.5, elec_cfg['harv_BE_share']), service_ratio=0.40
            )
            electric_planting_MJ = electric_planting_MJ + elec_add_harv_diesel_MJ + elec_add_harv_gas_MJ
    
            # Per-process emissions (kg CO2)
            gasoline_CO2   = gasoline_MJ * gasoline_kgCO2_per_MJ
            diesel_CO2     = diesel_MJ   * diesel_kgCO2_per_MJ
            electricity_CO2 = electric_planting_MJ * elec_kgCO2_per_MJ
            lubricant_CO2  = lubricant_kg * lubricant_kgCO2_per_kg
            equipment_CO2  = harvesting_equipment_kg * equipment_kgCO2_per_kg
        
            herbicide_CO2 = herbicide_kg * herbicide_kgCO2_per_kg
            nitrogen_CO2  = nitrogen_kg  * nitrogen_kgCO2_per_kg
            phosphate_CO2 = phosphate_kg * phosphate_kgCO2_per_kg
            potassium_CO2 = potassium_kg * potassium_kgCO2_per_kg
        
            # Subtotals and total
            harvest_ops_CO2 = gasoline_CO2 + diesel_CO2 + electricity_CO2 + lubricant_CO2 + equipment_CO2
            fert_herb_CO2   = herbicide_CO2 + nitrogen_CO2 + phosphate_CO2 + potassium_CO2
        
            # Transportation Harvesting:
            haul_km = trans_data["logs_2_sawmill_"+emission_mode][i]
            truck_ef_kgCO2_per_tkm = EF_data.truck_ef_kgCO2_per_tkm[i]   # diesel baseline EF (kgCO2/tkm)
            empty_backhaul_factor = EF_data['empty_backhaul_factor_'+emission_mode][i]
    
            # blended diesel-BEV:
            payload_t_per_m3_harv = (green_wood_mass_req / wood_vol_req) * (1.0 / wood_loss_efficiency)
            harvest_haul_CO2 = compute_transport_emissions(
                distance_km=haul_km,
                payload_t_per_m3=payload_t_per_m3_harv,
                empty_backhaul_factor=empty_backhaul_factor,
                diesel_ef_tkm=truck_ef_kgCO2_per_tkm,
                elec_kgCO2_per_MJ=elec_kgCO2_per_MJ,
                leg_name='logs2sawmill',
                tr_cfg=trucking_cfg,
                year=analysis_year)
            
            total_CO2       = harvest_ops_CO2 + fert_herb_CO2 
            total_CO2_w_transp       = harvest_ops_CO2 + fert_herb_CO2 + harvest_haul_CO2
            
            contrib_rows_harvest.append({
                "gasoline_CO2": gasoline_CO2,
                "diesel_CO2": diesel_CO2,
                "electricity_CO2": electricity_CO2,
                "lubricant_CO2": lubricant_CO2,
                "equipment_CO2": equipment_CO2,
                "herbicide_CO2": herbicide_CO2,
                "nitrogen_CO2": nitrogen_CO2,
                "phosphate_CO2": phosphate_CO2,
                "potassium_CO2": potassium_CO2,
                "harvest_ops_CO2": harvest_ops_CO2,
                "fert_herb_CO2":   fert_herb_CO2,
                "harvest_haul_CO2": harvest_haul_CO2,
                "total_harvest_CO2": total_CO2_w_transp})
        
            ############################################ Sawmill ############################################
            elec_kwh_m3 = sawmill_data.elec_kwh_m3[i]
            elec_kwh_m3_drying = sawmill_data.elec_kwh_m3_drying[i]
            diesel_kg_m3 = sawmill_data.diesel_kg_m3[i]
            diesel_kg_m3_drying = sawmill_data.diesel_kg_m3_drying[i]
            gasoline_kg_m3 = sawmill_data.gasoline_kg_m3[i]
            propane_gal_m3 = sawmill_data.propane_gal_m3[i]
            biomass_prop_drying_ = sawmill_data['biomass_prop_drying_'+biomass_drying][i]
            biomass_MJ_m3_drying = biomass_prop_drying_*sawmill_data.total_MJ_m3_drying[i]
            NG_MJ_m3_drying = sawmill_data.total_MJ_m3_drying[i] - biomass_MJ_m3_drying
        
            # Get total energy requirements 
            elec_kwh_m3 = elec_kwh_m3*wood_vol_req
            elec_kwh_m3_drying = elec_kwh_m3_drying*wood_vol_req
            diesel_kg_m3 = diesel_kg_m3*wood_vol_req
            diesel_kg_m3_drying = diesel_kg_m3_drying*wood_vol_req
            gasoline_kg_m3 = gasoline_kg_m3*wood_vol_req
            propane_gal_m3 = propane_gal_m3*wood_vol_req
            biomass_MJ_m3_drying = biomass_MJ_m3_drying*wood_vol_req
            NG_MJ_m3_drying = NG_MJ_m3_drying*wood_vol_req
        
            # Fix any units 
            elec_MJ_m3 = elec_kwh_m3*3.6
            elec_MJ_m3_drying = elec_kwh_m3_drying*3.6
            diesel_MJ_m3 = diesel_kg_m3*diesel_MJ_kg
            diesel_MJ_m3_drying = diesel_kg_m3_drying*diesel_MJ_kg
            gasoline_MJ_m3 = gasoline_kg_m3*gasoline_MJ_kg
        
            # Sawmill Electrification 
            diesel_MJ_m3, elec_add_yard_MJ = electrify_fuel_to_electric(
                fuel_MJ=diesel_MJ_m3, share_to_elec=elec_cfg['yard_BE_share'], service_ratio=elec_cfg['EER_yard_BE']
            )
            elec_MJ_m3 = elec_MJ_m3 + elec_add_yard_MJ
    
            gasoline_MJ_m3, elec_add_gas_mill_MJ = electrify_fuel_to_electric(
                fuel_MJ=gasoline_MJ_m3, share_to_elec=min(0.5, elec_cfg['yard_BE_share']), service_ratio=0.40
            )
            elec_MJ_m3 = elec_MJ_m3 + elec_add_gas_mill_MJ
    
            NG_MJ_m3_drying, elec_HP_kiln_MJ, elec_RES_kiln_MJ = electrify_fossil_heat_to_HP_or_resistance(
                fossil_heat_MJ=NG_MJ_m3_drying,
                share_HP=elec_cfg['kiln_HP_share'], share_RES=elec_cfg['kiln_RES_share'],
                COP_HP=elec_cfg['COP_kiln_HP'], eta_RES=elec_cfg['eta_resistance']
            )
            diesel_MJ_m3_drying, elec_HP_kiln_diesel_MJ, elec_RES_kiln_diesel_MJ = electrify_fossil_heat_to_HP_or_resistance(
                fossil_heat_MJ=diesel_MJ_m3_drying,
                share_HP=elec_cfg['kiln_HP_share'], share_RES=elec_cfg['kiln_RES_share'],
                COP_HP=elec_cfg['COP_kiln_HP'], eta_RES=elec_cfg['eta_resistance']
            )
            elec_MJ_m3_drying = elec_MJ_m3_drying + elec_HP_kiln_MJ + elec_RES_kiln_MJ + elec_HP_kiln_diesel_MJ + elec_RES_kiln_diesel_MJ
            # ==============================================================================================
    
            # Calculate CO2 emissions 
            saw_elec_kgCO2_per_m3 = elec_MJ_m3*elec_kgCO2_per_MJ
            saw_elec_drying_kgCO2_per_m3 = elec_MJ_m3_drying*elec_kgCO2_per_MJ
            saw_diesel_kgCO2_per_m3 = diesel_MJ_m3*diesel_kgCO2_per_MJ
            saw_diesel_drying_kgCO2_per_m3 = diesel_MJ_m3_drying*diesel_kgCO2_per_MJ
            saw_gasoline_kgCO2_per_m3 = gasoline_MJ_m3*gasoline_kgCO2_per_MJ
            propane_kgCO2_per_m3 = propane_gal_m3*propane_kgCO2_per_gallon
            biomass_kgCO2_per_m3_biogenic = biomass_MJ_m3_drying*biomass_kg_CO2_MJ
            NG_kgCO2_per_m3 = NG_MJ_m3_drying*NG_kg_CO2_MJ
        
            # Tabulate Emissions
            NG_kgCO2_per_m3 = max(0.0, NG_kgCO2_per_m3)
            
            # Subtotals by fuel 
            sawmill_electricity_CO2 = saw_elec_kgCO2_per_m3 + saw_elec_drying_kgCO2_per_m3
            sawmill_diesel_CO2      = saw_diesel_kgCO2_per_m3 + saw_diesel_drying_kgCO2_per_m3
            sawmill_gasoline_CO2    = saw_gasoline_kgCO2_per_m3
            sawmill_propane_CO2     = propane_kgCO2_per_m3
            sawmill_ng_CO2          = NG_kgCO2_per_m3
            
            # Biogenic kept SEPARATE (do not include in fossil totals)
            sawmill_biomass_CO2_biogenic = biomass_kgCO2_per_m3_biogenic
            
            # Sawmill operations total (fossil)
            sawmill_ops_CO2 = (
                sawmill_electricity_CO2
              + sawmill_diesel_CO2
              + sawmill_gasoline_CO2
              + sawmill_propane_CO2
              + sawmill_ng_CO2)
            
            # Transportation: Sawmill 2 CLT
            haul_km_lumber = trans_data[f"lumber_2_prod_{emission_mode}"][i]
            shipping_mc_factor = 1.0
            payload_t_per_m3 = (wood_mass_req / 1000.0) * shipping_mc_factor
            
            # blended diesel-BEV:
            sawmill_to_prod_haul_CO2 = compute_transport_emissions(
                distance_km=haul_km_lumber,
                payload_t_per_m3=payload_t_per_m3,
                empty_backhaul_factor=empty_backhaul_factor,
                diesel_ef_tkm=truck_ef_kgCO2_per_tkm,
                elec_kgCO2_per_MJ=elec_kgCO2_per_MJ,
                leg_name='lumber2prod',
                tr_cfg=trucking_cfg,
                year=analysis_year)
            
            total_sawmill_CO2_fossil      = sawmill_ops_CO2 + sawmill_to_prod_haul_CO2
            total_sawmill_CO2_incl_biogen = total_sawmill_CO2_fossil + sawmill_biomass_CO2_biogenic
            
            contrib_rows_sawmill.append({
                # electricity
                "sawmill_electricity_CO2": sawmill_electricity_CO2,
                "sawmill_elec_process_CO2":    saw_elec_kgCO2_per_m3,
                "sawmill_elec_drying_CO2":     saw_elec_drying_kgCO2_per_m3,
            
                # liquid fuels
                "sawmill_diesel_CO2":      sawmill_diesel_CO2,
                "sawmill_diesel_process_CO2":  saw_diesel_kgCO2_per_m3,
                "sawmill_diesel_drying_CO2":   saw_diesel_drying_kgCO2_per_m3,
                "sawmill_gasoline_CO2":    sawmill_gasoline_CO2,
            
                # stationary fuels
                "sawmill_propane_CO2":     sawmill_propane_CO2,
                "sawmill_ng_CO2":          sawmill_ng_CO2,
            
                # biogenic 
                "sawmill_biomass_CO2_biogenic": sawmill_biomass_CO2_biogenic,
                "sawmill_biomass_MJ_drying": biomass_MJ_m3_drying,   # <--- add this line
                
            
                # ops subtotal and transport
                "sawmill_ops_CO2":         sawmill_ops_CO2,
                "sawmill_haul_CO2":        sawmill_to_prod_haul_CO2,
            
                # totals
                "total_sawmill_CO2_fossil":      total_sawmill_CO2_fossil})
        
            ############################################ Product Manuf ############################################
            resin_per_m3 = op_data.resin_per[i]
            resin_kg_m3 = ((resin_per_m3)/(1-resin_per_m3))*CLT_wood
            resin_kg_CO2_kg = EF_data['resin_kg_CO2_kg_'+emission_mode][i]
            total_mass_kg_m3 = resin_kg_m3 + CLT_wood
            elec_MJ_m3 = op_data.elec_kwh[i]*3.6
            ng_MJ_m3 = op_data.ng_m3[i]*38.5
            diesel_MJ_m3 = op_data.diesel_kg_m3[i]*diesel_MJ_kg
            gasoline_MJ_m3 = op_data.gasoline_kg_m3[i]*gasoline_MJ_kg
            propane_gallon_m3 = op_data.propane_gallon_m3[i]
    
            # CLT Electrification 
            ng_MJ_m3, elec_HP_clt_MJ, elec_RES_clt_MJ = electrify_fossil_heat_to_HP_or_resistance(
                fossil_heat_MJ=ng_MJ_m3,
                share_HP=elec_cfg['clt_HP_share'], share_RES=elec_cfg['clt_RES_share'],
                COP_HP=elec_cfg['COP_clt_HP'], eta_RES=elec_cfg['eta_resistance']
            )
            diesel_MJ_m3, elec_add_prod_diesel_MJ = electrify_fuel_to_electric(
                fuel_MJ=diesel_MJ_m3, share_to_elec=min(0.5, elec_cfg['yard_BE_share']), service_ratio=elec_cfg['EER_yard_BE']
            )
            gasoline_MJ_m3, elec_add_prod_gas_MJ = electrify_fuel_to_electric(
                fuel_MJ=gasoline_MJ_m3, share_to_elec=min(0.5, elec_cfg['yard_BE_share']), service_ratio=0.40
            )
            elec_MJ_m3 = elec_MJ_m3 + elec_HP_clt_MJ + elec_RES_clt_MJ + elec_add_prod_diesel_MJ + elec_add_prod_gas_MJ
            # ================================================================================================
    
            # Get production emissions 
            op_resin_kgCO2_per_m3 = resin_kg_m3*resin_kg_CO2_kg
            op_elec_kgCO2_per_m3 = elec_MJ_m3*elec_kgCO2_per_MJ
            op_ng_kgCO2_per_m3 = ng_MJ_m3*NG_kg_CO2_MJ
            op_diesel_kgCO2_per_m3 = diesel_MJ_m3*diesel_kgCO2_per_MJ
            op_gasoline_kgCO2_per_m3 = gasoline_MJ_m3*gasoline_kgCO2_per_MJ
            op_propane_kgCO2_per_m3 = propane_gallon_m3*propane_kgCO2_per_gallon
        
            # Get Transportation Emissions 
            haul_km_lumber = trans_data[f"prod_2_const_{emission_mode}"][i]
            payload_t_per_m3 = (total_mass_kg_m3 / 1000.0) 
            
            # Blended diesel-BEV:
            prod_to_const_haul_CO2 = compute_transport_emissions(
                distance_km=haul_km_lumber,
                payload_t_per_m3=payload_t_per_m3,
                empty_backhaul_factor=empty_backhaul_factor,
                diesel_ef_tkm=truck_ef_kgCO2_per_tkm,
                elec_kgCO2_per_MJ=elec_kgCO2_per_MJ,
                leg_name='prod2const',
                tr_cfg=trucking_cfg,
                year=analysis_year)
            
            # Sum Total:
            operation_ops_CO2 = (
                op_resin_kgCO2_per_m3
              + op_elec_kgCO2_per_m3
              + op_ng_kgCO2_per_m3
              + op_diesel_kgCO2_per_m3
              + op_gasoline_kgCO2_per_m3
              + op_propane_kgCO2_per_m3)
            
            contrib_rows_product.append({
                "op_resin_CO2": op_resin_kgCO2_per_m3,
                "op_electricity_CO2":    op_elec_kgCO2_per_m3,
                "op_ng_CO2":     op_ng_kgCO2_per_m3,
                "op_diesel_CO2":      op_diesel_kgCO2_per_m3,
                "op_gasoline_CO2":  op_gasoline_kgCO2_per_m3,
                "op_propane_CO2":    op_propane_kgCO2_per_m3,
                "op_haul_CO2":        prod_to_const_haul_CO2,
    
                # totals
                "total_op_process_CO2_fossil":      operation_ops_CO2,
                "total_operation_CO2_fossil": operation_ops_CO2 + prod_to_const_haul_CO2
            })
        
            ############################################ EoL ############################################
            CREDIT_PYRO_HEAT = False   
            PYRO_EXPORT_SHARE = 0.0    
            # -------------------------------------------------------------------------------------

            #### Scenario 1: Biomass Combustion for Electricity
            carbon_prop = bio_data_col[i] 
            EoL_1_bio_CO2 = CLT_wood*carbon_prop*(44/12)
            bio_pplant_eff = EoL_1_data.bio_DC_eff[i]/100 
            bio_HV = EoL_1_data.bio_HV[i] #MJ/kg
            grid_EF_kwh = elec_kgCO2_per_MJ_EoL*3.6 #kgCO2/kWh
            energy_input = bio_pplant_eff*CLT_wood*bio_HV/3.6 #kWh
            resin_comb_CO2 = resin_kg_m3*EoL_1_data['resin_carb_'+emission_mode][i]*(44/12)
            EoL_1_avoided_fossil_co2 = energy_input*grid_EF_kwh 
        
            #### Scenario 2: Landfill Disposal
            wood_mass = float(CLT_wood)  # kg/m3 of CLT wood entering landfill at EoL
            carbon_prop_landfill   = float(EoL_2_data.carbon_prop_landfill[i])   # fraction of C to CO2
            methane_prop_landfill  = float(EoL_2_data.methane_prop_landfill[i])  # fraction of C to CH4
            wood_decomp_in_landfill= float(EoL_2_data.wood_decomp_in_landfill[i])# decomposable fraction
            LFG_C   = float(EoL_2_data.LFG_C[i]) / 100.0   # capture share
            LFG_R   = float(EoL_2_data.LFG_R[i]) / 100.0   # of captured gas routed to energy (vs. flare)
            LFG_HHV = float(EoL_2_data.LFG_HHV[i])         # MJ/kg landfill gas (as modeled in your input)
            LFG_elec_eff = float(EoL_2_data.LFG_elec_eff[i])
            
            # Totals
            CO2_DE_KG = wood_mass * carbon_prop * carbon_prop_landfill * wood_decomp_in_landfill * (1.0 - LFG_C) * (44.0/12.0)
            CH4_DE_KG = wood_mass * carbon_prop * methane_prop_landfill *  wood_decomp_in_landfill * (1.0 - LFG_C) * (16.0/12.0)
            CO2_energy_recovery_KG = wood_mass * carbon_prop * wood_decomp_in_landfill * (LFG_C) * (LFG_R) * (44.0/12.0)
            CO2_flaring_KG         = wood_mass * carbon_prop * wood_decomp_in_landfill * (LFG_C) * (1.0 - LFG_R) * (44.0/12.0)
            
            # Avoided fossil electricity CO2 
            LFG_energy_MJ = (
                LFG_HHV
              * wood_mass * carbon_prop * wood_decomp_in_landfill * LFG_C * LFG_R
              * ( (44.0/12.0)*carbon_prop_landfill + (16.0/12.0)*methane_prop_landfill )
            )
            EoL_2_avoided_fossil_co2 = float(elec_kgCO2_per_MJ_EoL) * LFG_energy_MJ * LFG_elec_eff
            
            # Data EoL Collect
            EoL_2_bio_CO2  = CO2_DE_KG + CO2_energy_recovery_KG + CO2_flaring_KG
            EoL_2_fossil_CH4 = CH4_DE_KG
            
            contrib_rows_EoL.append({
                "EoL_1_fossil_CO2": resin_comb_CO2,
                "EoL_1_avoided_fossil_CO2": EoL_1_avoided_fossil_co2,
                "EoL_1_bio_CO2":    EoL_1_bio_CO2,
                "EoL_2_bio_CO2": EoL_2_bio_CO2,
                "EoL_2_avoided_fossil_CO2": EoL_2_avoided_fossil_co2,
                "EoL_2_fossil_CH4": EoL_2_fossil_CH4,
                "Wood_mass_biochar_kg_m3": CLT_wood
            })
            

            # ------------------  Landfill time series ------------------
            Y_HORIZON = 200
            
            # Sample a half-life and compute first-order decay fractions 
            if region_ == "PNW":
                hl_landfill = float(np.random.triangular(10.0, 25.0, 60.0))
            else:  # "SE"
                hl_landfill = float(np.random.triangular(8.0, 20.0, 50.0))
            
            k_landfill = np.log(2.0) / max(1e-12, hl_landfill)
            t = np.arange(1, Y_HORIZON + 1, dtype=float)
            fod_frac = np.exp(-k_landfill*(t - 1.0)) - np.exp(-k_landfill*t) 
            
            # Per-year landfill flows 
            co2_ts  = float(EoL_2_bio_CO2)            * fod_frac
            ch4_ts  = float(EoL_2_fossil_CH4)         * fod_frac
            cred_ts = float(EoL_2_avoided_fossil_co2) * fod_frac
            
            landfill_row = {"simulation": i, "hl_landfill_y": hl_landfill, "k_landfill": float(k_landfill)}
            for j, y in enumerate(range(1, Y_HORIZON + 1)):
                landfill_row[f"co2_year_{y}"]  = float(co2_ts[j])   # biogenic CO2 from landfill (kg)
                landfill_row[f"ch4_year_{y}"]  = float(ch4_ts[j])   # methane mass (kg CH4)
                landfill_row[f"cred_year_{y}"] = float(cred_ts[j])  # avoided fossil CO2 credit (kg CO2)
            landfill_ts_rows.append(landfill_row)
            # ---------------- end landfill time series ----------------

        
            ############################################ Biogenic Data ############################################
         
            RESIDUE_DEF = 'share_total'   
            
            # Product carbon storage (in the CLT panel) 
            carbon_stor = carbon_prop * float(CLT_wood) * (44.0/12.0)  # kgCO2 in 1 m3 CLT
            
            # Yields
            Y_clt = float(wood_loss_efficiency)           # lumber 2 CLT yield (OD)
            Y_clt = np.clip(Y_clt, 0.60, 0.98)           
            
            # Mass bookkeeping (OD and green)
            m_CLT_OD        = float(CLT_wood)             # kg OD in finished CLT per m3
            m_lumber_in_OD  = m_CLT_OD / Y_clt            # kg OD lumber into CLT plant per m3
            m_log_OD        = m_lumber_in_OD / Y_saw      # kg OD logs harvested per m3
            
            MCwb_log   = float(EF_data['MCwb_'+region_][i])  # green log MC (wet-basis)
            MCwb_slash = MCwb_log                             
            m_log_green = m_log_OD / max(1e-9, (1.0 - MCwb_log))  
            
            # Harvest slash at felling site
            R = float(EF_data.residue_per_green_wood[i])   
            if RESIDUE_DEF == 'share_total':
                residue_green_kg = m_log_green * (R / max(1e-9, (1.0 - R)))
            else:  # 'per_log'
                residue_green_kg = m_log_green * R
            
            residue_mass_kg = residue_green_kg * max(0.0, 1.0 - MCwb_slash)   # OD kg
            Sequestered_CO2_residue = residue_mass_kg * carbon_prop * (44.0/12.0)
            
            # Decay parameters
            residue_decay_mass_fraction = EF_data['residue_decay_'+region_][i]
            residue_kg_CO2_biogenic = residue_decay_mass_fraction * residue_mass_kg * carbon_prop * (44.0/12.0)
            
            # Sawmill + CLT-plant residues (OD)
            m_res_sawmill_OD = max(0.0, m_log_OD       - m_lumber_in_OD)  # wood-only residues at sawmill
            m_res_clt_OD     = max(0.0, m_lumber_in_OD - m_CLT_OD)        # residues at CLT plant
            
            # Bark (OD) as share of wood-only OD at the log (region-dependent)
            bark_ratio_OD = (np.random.uniform(0.07, 0.12) if region_ == 'PNW'
                             else np.random.uniform(0.08, 0.14))
            bark_OD = m_log_OD * bark_ratio_OD
            
            m_res_sawmill_total_OD = m_res_sawmill_OD + bark_OD
            
            CO2_res_sawmill = m_res_sawmill_total_OD * carbon_prop * (44.0/12.0)
            CO2_res_clt     = m_res_clt_OD            * carbon_prop * (44.0/12.0)
            CO2_res_total   = CO2_res_sawmill + CO2_res_clt
            
            # Residue fate by region 
            Residue_new_CO2_val = 0.0
            Residue_new_CH4_val = 0.0
            Residue_new_CO2_100_val = 0.0
            
            if region_ == "PNW":
                co2_kg, ch4_kg = _pnw_pile_burn_CO2_CH4(residue_mass_kg, consumption_fraction=1.0)
                Residue_new_CO2_val = co2_kg
                Residue_new_CH4_val = ch4_kg
                Residue_new_CO2_100_val = 0.0
            

            elif region_ == "SE":
                # Two-pool exponential decay of slash; record only per-year CO2 to air (no sums)
                kF = float(np.random.uniform(0.011, 0.076))  # FWD 1/yr
                kC = float(np.random.uniform(0.048, 0.127))  # CWD 1/yr
                fC = float(np.random.uniform(0.42, 0.69))    # CWD mass fraction
                fF = 1.0 - fC
            
                YEARS_SLASH = 200
                years = np.arange(0, YEARS_SLASH + 1, dtype=float)
            
                # Residue pool (OD kg) and carbon content
                residue_mass_kg = residue_green_kg * max(0.0, 1.0 - MCwb_slash)  # OD kg
                frac_to_air = 0.76  # aerobic share to atmosphere
            
                # Remaining mass
                M_t = residue_mass_kg * (fF*np.exp(-kF*years) + fC*np.exp(-kC*years))
                decayed_mass = M_t[:-1] - M_t[1:]                 # kg/yr
                C_to_air     = decayed_mass * carbon_prop * frac_to_air
                CO2_to_air   = C_to_air * (44.0/12.0)            # kg CO2/yr
            
                residue_co2_ts_rows.append(
                    {"simulation": i, **{f"year_{y}": float(CO2_to_air[y-1]) for y in range(1, YEARS_SLASH + 1)}}
                )
            
                Residue_new_CO2_val = 0.0
                Residue_new_CH4_val = 0.0
                Residue_new_CO2_100_val = 0.0
                        
            # Append
            contrib_rows_bio.append({
                "Sequestered_CO2": carbon_stor,
                "residue_mass_kg": residue_mass_kg,                      # OD kg slash
                "Sequestered_CO2_residue": Sequestered_CO2_residue,      # initial stock
                "Residue_CO2": residue_kg_CO2_biogenic,                  # modeled biogenic emissions 
            
                "Mill_residue_CO2": CO2_res_total,                       # includes bark
                "Sawmill_residue_CO2": CO2_res_sawmill,
                "CLT_residue_CO2": CO2_res_clt,
            
            })
            contrib_rows_bio[-1]["Residue_new_CO2"] = Residue_new_CO2_val
            contrib_rows_bio[-1]["Residue_new_CH4"] = Residue_new_CH4_val
            contrib_rows_bio[-1]["Residue_new_CO2_100"] = Residue_new_CO2_100_val

        
        ############################################  Data Collect ####################
        harvest_df = pd.DataFrame(contrib_rows_harvest)
        sawmill_df = pd.DataFrame(contrib_rows_sawmill)
        operation_df = pd.DataFrame(contrib_rows_product)
        EoL_df = pd.DataFrame(contrib_rows_EoL)
        seq_df = pd.DataFrame(contrib_rows_bio)
        residue_co2_timeseries_df = pd.DataFrame(residue_co2_ts_rows)
        landfill_timeseries_df = pd.DataFrame(landfill_ts_rows)

        ############################################ EoL Scen 3 Data ############################################
        if emission_mode == 'low':
            # ---------- CLT EoL pyro energy (for product biochar) ----------
            EoL_3_name = region_ + '_' + pre_type_ + emission_mode
            EoL_3_all = pd.read_csv(os.path.join('..','Biochar_python','Results',EoL_3_name,EoL_3_name+'_all_results.csv'), index_col=0)
            EoL_3_summ = pd.read_csv(os.path.join('..','Biochar_python','Results',EoL_3_name,EoL_3_name+'_summary_results.csv'), index_col=0)
        
            EoL_3_summ['EoL_3_bio_CO2']   = EoL_3_summ.bio_co2  * EoL_df.Wood_mass_biochar_kg_m3
            EoL_3_summ['EoL_3_bound_CO2'] = EoL_3_summ.bound_CO2* EoL_df.Wood_mass_biochar_kg_m3
            EoL_3_released_bio = EoL_3_summ.EoL_3_bio_CO2/(EoL_3_summ.EoL_3_bio_CO2 + EoL_3_summ.EoL_3_bound_CO2)
            EoL_3_bound_bio    = EoL_3_summ.EoL_3_bound_CO2/(EoL_3_summ.EoL_3_bio_CO2 + EoL_3_summ.EoL_3_bound_CO2)
        
            EoL_df['EoL_3_bio_CO2']   = EoL_3_released_bio * seq_df.Sequestered_CO2
            EoL_df['EoL_3_bound_CO2'] = EoL_3_bound_bio    * seq_df.Sequestered_CO2
        
            # Physical Constraints
            len_req = len(EoL_3_all.loc[EoL_3_all['pyrolysis_energy']<1])
            if len_req > 0:
                EoL_3_all.loc[EoL_3_all['pyrolysis_energy']<1,'pyrolysis_energy'] = [random.uniform(1,6) for _ in range(len_req)]
            EoL_3_all['pyrolysis_energy_MJ_CLT'] = EoL_3_all.pyrolysis_energy * EoL_df.Wood_mass_biochar_kg_m3
            
            if CREDIT_PYRO_HEAT and PYRO_EXPORT_SHARE > 0.0:
                exported_MJ_CLT = PYRO_EXPORT_SHARE * EoL_3_all['pyrolysis_energy_MJ_CLT']
                EoL_df['EoL_3_avoided_fossil_CO2'] = exported_MJ_CLT * EF_data.NG_kg_CO2_MJ
            else:
                EoL_df['EoL_3_avoided_fossil_CO2'] = 0.0

        
            # ---------- Residue stream pyro energy ----------
            EoL_3_all_res = pd.read_csv(os.path.join('..','Biochar_python','Results',EoL_3_name,EoL_3_name+'_all_results.csv'), index_col=0)
            EoL_3_summ_res = pd.read_csv(os.path.join('..','Biochar_python','Results',EoL_3_name,EoL_3_name+'_summary_results.csv'), index_col=0)
        
            EoL_3_summ_res['EoL_3_bio_CO2']   = EoL_3_summ_res.bio_co2  * seq_df.residue_mass_kg
            EoL_3_summ_res['EoL_3_bound_CO2'] = EoL_3_summ_res.bound_CO2* seq_df.residue_mass_kg
            EoL_3_released_bio_res = EoL_3_summ_res.EoL_3_bio_CO2/(EoL_3_summ_res.EoL_3_bio_CO2 + EoL_3_summ_res.EoL_3_bound_CO2)
            EoL_3_bound_bio_res    = EoL_3_summ_res.EoL_3_bound_CO2/(EoL_3_summ_res.EoL_3_bio_CO2 + EoL_3_summ_res.EoL_3_bound_CO2)
        
            seq_df['clean_residue_bio_CO2']   = EoL_3_released_bio_res * seq_df.Sequestered_CO2_residue
            seq_df['clean_residue_bound_CO2'] = EoL_3_bound_bio_res    * seq_df.Sequestered_CO2_residue
        
            len_req = len(EoL_3_all_res.loc[EoL_3_all_res['pyrolysis_energy']<1])
            if len_req > 0:
                EoL_3_all_res.loc[EoL_3_all_res['pyrolysis_energy']<1,'pyrolysis_energy'] = [random.uniform(1,6) for _ in range(len_req)]
            EoL_3_all_res['pyrolysis_energy_MJ_RES'] = EoL_3_all_res.pyrolysis_energy * seq_df.residue_mass_kg
            
            if CREDIT_PYRO_HEAT and PYRO_EXPORT_SHARE > 0.0:
                exported_MJ_RES = PYRO_EXPORT_SHARE * EoL_3_all_res['pyrolysis_energy_MJ_RES']
                seq_df['clean_residue_avoided_fossil_CO2'] = exported_MJ_RES * EF_data.NG_kg_CO2_MJ
            else:
                seq_df['clean_residue_avoided_fossil_CO2'] = 0.0

        else:
            # No biochar pathway in "dirty" scenario
            EoL_df['EoL_3_bio_CO2'] = 0
            EoL_df['EoL_3_bound_CO2'] = 0
            EoL_df['EoL_3_avoided_fossil_CO2'] = 0
            seq_df['clean_residue_bio_CO2'] = 0
            seq_df['clean_residue_bound_CO2'] = 0
            seq_df['clean_residue_avoided_fossil_CO2'] = 0

        
        Y_MR = 200
        n = len(seq_df)
        years = np.arange(1, Y_MR + 1, dtype=float)
        
        biomass_EF = EF_data['biomass_kg_CO2_MJ'].astype(float).to_numpy()  # kgCO2/MJ (biogenic)
        NG_EF      = EF_data['NG_kg_CO2_MJ'].astype(float).to_numpy()       # kgCO2/MJ (fossil)
        elec_EF    = (EF_data['EF_kgCO2e_per_MJ_' + emission_mode + '_' + region_]
                      * EF_data['grid_factor_' + str(analysis_year)]).astype(float).to_numpy()
        
        # Initial mill-residue pool as CO2-equivalent 
        pool_total_CO2 = seq_df["Mill_residue_CO2"].astype(float).to_numpy()
        
        # Biomass used for kiln drying 
        bio_CO2_drying = sawmill_df["sawmill_biomass_CO2_biogenic"].astype(float).to_numpy()      # kg CO2 (bio) at t=0
        biomass_MJ_for_drying = bio_CO2_drying / np.maximum(biomass_EF, 1e-12)
        credit_CO2_drying = biomass_MJ_for_drying * NG_EF                                           # kg CO2 avoided (fossil)
        
        # Residue pool after drying
        pool_after_drying_CO2 = np.maximum(0.0, pool_total_CO2 - bio_CO2_drying)
        
        # Prepare result matrices
        emit_mat  = np.zeros((n, Y_MR), dtype=float)  # biogenic CO2 to air per year
        credit_mat = np.zeros((n, Y_MR), dtype=float) # avoided fossil CO2 per year
        
        # Record drying as year-1 emission & credit
        emit_mat[:, 0]  += bio_CO2_drying
        credit_mat[:, 0] += credit_CO2_drying

        # carbon fraction per sim (kg C / kg OD wood) 
        carbon_prop_vec = np.asarray(bio_data_col, dtype=float)
        if carbon_prop_vec.shape[0] != n:
            carbon_prop_vec = carbon_prop_vec[:n]
        
        # ---- Panelboard/fiberboard Emission parameters  ----
        PB_WOOD_KG_PER_M3 = 612.49
        PB_RESIN_KG_PER_M3 = (39.72 + 3.48 + 2.31 + 0.24)   # UF + MUF + pMDI + PF
        PB_A3_GWP_FOSSIL_KG_PER_M3 = 172.41
        
        MDF_WOOD_KG_PER_M3 = 654.37
        MDF_RESIN_KG_PER_M3 = (45.77 + 12.63 + 4.60)        # UF + MUF + pMDI
        MDF_A3_GWP_FOSSIL_KG_PER_M3 = 296.82
        
        PB_SHARE  = 0.6
        MDF_SHARE = 0.4
        
        # Resin production EF (kg CO2e / kg resin)
        RESIN_GWP_FOSSIL_KG_PER_KG = 1.806
        
        # ---- Pulp/paper A3 EF 
        INCLUDE_PULP_A3 = True
        PAPER_A3_GWP_FOSSIL_KG_PER_KG = 0.442
        
        # ---- Pellets A1–A3 fossil intensity (kg CO2e / MJ pellet produced) 
        INCLUDE_PELLET_A3 = True
        PELLET_A1A3_FOSSIL_KGCO2_PER_MJ = (114.0 + 65.0) / 17000.0
        # =================================================================================================================

        
        if emission_mode == 'low':
            # CLEAN: first use residue to displace process NG, then electricity (all credited at analysis_year)
            NG_CO2_demand   = (sawmill_df['sawmill_ng_CO2'].astype(float) + operation_df['op_ng_CO2'].astype(float)).to_numpy()
            NG_MJ_demand    = NG_CO2_demand / np.maximum(NG_EF, 1e-12)
        
            Elec_CO2_demand = (sawmill_df['sawmill_electricity_CO2'].astype(float) + operation_df['op_electricity_CO2'].astype(float)).to_numpy()
            Elec_MJ_demand  = Elec_CO2_demand / np.maximum(elec_EF, 1e-12)
        
            avail_MJ = pool_after_drying_CO2 / np.maximum(biomass_EF, 1e-12)
            use_NG_MJ   = np.minimum(avail_MJ, NG_MJ_demand)
            rem_MJ      = np.maximum(0.0, avail_MJ - use_NG_MJ)
            use_Elec_MJ = np.minimum(rem_MJ, Elec_MJ_demand)
        
            # Emissions from burning residue for energy (year 1)
            CO2_energy_now = (use_NG_MJ + 0) * biomass_EF
            emit_mat[:, 0]  += CO2_energy_now
        
            # Avoided fossil/grid credits (year 1)
            credit_mat[:, 0] += (use_NG_MJ * NG_EF) + 0.0
        
            # Leftover residue goes into panels → exponential release over Y_MR years
            CO2_to_panels = np.maximum(0.0, pool_after_drying_CO2 - CO2_energy_now)

            # add PB/MDF + resin fossil A3 to YEAR 1 ONLY 
            C_to_panels = CO2_to_panels * (12.0 / 44.0)
            wood_od_to_panels = C_to_panels / np.maximum(carbon_prop_vec, 1e-12)
            
            # Split wood into PB vs MDF (by wood mass)
            wood_od_pb  = PB_SHARE  * wood_od_to_panels
            wood_od_mdf = MDF_SHARE * wood_od_to_panels
            
            # Convert to product volume using EPD wood content per m3
            V_pb  = wood_od_pb  / PB_WOOD_KG_PER_M3
            V_mdf = wood_od_mdf / MDF_WOOD_KG_PER_M3

            # Green Panel Factors to normalize production emission reductions
            PANEL_A3_MULT      = 0.65   
            RESIN_EF_MULT      = 0.78   
            RESIN_LOAD_MULT    = 0.85   
            
            # A3 fossil emissions (manufacturing) in kg CO2e
            panel_A3_foss = (V_pb * PB_A3_GWP_FOSSIL_KG_PER_M3*PANEL_A3_MULT) + (V_mdf * MDF_A3_GWP_FOSSIL_KG_PER_M3*PANEL_A3_MULT)
            
            # Resin production fossil emissions in kg CO2e
            resin_kg = (V_pb * PB_RESIN_KG_PER_M3*RESIN_LOAD_MULT) + (V_mdf * MDF_RESIN_KG_PER_M3*RESIN_LOAD_MULT)
            panel_resin_foss = resin_kg * RESIN_GWP_FOSSIL_KG_PER_KG *RESIN_EF_MULT
            
            # BOOK EVERYTHING INTO YEAR 1 ONLY (emit_year_1)
            emit_mat[:, 0] += (panel_A3_foss + panel_resin_foss)
            # ========================================================================

            hl_panels = np.random.triangular(10.0, 20.0, 30.0, size=n)     # years
            k_panels  = np.log(2.0) / np.maximum(hl_panels, 1e-12)
            panel_frac = (np.exp(-k_panels[:, None]*(years - 1.0)) - np.exp(-k_panels[:, None]*years))
            emit_mat += CO2_to_panels[:, None] * panel_frac
        
        else:
            pool = pool_after_drying_CO2
        
            # Split market shares 
            if region_ == "PNW":
                pulp   = np.random.uniform(0.30, 0.65, size=n)
                panels = np.random.uniform(0.10, 0.25, size=n)
                energy = np.random.uniform(0.15, 0.40, size=n)
                pellets= np.random.uniform(0.00, 0.15, size=n)
                bedding= np.random.uniform(0.05, 0.20, size=n)
            else:  # SE
                pulp   = np.random.uniform(0.40, 0.70, size=n)
                panels = np.random.uniform(0.05, 0.20, size=n)
                energy = np.random.uniform(0.10, 0.35, size=n)
                pellets= np.random.uniform(0.00, 0.20, size=n)
                bedding= np.random.uniform(0.05, 0.15, size=n)
        
            landfill = np.zeros(n)
            S = pulp + panels + energy + pellets + bedding + landfill
            pulp, panels, energy, pellets, bedding = pulp/S, panels/S, energy/S, pellets/S, bedding/S
        
            M_pulp   = pool * pulp
            M_panels = pool * panels
            M_energy = pool * energy
            M_pellet = pool * pellets
            M_bed    = pool * bedding

            # ================= add PB/MDF (+ resin) + pellet + pulp fossil A3 to YEAR 1 ONLY =================
            mr_prod_foss_y1 = np.zeros(n, dtype=float)
            
            #  Panels (PB/MDF) fossil A3 + resin production 
            C_panels = M_panels * (12.0 / 44.0)
            wood_od_panels = C_panels / np.maximum(carbon_prop_vec, 1e-12)
            
            wood_od_pb  = PB_SHARE  * wood_od_panels
            wood_od_mdf = MDF_SHARE * wood_od_panels
            
            V_pb  = wood_od_pb  / PB_WOOD_KG_PER_M3
            V_mdf = wood_od_mdf / MDF_WOOD_KG_PER_M3
            # Green Panel Factors
            PANEL_A3_MULT      = 0.65   
            RESIN_EF_MULT      = 0.78   
            RESIN_LOAD_MULT    = 0.85   
            
            panel_A3_foss = (V_pb * PB_A3_GWP_FOSSIL_KG_PER_M3*PANEL_A3_MULT) + (V_mdf * MDF_A3_GWP_FOSSIL_KG_PER_M3*PANEL_A3_MULT)
            resin_kg = (V_pb * PB_RESIN_KG_PER_M3*RESIN_LOAD_MULT) + (V_mdf * MDF_RESIN_KG_PER_M3*RESIN_LOAD_MULT)
            panel_resin_foss = resin_kg * RESIN_GWP_FOSSIL_KG_PER_KG * RESIN_EF_MULT
            
            mr_prod_foss_y1 += (panel_A3_foss + panel_resin_foss)
            
            #   Pulp/paper A3 fossil emissions  
            if INCLUDE_PULP_A3:
                C_pulp = M_pulp * (12.0 / 44.0)
                wood_od_pulp = C_pulp / np.maximum(carbon_prop_vec, 1e-12)
                mr_prod_foss_y1 += wood_od_pulp * PAPER_A3_GWP_FOSSIL_KG_PER_KG
            
            #  Pellets: fossil A1–A3 per MJ pellet produced 
            if INCLUDE_PELLET_A3:
                MJ_pellet = M_pellet / np.maximum(biomass_EF, 1e-12)  # MJ pellet (thermal basis consistent w/ your model)
                mr_prod_foss_y1 += MJ_pellet * PELLET_A1A3_FOSSIL_KGCO2_PER_MJ
            
            emit_mat[:, 0] += mr_prod_foss_y1
            # =================================================================================================================

        
            # Pulp (short HL), Panels (longer HL), Bedding (fast+slow), Energy/Pellets (instant)
            hl_pulp = np.random.uniform(0.5, 2.0, size=n)
            pulp_frac = (np.exp(-np.log(2)/hl_pulp[:,None]*(years-1.0)) - np.exp(-np.log(2)/hl_pulp[:,None]*years))
        
            hl_pan = np.random.triangular(10.0, 20.0, 30.0, size=n)
            pan_frac = (np.exp(-np.log(2)/hl_pan[:,None]*(years-1.0)) - np.exp(-np.log(2)/hl_pan[:,None]*years))
        
            share_fast = np.random.uniform(0.80, 0.95, size=n)
            hl_fast = np.random.uniform(0.5, 2.0, size=n)
            hl_slow = np.random.uniform(10.0, 40.0, size=n)
            bed_fast = (np.exp(-np.log(2)/hl_fast[:,None]*(years-1.0)) - np.exp(-np.log(2)/hl_fast[:,None]*years))
            bed_slow = (np.exp(-np.log(2)/hl_slow[:,None]*(years-1.0)) - np.exp(-np.log(2)/hl_slow[:,None]*years))
            bed_frac = share_fast[:,None]*bed_fast + (1.0 - share_fast)[:,None]*bed_slow
        
            # Assemble per-year emissions (no credits in dirty)
            emit_mat += M_pulp[:,None]*pulp_frac + M_panels[:,None]*pan_frac + M_bed[:,None]*bed_frac
            emit_mat[:, 0] += (M_energy + M_pellet)  # instant release in year 1

            # ================= ADD: MR bioenergy + pellet credits (YEAR 1 ONLY) =================
            den = np.maximum(biomass_EF, 1e-12)
            
            # Convert the CO2-allocated "energy" and "pellet" fates back to MJ basis
            MJ_energy = M_energy / den   # thermal MJ from residue energy fate
            MJ_pellet = M_pellet / den   # thermal MJ embodied in pellets fate
            
            # Electricity credit from MR energy (displacing grid electricity)
            MR_elec_eff = 0.20 
            credit_energy_grid = MJ_energy * MR_elec_eff * elec_EF   # kgCO2 avoided
            
            # Heat credit from pellets (displacing natural gas heat)
            credit_pellet_NG = MJ_pellet * NG_EF                     # kgCO2 avoided
            
            # Book credits into YEAR 1 only
            credit_mat[:, 0] += (credit_energy_grid + credit_pellet_NG)
            # =================================================================================================
        
        # ---- Write tidy time-series DF 
        mill_residue_co2_timeseries_df = pd.DataFrame({"simulation": np.arange(n)})
        for j, y in enumerate(range(1, Y_MR + 1)):
            mill_residue_co2_timeseries_df[f"emit_year_{y}"]  = emit_mat[:, j]
            mill_residue_co2_timeseries_df[f"cred_year_{y}"]  = credit_mat[:, j]

        ############################################ Generate Construction and Demolition Data ############################################
        # Construction
        if emission_mode == 'low':
            const_lst = [7,(7+19)/2]
        elif emission_mode == 'high':
            const_lst = [(7+19)/2,19]
        const_dem_df = pd.DataFrame()
        data_now = np.array([const_lst[0],const_lst[1]])
        dist_now = stats.uniform
        HD_param = dist_now.fit(data_now)
        arg = HD_param[:-2]
        loc = HD_param[-2]
        scale = HD_param[-1]
        random_samples = dist_now.rvs(loc=loc, scale=scale, *arg, size=sim_nums)
        const_dem_df['construction_CO2'] = random_samples*EF_data['grid_factor_'+str(analysis_year)]
    
        ########### Add the Eol Scenario 4 which is Reuse through construction activties 
        EoL_df['EoL_4_fossil_CO2'] = const_dem_df['construction_CO2']
        
        # Demolition
        if emission_mode == 'low':
            const_lst = [7,(7+13)/2]
        elif emission_mode == 'high':
            const_lst = [(7+13)/2,13]
        data_now = np.array([const_lst[0],const_lst[1]])
        dist_now = stats.uniform
        HD_param = dist_now.fit(data_now)
        arg = HD_param[:-2]
        loc = HD_param[-2]
        scale = HD_param[-1]
        random_samples = dist_now.rvs(loc=loc, scale=scale, *arg, size=sim_nums)
        const_dem_df['deconstruction_CO2'] = random_samples*EF_data['grid_factor_'+str(analysis_year)]
        
        ############################################ Export Results ############################################
        folder_name = region_+'_'+pre_type_+prod_type_+'_'+emission_mode
        new_folder = os.path.join('MCS_Results', folder_name)
        
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        
        harvest_df.to_csv(os.path.join(new_folder,'harvest_df_results_'+str(analysis_year)+'.csv'))
        sawmill_df.to_csv(os.path.join(new_folder,'sawmill_df_results_'+str(analysis_year)+'.csv'))
        operation_df.to_csv(os.path.join(new_folder,'operation_df_results_'+str(analysis_year)+'.csv'))
        EoL_df.to_csv(os.path.join(new_folder,'EoL_df_results_'+str(analysis_year)+'.csv'))
        seq_df.to_csv(os.path.join(new_folder,'seq_df_results_'+str(analysis_year)+'.csv'))
        const_dem_df.to_csv(os.path.join(new_folder,'const_dem_df_results_'+str(analysis_year)+'.csv'))
        
        # Save the time series
        mill_residue_co2_timeseries_df.to_csv(
        os.path.join(new_folder, f'mill_residue_co2_timeseries_df_results_{analysis_year}.csv'),
        index=False)
        landfill_timeseries_df.to_csv(
            os.path.join(new_folder, f'landfill_timeseries_{analysis_year}.csv'),
            index=False
        )
        
        if not residue_co2_timeseries_df.empty:
            residue_co2_timeseries_df.to_csv(
                os.path.join(new_folder, f'slash_residue_co2_timeseries_{analysis_year}.csv'),
                index=False
            )

        print(folder_name + '_' + str(analysis_year) + ' Success!')
        
############################################ Final Export Results ############################################
for analysis_year in analysis_years:
    for emission_mode in ("low", "high"):
        scenario_name = f"{region_}_{pre_type_}{prod_type_}_{emission_mode}"
        new_folder = os.path.join('MCS_Results', scenario_name)
        print(f"{scenario_name}{analysis_year}")

        # -------- core per-scenario/year results 
        seq_df       = pd.read_csv(os.path.join(new_folder, f'seq_df_results_{analysis_year}.csv'), index_col=0)
        harvest_df   = pd.read_csv(os.path.join(new_folder, f'harvest_df_results_{analysis_year}.csv'), index_col=0)
        sawmill_df   = pd.read_csv(os.path.join(new_folder, f'sawmill_df_results_{analysis_year}.csv'), index_col=0)
        operation_df = pd.read_csv(os.path.join(new_folder, f'operation_df_results_{analysis_year}.csv'), index_col=0)
        const_dem_df = pd.read_csv(os.path.join(new_folder, f'const_dem_df_results_{analysis_year}.csv'), index_col=0)
        EoL_df       = pd.read_csv(os.path.join(new_folder, f'EoL_df_results_{analysis_year}.csv'), index_col=0)

        
        mill_ts_path     = os.path.join(new_folder, f'mill_residue_co2_timeseries_df_results_{analysis_year}.csv')
        landfill_ts_path = os.path.join(new_folder, f'landfill_timeseries_{analysis_year}.csv')
        slash_ts_path    = os.path.join(new_folder, f'slash_residue_co2_timeseries_{analysis_year}.csv')

       
        mill_ts     = _read_optional_csv(mill_ts_path)
        landfill_ts = _read_optional_csv(landfill_ts_path)
        slash_ts    = _read_optional_csv(slash_ts_path)


        out_df = make_out_df(
            seq_df, harvest_df, sawmill_df, operation_df, const_dem_df, EoL_df, emission_mode,
            landfill_ts=landfill_ts,  
            mill_ts=mill_ts,          
            slash_ts=slash_ts,        
            use_years=50, eol_years=27)

        out_path = os.path.join(new_folder, f'TAWP_input_{emission_mode}_{analysis_year}.csv')
        out_df.to_csv(out_path)