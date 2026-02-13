############### TAWP100 D-CUBE Analysis
#%% Import Data and Functions from Other Modules
from os import chdir, getcwd
wd=getcwd()
chdir(wd)
from tqdm import tqdm
import numpy as np
from scipy import stats
import pandas as pd
import os
from DCUBE_fnxs_final_NEW import *
#%%## Designate Scenarios to Run:
emission_mode_lst = ['low','high']
region_   = "PNW" # options: {'PNW', 'SE'}
pre_wood_ = "lumber_" # options: {'lumber_', 'LVL_'}
pre_type_ = "softwood_" # options: {'softwood_','hardwood_'}
prod_type_ = 'Glulam' # options: {'CLT','Glulam', 'LVL'}
analysis_years = [2020,2030,2040,2050,
                  2060,2070,2080,2090,2100,
                  2110,2120]
#%%############################### Delcaration of Model Paramaters:
for analysis_year in (analysis_years):
    for i in range(len(emission_mode_lst)):

        emission_mode = emission_mode_lst[i] #options: {'low', 'high'}
        
        scenario_name = region_+'_'+pre_type_+prod_type_+'_'+emission_mode
        new_folder = os.path.join('..','MCS_Results', scenario_name)
        print(scenario_name,'_',str(analysis_year))
        out_df = pd.read_csv(os.path.join(new_folder,'TAWP_input_'+emission_mode+'_'+str(analysis_year)+'.csv'),index_col = 0)
    
        # Get the final df 
        # Run All rows, all scenarios (1,2,3), wide format
        eol_scenarios_choose = [1,2]
        res_all = run_many_realizations_multi_eol(out_df, rows=10000, eol_scenarios=eol_scenarios_choose, return_style="long")
        
        # Output Out Files 
        out_file_name = os.path.join('TAWP_results',scenario_name+'_'+str(analysis_year)+".csv")
        res_all.to_csv(out_file_name)
        in_file_name = os.path.join('TAWP_results',scenario_name+'_'+str(analysis_year)+"_inputs.csv")
        out_df.to_csv(in_file_name)

