#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 08:02:11 2020

@author: edf
"""

#%% Package Imports

import os
import pandas as pd
os.chdir('/Users/edf/repos/cec_ng/pathways/')    

import numpy as np
from pkg.simulation import Simulation
import pkg.utils as utils

#%% Read in Building Model Load Profiles
    
# Load BEOpt Profile Data
d1 = '/Users/edf/repos/cec_ng/pathways/prototypes/sf_der/BEopt/'
field = 'Site Energy|Total (E)'
p1, nm1 = utils.ReadBEopt(d1, field)

# Load Homer Profile Data
d2 = '/Users/edf/repos/cec_ng/pathways/prototypes/sf_der/REopt/'
p2, nm2 = utils.ReadREopt(d2)

# Merge Profiles
p = pd.merge(p1, p2, left_index = True, right_index = True)
nm = pd.concat((nm1, nm2), axis = 0)

# Specify Output Figure Root Filepath
root = '/Users/edf/repos/cec_ng/pathways/fig/sf_der/'

#%% Calibration Figures

# Building Stock
total_single_family_sqft = 14376560.0

# Occupancy
average_vacancy_percentage = 0.039374983701637523

# Total SF Parcels
total_single_family_parcels = 10649

# Models
single_family_prototype_sqft = 1254.0

# Energy
total_res_load_kwh = 113016309.0

# SF Res Square Footage Fraction
sf_res_sqft_frac = 0.6783

# Energy Intensity Adjustment Factor
eui_adj_factor = 0.80

# Total SF Res Energy Estimate
sf_total_res_load_kwh = total_res_load_kwh * eui_adj_factor * sf_res_sqft_frac

#%% Specify Simulation Parameters Simulation

#n = np.floor((total_single_family_sqft / single_family_prototype_sqft) * ((100.0 - average_vacancy_percentage )/ 100.0)).astype(int)
n = np.floor(total_single_family_parcels * ((1.0 - average_vacancy_percentage))).astype(int)
rw = 3
t_start = '1/1/2020'
t_stop = '1/1/2100'
t_end = '1/1/2046'
w1 = [0.28, 0.26, 0.06, 0.17, 0.23] # Reflects the relative distribution of the baseline prototypes (must sum to 1)
low = -0.75
med = -0.55
high = -0.25

# Initial Non-Baseline Group Counts
custom_init = [31, 31, 31]

#%% Baseline

# Baseline-Growth

# Parameterize Simulation Run
tl = 'DER_Baseline'
w0 = [0.0, 0.0, 0.0]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sum to 0)
k = -0.95

# Execute Simulation
baseline = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
baseline.Run(custom_init = custom_init)

#%% Check Calibration

error_pct = 100.0 * (1.0 - (sf_total_res_load_kwh / baseline.composite_load_profile.resample('Y').agg('sum').iloc[0]))

#%% BESS Only Dominant Growth

# Low-Growth

# Parameterize Simulation Run
tl = 'DER_Low-Growth_BESS-Only-Dominant'
w0 = [-0.5, 1.0, -0.5]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = low

# Execute Simulation
bess_low = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
bess_low.Run(baseline = baseline, custom_init = custom_init)

# Medium-Growth

# Parameterize Simulation Run
tl = 'DER_Medium-Growth_BESS-Only-Dominant'
w0 = [-0.5, 1.0, -0.5]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = med

# Execute Simulation
bess_med = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
bess_med.Run(baseline = baseline, custom_init = custom_init)

# High-Growth

# Parameterize Simulation Run
tl = 'DER_High-Growth_BESS-Only-Dominant'
w0 = [-0.5, 1.0, -0.5]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = high

# Execute Simulation
bess_high = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
bess_high.Run(baseline = baseline, custom_init = custom_init)

#%% PV-Only Dominant Growth

# Low-Growth

# Parameterize Simulation
tl = 'DER_Low-Growth_PV-Only-Dominant'
w0 = [1.0, -0.5, -0.5]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = low

# Execute Simulation
pv_low = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
pv_low.Run(baseline = baseline, custom_init = custom_init)

#  Medium-Growth

# Parameterize Simulation Run
tl = 'DER_Medium-Growth_PV-Only-Dominant'
w0 = [1.0, -0.5, -0.5]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = med

# Execute Simulation
pv_med = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
pv_med.Run(baseline = baseline, custom_init = custom_init)

# High Growth

# Parameterize Simulation Run
tl = 'DER_High-Growth_PV-Only-Dominant'
w0 = [1.0, -0.5, -0.5]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = high

# Execute Simulation
pv_high = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
pv_high.Run(baseline = baseline, custom_init = custom_init)

#%% PV+BESS Dominant Growth

# Low-Growth

# Parameterize Simulation Run
tl = 'DER_Low-Growth_PV+BESS-Dominant'
w0 = [-0.5, -0.5, 1.0]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = low

# Execute Simulation
pvbess_low = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
pvbess_low.Run(baseline = baseline, custom_init = custom_init)

# Medium-Growth

# Parameterize Simulation Run
tl = 'DER_Medium-Growth_PV+BESS-Dominant'
w0 = [-0.5, -0.5, 1.0]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = med

# Execute Simulation
pvbess_med = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
pvbess_med.Run(baseline = baseline, custom_init = custom_init)

# High-Growth

# Parameterize Simulation Run
tl = 'DER_High-Growth_PV+BESS-Dominant'
w0 = [-0.5, -0.5, 1.0]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = high

# Execute Simulation
pvbess_high = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
pvbess_high.Run(baseline = baseline, custom_init = custom_init)

#%% Write Simulation Outputs to File

data_root = '/users/edf/repos/cec_ng/pathways/data/sf_der/'

p.to_pickle(data_root + 'profiles.pkl')
nm.to_pickle(data_root + 'names.pkl')

baseline.Pickle(data_root + 'baseline.pkl')

bess_low.Pickle(data_root + 'bess_low.pkl')
bess_med.Pickle(data_root + 'bess_med.pkl')
bess_high.Pickle(data_root + 'bess_high.pkl')

pv_low.Pickle(data_root + 'pv_low.pkl')
pv_med.Pickle(data_root + 'pv_med.pkl')
pv_high.Pickle(data_root + 'pv_high.pkl')

pvbess_low.Pickle(data_root + 'pvbess_low.pkl')
pvbess_med.Pickle(data_root + 'pvbess_med.pkl')
pvbess_high.Pickle(data_root + 'pvbess_high.pkl')
