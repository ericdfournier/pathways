#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:19:50 2020

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
d1 = '/Users/edf/repos/cec_ng/pathways/prototypes/sf_ev/BEopt/'
field = 'Site Energy|Total (E)'
p1, nm1 = utils.ReadBEopt(d1, field)

# Load Homer Profile Data
d2 = '/Users/edf/repos/cec_ng/pathways/prototypes/sf_ev/EV/'
p2, nm2 = utils.ReadEV(d2)

# Merge Profiles
p = pd.merge(p1, p2, left_index = True, right_index = True)
nm = pd.concat((nm1, nm2), axis = 0)

# Specify Output Figure Root Filepath
root = '/Users/edf/repos/cec_ng/pathways/fig/sf_ev/'

#%% Calibration Figures

# Building Stock
total_single_family_sqft = 14376560.0

# Occupancy
average_vacancy_percentage = 0.039374983701637523

# Total SF Parcels
total_single_family_parcels = 10649

# Models
single_family_prototype_sqft = 1368.0

# Energy
total_res_load_kwh = 113016309.0

# SF Res Square Footage Fraction
sf_res_sqft_frac = 0.6783

# Energy Intensity Adjustment Factor
eui_adj_factor = 0.80

# Total SF Res Energy Estimate
sf_total_res_load_kwh = total_res_load_kwh * eui_adj_factor * sf_res_sqft_frac

#%% Specify Simulation Parameters Simulation

# n = np.floor((total_single_family_sqft / single_family_prototype_sqft) * ((100.0 - average_vacancy_percentage )/ 100.0)).astype(int)
n = np.floor(total_single_family_parcels * ((1.0 - average_vacancy_percentage))).astype(int)
rw = 3
t_start = '1/1/2020'
t_stop = '1/1/2100'
t_end = '1/1/2046'
w1 = [0.28, 0.26, 0.06, 0.17, 0.23] # Reflects the relative distribution of the baseline prototypes (must sum to 1)
low = -0.75
med = -0.55
high = -0.25
custom_init = [400] # Allocating 400 of 500 to the single-family context

#%% Baseline

# Baseline-Growth

# Parameterize Simulation Run
tl = 'EV_Baseline'
w0 = [0.0]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sum to 0)
k = -0.7

# Execute Simulation
baseline = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
baseline.Run(custom_init = custom_init)

#%% EV Dominant Growth

# Low-Growth

# Parameterize Simulation
tl = 'EV_Low-Growth'
w0 = [0.0]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sum to 0)
k = low

# Execute Simulation
ev_low = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
ev_low.Run(baseline = baseline, custom_init = custom_init)

#  Medium-Growth

# Parameterize Simulation Run
tl = 'EV_Medium-Growth'
w0 = [0.0]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sum to 0)
k = med

# Execute Simulation
ev_med = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
ev_med.Run(baseline = baseline, custom_init = custom_init)

# High Growth

# Parameterize Simulation Run
tl = 'EV_High-Growth'
w0 = [0.0]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sum to 0)
k = high

# Execute Simulation
ev_high = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
ev_high.Run(baseline = baseline, custom_init = custom_init)

#%% Write Simulation Outputs to File

data_root = '/users/edf/repos/cec_ng/pathways/data/sf_ev/'

p.to_pickle(data_root + 'profiles.pkl')
nm.to_pickle(data_root + 'names.pkl')

baseline.Pickle(data_root + 'baseline.pkl')

ev_low.Pickle(data_root + 'ev_low.pkl')
ev_med.Pickle(data_root + 'ev_med.pkl')
ev_high.Pickle(data_root + 'ev_high.pkl')
