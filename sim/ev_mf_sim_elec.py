#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 08:51:48 2020

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
d1 = '/Users/edf/repos/cec_ng/pathways/prototypes/mf_ev/BEopt/'
field = 'Site Energy|Total (E)'
p1, nm1 = utils.ReadBEopt(d1, field)

# Load Homer Profile Data
d2 = '/Users/edf/repos/cec_ng/pathways/prototypes/mf_ev/EV/'
p2, nm2 = utils.ReadEV(d2)

# Merge Profiles
p = pd.merge(p1, p2, left_index = True, right_index = True)
nm = pd.concat((nm1, nm2), axis = 0)

# Specify Output Figure Root Filepath
root = '/Users/edf/repos/cec_ng/pathways/fig/mf_ev/'

#%% Calibration Figures

# Building Stock
total_multi_family_sqft= 6821470.0

# Occupancy
average_vacancy_percentage = 3.9374983701637523

# Models
multi_family_prototype_sqft = 700.0

# Energy
total_res_load_kwh = 113016309.0

# MF Res Square Footage Fraction
mf_res_sqft_frac = 0.3217

# Energy Intensity Adjustment Factor
eui_adj_factor = 0.90

# Total SF Res Energy Estimate
mf_total_res_load_kwh = total_res_load_kwh * eui_adj_factor * mf_res_sqft_frac

#%% Specify Simulation Parameters Simulation

n = np.floor((total_multi_family_sqft / multi_family_prototype_sqft)).astype(int)
rw = 3
t_start = '1/1/2020'
t_stop = '1/1/2100'
t_end = '1/1/2046'
w1 = [(1.0/3.0), (1.0/3.0), (1.0/3.0)] # Reflects the relative distribution of the baseline prototypes (must sum to 1)
low = -0.75
med = -0.55
high = -0.25
custom_init = [100] # Allocating 100 of 500 to the multi-family context

#%% Baseline

# Baseline-Growth

# Parameterize Simulation Run
tl = 'EV_Baseline'
w0 = [0.0]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sum to 0)
k = -0.9

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

data_root = '/users/edf/repos/cec_ng/pathways/data/mf_ev/'

p.to_pickle(data_root + 'profiles.pkl')
nm.to_pickle(data_root + 'names.pkl')

baseline.Pickle(data_root + 'baseline.pkl')

ev_low.Pickle(data_root + 'ev_low.pkl')
ev_med.Pickle(data_root + 'ev_med.pkl')
ev_high.Pickle(data_root + 'ev_high.pkl')
