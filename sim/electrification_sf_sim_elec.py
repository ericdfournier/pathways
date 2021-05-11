#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 08:02:11 2020

@author: edf
"""

#%% Package Imports

import os
os.chdir('/Users/edf/repos/cec_ng/pathways/')    

import numpy as np
from pkg.simulation import Simulation
import pkg.utils as utils

#%% Read in Building Model Load Profiles
    
# Load Profile Data
directory = '/Users/edf/repos/cec_ng/pathways/prototypes/sf_electrification/BEopt/'
field = 'Site Energy|Total (E)'
unit = 'kwh'
p, nm = utils.ReadBEopt(directory, field, unit)
root = '/Users/edf/repos/cec_ng/pathways/fig/sf_electrification/'

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

#%% Specify Simulation Parameters Simulation

n = np.floor(total_single_family_parcels * ((1.0 - average_vacancy_percentage))).astype(int)
rw = 3
t_start = '1/1/2020'
t_stop = '1/1/2100'
t_end = '1/1/2046'
w1 = [0.28, 0.26, 0.06, 0.17, 0.23] # Reflects the relative distribution of the baseline prototypes (must sum to 1)
low = -0.75
med = -0.55
high = -0.25
custom_init = [100, 10, 1]

#%% Baseline

# Baseline-Growth

# Parameterize Simulation Run
tl = 'Electrification_Baseline'
w0 = [0.0, 0.0, 0.0]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = -0.9

# Execute Simulation
baseline = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
baseline.Run(custom_init = custom_init)

#%% IAQ Dominant Electrification 

# Low-Growth

# Parameterize Simulation
tl = 'Electrification_Low-Growth_IAQ-Minor-Dominant'
w0 = [1.0, -0.5, -0.5]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = low

# Execute Simulation
iaq_low = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
iaq_low.Run(baseline = baseline, custom_init = custom_init)

#  Medium-Growth

# Parameterize Simulation Run
tl = 'Electrification_Medium-Growth_IAQ-Minor-Dominant'
w0 = [1.0, -0.5, -0.5]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = med

# Execute Simulation
iaq_med = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
iaq_med.Run(baseline = baseline, custom_init = custom_init)

# High Growth

# Parameterize Simulation Run
tl = 'Electrification_High-Growth_IAQ-Minor-Dominant'
w0 = [1.0, -0.5, -0.5]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = high

# Execute Simulation
iaq_high = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
iaq_high.Run(baseline = baseline, custom_init = custom_init)

#%% Cost Minimizing Electrification

# Low-Growth

# Parameterize Simulation Run
tl = 'Electrification_Low-Growth_IAQ-Moderate-Dominant'
w0 = [-0.5, 1.0, -0.5]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = low

# Execute Simulation
cmin_low = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
cmin_low.Run(baseline = baseline, custom_init = custom_init)

# Medium-Growth

# Parameterize Simulation Run
tl = 'Electrification_Medium-Growth_IAQ-Moderate-Dominant'
w0 = [-0.5, 1.0, -0.5]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = med

# Execute Simulation
cmin_med = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
cmin_med.Run(baseline = baseline, custom_init = custom_init)

# High-Growth

# Parameterize Simulation Run
tl = 'Electrification_High-Growth_IAQ-Moderate-Dominant'
w0 = [-0.5, 1.0, -0.5]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = high

# Execute Simulation
cmin_high = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
cmin_high.Run(baseline = baseline, custom_init = custom_init)

#%% Full Home Electrification Dominant

# Low-Growth

# Parameterize Simulation Run
tl = 'Electrification_Low-Growth_Full-Home-Dominant'
w0 = [-0.5, -0.5, 1.0]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = low

# Execute Simulation
fhe_low = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
fhe_low.Run(baseline = baseline, custom_init = custom_init)

# Medium-Growth

# Parameterize Simulation Run
tl = 'Electrification_Medium-Growth_Full-Home-Dominant'
w0 = [-0.5, -0.5, 1.0]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = med

# Execute Simulation
fhe_med = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
fhe_med.Run(baseline = baseline, custom_init = custom_init)

# High-Growth

# Parameterize Simulation Run
tl = 'Electrification_High-Growth_Full-Home-Dominant'
w0 = [-0.5, -0.5, 1.0]  # Reflects the relative likelihood of the non-baseline scenarios occurring (must sume to 0)
k = high

# Execute Simulation
fhe_high = Simulation(tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end)
fhe_high.Run(baseline = baseline, custom_init = custom_init)

#%% Write Simulation Outputs to File

data_root = '/users/edf/repos/cec_ng/pathways/data/sf_electrification/elec/'

p.to_pickle(data_root + 'profiles.pkl')
nm.to_pickle(data_root + 'names.pkl')

baseline.Pickle(data_root + 'baseline.pkl')

iaq_low.Pickle(data_root + 'iaq_low.pkl')
iaq_med.Pickle(data_root + 'iaq_med.pkl')
iaq_high.Pickle(data_root + 'iaq_high.pkl')

cmin_low.Pickle(data_root + 'cmin_low.pkl')
cmin_med.Pickle(data_root + 'cmin_med.pkl')
cmin_high.Pickle(data_root + 'cmin_high.pkl')

fhe_low.Pickle(data_root + 'fhe_low.pkl')
fhe_med.Pickle(data_root + 'fhe_med.pkl')
fhe_high.Pickle(data_root + 'fhe_high.pkl')
