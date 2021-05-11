#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 07:20:49 2020

@author: edf
"""

#%% Package Imports

import pandas as pd
import numpy as np
import pickle
import os
os.chdir('/Users/edf/repos/cec_ng/pathways/')    

import pkg.plot as plot

#%% Read Simulation Outputs from File

data_root = '/users/edf/repos/cec_ng/pathways/data/mf_electrification/gas/'

profiles = pd.read_pickle(data_root + 'profiles.pkl')
names = pd.read_pickle(data_root + 'names.pkl')

baseline = pickle.load(open(data_root + 'baseline.pkl', 'rb'))

cmin_low = pickle.load(open(data_root + 'cmin_low.pkl', 'rb'))
cmin_med = pickle.load(open(data_root + 'cmin_med.pkl', 'rb'))
cmin_high = pickle.load(open(data_root + 'cmin_high.pkl', 'rb'))

iaq_low = pickle.load(open(data_root + 'iaq_low.pkl', 'rb'))
iaq_med = pickle.load(open(data_root + 'iaq_med.pkl', 'rb'))
iaq_high = pickle.load(open(data_root + 'iaq_high.pkl', 'rb'))

fhe_low = pickle.load(open(data_root + 'fhe_low.pkl', 'rb'))
fhe_med = pickle.load(open(data_root + 'fhe_med.pkl', 'rb'))
fhe_high = pickle.load(open(data_root + 'fhe_high.pkl', 'rb'))

#%% Set Plotting Parameters

root = root = '/Users/edf/repos/cec_ng/pathways/fig/mf_electrification/gas/'

vlim_month = (-np.round(fhe_high.monthly_baseline_change.max().max() / 1000000, 2),
              np.round(fhe_high.monthly_baseline_change.max().max() / 1000000, 2))

vlim_hour = (-np.round(fhe_high.mean_hourly_baseline_change.max().max() / 1000, 0),
             np.round(fhe_high.mean_hourly_baseline_change.max().max() / 1000, 0))

vlim_hour_peak = (np.round(iaq_high.hourly_peak_loads.min().min() / 1000, 0), 
                  np.round(fhe_high.hourly_peak_loads.max().max() / 1000, 0))

vlim_month_lf = (np.round(iaq_high.monthly_load_factors.min().min(), 3),
                 np.round(fhe_high.monthly_load_factors.max().max(), 3))

#%% Generate Individual Plots

baseline.Plots(root = root,
               vlim_month = vlim_month,
               vlim_hour = vlim_hour,
               vlim_hour_peak = vlim_hour_peak,
               vlim_month_lf = vlim_month_lf)

cmin_low.Plots(root = root,
               baseline = baseline,
               vlim_month = vlim_month,
               vlim_hour = vlim_hour,
               vlim_hour_peak = vlim_hour_peak,
               vlim_month_lf = vlim_month_lf)

cmin_med.Plots(root = root,
               baseline = baseline,
               vlim_month = vlim_month,
               vlim_hour = vlim_hour,
               vlim_hour_peak = vlim_hour_peak,
               vlim_month_lf = vlim_month_lf)

cmin_high.Plots(root = root,
                baseline = baseline,
                vlim_month = vlim_month,
                vlim_hour = vlim_hour,
                vlim_hour_peak = vlim_hour_peak,
                vlim_month_lf = vlim_month_lf)

iaq_low.Plots(root = root,
              baseline = baseline,
              vlim_month = vlim_month,
              vlim_hour = vlim_hour,
              vlim_hour_peak = vlim_hour_peak,
              vlim_month_lf = vlim_month_lf)

iaq_med.Plots(root = root,
              baseline = baseline,
              vlim_month = vlim_month,
              vlim_hour = vlim_hour,
              vlim_hour_peak = vlim_hour_peak,
              vlim_month_lf = vlim_month_lf)

iaq_high.Plots(root = root,
               baseline = baseline,
               vlim_month = vlim_month,
               vlim_hour = vlim_hour,
               vlim_hour_peak = vlim_hour_peak,
               vlim_month_lf = vlim_month_lf)

fhe_low.Plots(root = root,
              baseline = baseline,
              vlim_month = vlim_month,
              vlim_hour = vlim_hour,
              vlim_hour_peak = vlim_hour_peak,
              vlim_month_lf = vlim_month_lf)

fhe_med.Plots(root = root,
              baseline = baseline,
              vlim_month = vlim_month,
              vlim_hour = vlim_hour,
              vlim_hour_peak = vlim_hour_peak,
              vlim_month_lf = vlim_month_lf)

fhe_high.Plots(root = root,
               baseline = baseline,
               vlim_month = vlim_month,
               vlim_hour = vlim_hour,
               vlim_hour_peak = vlim_hour_peak,
               vlim_month_lf = vlim_month_lf)

#%% Generate Comparison Plots

runs = [iaq_low, cmin_low, fhe_low, 
        iaq_med, cmin_med, fhe_med,
        iaq_high, cmin_high, fhe_high]

units = 'btu'

plot.CompareInputProfiles(profiles, names, units, root = root, figsize=(20,20))
plot.CompareCompositions(runs, root = root, n = 3, m = 3)
plot.CompareCompositeAnnualProfiles(runs, root = root)
plot.CompareCompositeAnnualBaselineChanges(baseline, runs, root = root)
plot.CompareCompositeHourlyMeanBaselineChanges(runs, 
                                                vlim_hour = vlim_hour, 
                                                n = 3, m = 3, root = root,
                                                pct = False)
plot.CompareCompositeMonthlyBaselineChanges(runs, 
                                             vlim_month = vlim_month, 
                                             n = 3, m = 3, root = root)
plot.CompareCompositeHourlyPeakLoads(runs, 
                                      vlim_hour_peak = vlim_hour_peak,
                                      n = 3, m = 3, root = root)
plot.CompareCompositeMonthlyLoadFactors(runs,
                                         vlim_month_lf = vlim_month_lf,
                                         n = 3, m = 3, root = root)