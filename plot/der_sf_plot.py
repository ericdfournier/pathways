#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 09:20:31 2020

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

data_root = '/users/edf/repos/cec_ng/pathways/data/sf_der/'

profiles = pd.read_pickle(data_root + 'profiles.pkl')
names = pd.read_pickle(data_root + 'names.pkl')

baseline = pickle.load(open(data_root + 'baseline.pkl', 'rb'))

pv_low = pickle.load(open(data_root + 'pv_low.pkl', 'rb'))
pv_med = pickle.load(open(data_root + 'pv_med.pkl', 'rb'))
pv_high = pickle.load(open(data_root + 'pv_high.pkl', 'rb'))

bess_low = pickle.load(open(data_root + 'bess_low.pkl', 'rb'))
bess_med = pickle.load(open(data_root + 'bess_med.pkl', 'rb'))
bess_high = pickle.load(open(data_root + 'bess_high.pkl', 'rb'))

pvbess_low = pickle.load(open(data_root + 'pvbess_low.pkl', 'rb'))
pvbess_med = pickle.load(open(data_root + 'pvbess_med.pkl', 'rb'))
pvbess_high = pickle.load(open(data_root + 'pvbess_high.pkl', 'rb'))

#%% Set Plotting Parameters

root = root = '/Users/edf/repos/cec_ng/pathways/fig/sf_der/'

vlim_month = (np.round(pvbess_high.monthly_baseline_change.min().min() / 1000000, 2),
              -np.round(pvbess_high.monthly_baseline_change.min().min() / 1000000, 2))

vlim_hour = (np.round(pvbess_high.mean_hourly_baseline_change.min().min() / 1000, 0),
             -np.round(pvbess_high.mean_hourly_baseline_change.min().min() / 1000, 0))

vlim_hour_peak = (np.round(pvbess_high.hourly_peak_loads.min().min() / 1000), 
                  np.round(bess_high.hourly_peak_loads.max().max() / 1000))

vlim_month_lf = (np.round(pvbess_high.monthly_load_factors.min().min(), 2),
                 np.round(bess_high.monthly_load_factors.max().max(), 2))

#%% Generate Individual Plots

baseline.Plots(root = root,
               vlim_month = vlim_month,
               vlim_hour = vlim_hour,
               vlim_hour_peak = vlim_hour_peak,
               vlim_month_lf = vlim_month_lf)

pv_low.Plots(root = root,
             baseline = baseline,
             vlim_month = vlim_month,
             vlim_hour = vlim_hour,
             vlim_hour_peak = vlim_hour_peak,
             vlim_month_lf = vlim_month_lf)

pv_med.Plots(root = root,
             baseline = baseline,
             vlim_month = vlim_month,
             vlim_hour = vlim_hour,
             vlim_hour_peak = vlim_hour_peak,
             vlim_month_lf = vlim_month_lf)

pv_high.Plots(root = root,
              baseline = baseline, 
              vlim_month = vlim_month,
              vlim_hour = vlim_hour,
              vlim_hour_peak = vlim_hour_peak,
              vlim_month_lf = vlim_month_lf)

bess_low.Plots(root = root,
               baseline = baseline,
               vlim_month = vlim_month,
               vlim_hour = vlim_hour,
               vlim_hour_peak = vlim_hour_peak,
               vlim_month_lf = vlim_month_lf)

bess_med.Plots(root = root,
               baseline = baseline,
               vlim_month = vlim_month,
               vlim_hour = vlim_hour,
               vlim_hour_peak = vlim_hour_peak,
               vlim_month_lf = vlim_month_lf)

bess_high.Plots(root = root,
                baseline = baseline,
                vlim_month = vlim_month,
                vlim_hour = vlim_hour,
                vlim_hour_peak = vlim_hour_peak,
                vlim_month_lf = vlim_month_lf)

pvbess_low.Plots(root = root,
                 baseline = baseline,
                 vlim_month = vlim_month,
                 vlim_hour = vlim_hour,
                 vlim_hour_peak = vlim_hour_peak,
                 vlim_month_lf = vlim_month_lf)

pvbess_med.Plots(root = root,
                 baseline = baseline,
                 vlim_month = vlim_month,
                 vlim_hour = vlim_hour,
                 vlim_hour_peak = vlim_hour_peak,
                 vlim_month_lf = vlim_month_lf)

pvbess_high.Plots(root = root,
                  baseline = baseline,
                  vlim_month = vlim_month,
                  vlim_hour = vlim_hour,
                  vlim_hour_peak = vlim_hour_peak,
                  vlim_month_lf = vlim_month_lf)

#%% Generate Comparison Plots

runs = [pv_low, bess_low, pvbess_low, 
        pv_med, bess_med, pvbess_med,
        pv_high, bess_high, pvbess_high]

plot.CompareInputProfiles(profiles, names, root = root, figsize = (20,25))
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