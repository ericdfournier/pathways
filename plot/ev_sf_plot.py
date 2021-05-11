#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:22:49 2020

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

data_root = '/users/edf/repos/cec_ng/pathways/data/sf_ev/'

profiles = pd.read_pickle(data_root + 'profiles.pkl')
names = pd.read_pickle(data_root + 'names.pkl')

baseline = pickle.load(open(data_root + 'baseline.pkl', 'rb'))

ev_low = pickle.load(open(data_root + 'ev_low.pkl', 'rb'))
ev_med = pickle.load(open(data_root + 'ev_med.pkl', 'rb'))
ev_high = pickle.load(open(data_root + 'ev_high.pkl', 'rb'))

#%% Set Plot Parameters

root = root = '/Users/edf/repos/cec_ng/pathways/fig/sf_ev/'

vlim_month = (-np.round(ev_high.monthly_baseline_change.max().max() / 1000000, 2),
              np.round(ev_high.monthly_baseline_change.max().max() / 1000000, 2))

vlim_hour = (-np.round(ev_high.mean_hourly_baseline_change.max().max() / 1000, 0),
             np.round(ev_high.mean_hourly_baseline_change.max().max() / 1000, 0))

vlim_hour_peak = (np.round(ev_low.hourly_peak_loads.min().min() / 1000), 
                  np.round(ev_high.hourly_peak_loads.max().max() / 1000))

vlim_month_lf = (np.round(ev_low.monthly_load_factors.min().min(), 2),
                 np.round(ev_high.monthly_load_factors.max().max(), 2))

#%% Generate Individual Plots

baseline.Plots(root = root,
               vlim_month = vlim_month,
               vlim_hour = vlim_hour,
               vlim_hour_peak = vlim_hour_peak,
               vlim_month_lf = vlim_month_lf)

ev_low.Plots(root = root,
             baseline = baseline,
             vlim_month = vlim_month,
             vlim_hour = vlim_hour,
             vlim_hour_peak = vlim_hour_peak,
             vlim_month_lf = vlim_month_lf)

ev_med.Plots(root = root,
             baseline = baseline,
             vlim_month = vlim_month,
             vlim_hour = vlim_hour,
             vlim_hour_peak = vlim_hour_peak,
             vlim_month_lf = vlim_month_lf)

ev_high.Plots(root = root,
              baseline = baseline,
              vlim_month = vlim_month,
              vlim_hour = vlim_hour,
              vlim_hour_peak = vlim_hour_peak,
              vlim_month_lf = vlim_month_lf)

#%% Generate Comparison Plots

runs = [ev_low, None, None,
        ev_med, None, None,
        ev_high, None, None]

plot.CompareInputProfiles(profiles, names, root = root, figsize=(20,20))
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