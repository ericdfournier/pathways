#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 13:50:06 2020

@author: edf
"""

#%% Package Imports

import pandas as pd

#%% Population Data

pop = pd.read_pickle('/Users/edf/gdrive/projects/cec_natural_gas/analysis/scag/pkl/zip_approx_total_tier_2_pop.pkl')

#%% Set Query Parameters

zips = [91746, 91732]
ys = '2020-01-01'
ye = '2045-01-01'

#%% EV Calibration Data

ev = pd.read_pickle('/Users/edf/gdrive/projects/sgc_solar_pt/forecasts/pkl/outputs/ev_fcst_cap.pkl')

total_evs_ys = (ev.loc[ys,zips] * pop.loc[ys,zips]).sum()
total_evs_ye = (ev.loc[ye,zips] * pop.loc[ye,zips]).sum()

#%% DER Calibration Data

der = pd.read_pickle('/Users/edf/gdrive/projects/sgc_solar_pt/forecasts/pkl/outputs/der_fcst_cap.pkl')

total_der_ys = (der.loc[ys,zips] * 1000.0) * (pop.loc[ys,zips] / 1000.0) / 3.0
total_der_ye = (der.loc[ye,zips] * 1000.0) * (pop.loc[ye,zips] / 1000.0) / 3.0

