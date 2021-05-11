
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 07:14:53 2020

@author: edf
"""

#%% Package Imports

import itertools as it
import pickle
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
os.chdir('/Users/edf/repos/cec_ng/pathways/')    

import pkg.plot as plot

#%% Generate Reader Function

def ExtractMonthlyPeakBaselineDeviations(root, paths):
    '''Helper function to extract and combine the baseline hourly load profile
    and composite hourly load profile from a single simulation run into a set
    of annual peak deviations from the baseline laod. This function helps 
    avoid the need to maintain an entire simulation object's contents in 
    memory for the synthesis analysis routines'''
    
    data_dict = {}
    
    for p in paths:
    
        pathway = pickle.load(open(root + p, 'rb'))
        data = pathway.composite_load_profile - pathway.baseline_load_profile
        data = (data.resample('D').agg('max').resample('Y').agg('median') / pathway.baseline_load_profile.resample('D').agg('max').resample('Y').agg('median')) * 100.0
        data_dict[p[:-4]] = data
    
    baseline = data_dict[p[:-4]].copy(deep = True)
    baseline.loc[:] = 0
    data_dict['baseline'] = baseline

    return data_dict

#%% Compute Cartesian Product List

def CartesianProduct(a, b, c):
    '''Function to compute a list of cartesian product combinations for a set
    of input data dictionary keys'''
    
    product = list(it.product(a, b, c))
    
    return product

#%% Read in DER Data

der_root = '/users/edf/repos/cec_ng/pathways/data/sf_der/'

der_paths = ['pv_low.pkl', 'pv_med.pkl', 'pv_high.pkl', 
         'bess_low.pkl', 'bess_med.pkl', 'bess_high.pkl', 
         'pvbess_low.pkl', 'pvbess_med.pkl', 'pvbess_high.pkl']

der_peak = ExtractMonthlyPeakBaselineDeviations(der_root, der_paths)

a = list(der_peak.keys())

#%% Read in Electrification Data

elec_root = '/users/edf/repos/cec_ng/pathways/data/sf_electrification/elec/'

elec_paths = ['cmin_low.pkl', 'cmin_med.pkl','cmin_high.pkl',
              'iaq_low.pkl', 'iaq_med.pkl', 'iaq_high.pkl',
              'fhe_low.pkl', 'fhe_med.pkl', 'fhe_high.pkl']

elec_peak = ExtractMonthlyPeakBaselineDeviations(elec_root, elec_paths)

b = list(elec_peak.keys())

#%% Read in EV Data

ev_root = '/users/edf/repos/cec_ng/pathways/data/sf_ev/'

ev_paths = ['ev_low.pkl','ev_med.pkl',  'ev_high.pkl']

ev_peak = ExtractMonthlyPeakBaselineDeviations(ev_root, ev_paths)

c = list(ev_peak.keys())

#%% Generate Combinations

combs = CartesianProduct(a, b, c)

#%% Peak Loads - EV-Low

p = [('pvbess_high', 'iaq_high', 'ev_low'),
     ('baseline', 'iaq_high', 'baseline')]
c = ['darkgreen','tab:green']
t = 'High Growth Rate Adoption of \nthe IAQ-Minor Dominant Electrification \n & PV+BESS Dominant Pathways'
fig, ax = plot.PathwaysFocusPlot(combs, 
                                 der = der_peak, 
                                 elec = elec_peak, 
                                 ev = ev_peak, 
                                 select = p, 
                                 color = c,
                                 title = t,
                                 style = 'peak_pct')

#%%

p = [('pvbess_high', 'cmin_high', 'ev_low'),
    ('baseline', 'cmin_high', 'baseline')]
c = ['darkblue','tab:blue']
t = 'High Growth Rate Adoption of \nthe IAQ-Moderate Dominant Electrification \n & PV+BESS Dominant Pathways'
fig, ax = plot.PathwaysFocusPlot(combs, 
                                 der = der_peak, 
                                 elec = elec_peak, 
                                 ev = ev_peak, 
                                 select = p, 
                                 color = c,
                                 title = t,
                                 style = 'peak_pct')

#%%

p = [('pvbess_high', 'fhe_high', 'ev_low'),
    ('baseline', 'fhe_high', 'baseline')]
c = ['darkred','tab:red']
t = 'High Growth Rate Adoption of \nthe FHE Dominant Electrification \n & PV+BESS Dominant Pathways'
fig, ax = plot.PathwaysFocusPlot(combs, 
                                 der = der_peak, 
                                 elec = elec_peak, 
                                 ev = ev_peak, 
                                 select = p, 
                                 color = c,
                                 title = t,
                                 style = 'peak_pct')

#%% Generate Net Load Profiles

b = 'baseline.pkl'

pvbess_path = 'pvbess_high.pkl'
pvbess_root = '/users/edf/repos/cec_ng/pathways/data/sf_der/'
pvbess = pickle.load(open(pvbess_root + pvbess_path, 'rb'))

pv_path = 'pv_high.pkl'
pv_root = '/users/edf/repos/cec_ng/pathways/data/sf_der/'
pv = pickle.load(open(pv_root + pv_path, 'rb'))

bess_path = 'bess_high.pkl'
bess_root = '/users/edf/repos/cec_ng/pathways/data/sf_der/'
bess = pickle.load(open(bess_root + bess_path, 'rb'))

iaq_path = 'iaq_high.pkl'
iaq_root = '/users/edf/repos/cec_ng/pathways/data/sf_electrification/elec/'
iaq = pickle.load(open(iaq_root + iaq_path, 'rb'))

cmin_path = 'cmin_high.pkl'
cmin_root = '/users/edf/repos/cec_ng/pathways/data/sf_electrification/elec/'
cmin = pickle.load(open(cmin_root + cmin_path, 'rb'))

fhe_path = 'fhe_high.pkl'
fhe_root = '/users/edf/repos/cec_ng/pathways/data/sf_electrification/elec/'
fhe = pickle.load(open(fhe_root + fhe_path, 'rb'))

pvbess_base = pickle.load(open(pvbess_root + b, 'rb'))
pv_base = pickle.load(open(pv_root + b, 'rb'))
bess_base = pickle.load(open(bess_root + b, 'rb'))

iaq_base = pickle.load(open(iaq_root + b, 'rb'))
cmin_base = pickle.load(open(cmin_root + b, 'rb'))
fhe_base = pickle.load(open(fhe_root + b, 'rb'))

pvbess_net_loads = pvbess.composite_load_profile - pvbess.baseline_load_profile
pv_net_loads = pv.composite_load_profile - pv.baseline_load_profile
bess_net_loads = bess.composite_load_profile - bess.baseline_load_profile

iaq_net_loads = iaq.composite_load_profile - iaq.baseline_load_profile
cmin_net_loads = cmin.composite_load_profile - cmin.baseline_load_profile
fhe_net_loads = fhe.composite_load_profile - fhe.baseline_load_profile

#%% Compute Pairwise Combinations

iaq_pvbess_overall_net = pvbess_net_loads + iaq_net_loads
cmin_pvbess_overall_net = pvbess_net_loads + cmin_net_loads
fhe_pvbess_overall_net = pvbess_net_loads + fhe_net_loads

iaq_pv_overall_net = pv_net_loads + iaq_net_loads
cmin_pv_overall_net= pv_net_loads + cmin_net_loads
fhe_pv_overall_net = pv_net_loads + fhe_net_loads

iaq_bess_overall_net = bess_net_loads + iaq_net_loads
cmin_bess_overall_net= bess_net_loads + cmin_net_loads
fhe_bess_overall_net = bess_net_loads + fhe_net_loads

#%% Compute Stats

def ComputeStats(net, baseline, year):

    ref = baseline.composite_load_profile.loc[year].divide(1000)
    stats = (net.loc[year].divide(1000.0) / ref) * 100.0

    return stats

#%% Compute Quantiles for Plotting

def ComputeHourlyQuantiles(df, interval):

    qtile_rng = np.arange(0,1.0,interval)
    hours = df.index.hour.unique()
    quantiles = pd.DataFrame(index = qtile_rng, columns = hours)
    for h in hours:
        ind = df.index.hour == h
        quantiles[h] = df.loc[ind].quantile(q = qtile_rng)
    quantiles = quantiles.transpose()
    
    return quantiles

#%%

def QuantilesFanPlot(df, interval, color, title, ylim, xlim, reverse_flows, peak_hours, figsize):

    fig, ax = plt.subplots(1,1, figsize = figsize)

    qtile_rng = np.arange(0,1.0,interval)
    alpha = np.sin(np.pi*qtile_rng)
    for i in range(len(qtile_rng)-1):
        ax.fill_between(df[qtile_rng[i]].index, 
                        df[qtile_rng[i]], 
                        df[qtile_rng[i-1]],
                        color=color,
                        linewidth=0.0,
                        alpha=alpha[i])
        ax.plot(df[0.5],
                color = 'black',
                linestyle='-',
                linewidth = 2.0)
    
    x = df.index

    if reverse_flows:
        ax.axhline(-100.0, color = 'red', zorder = 0)
        ax.fill_between(x, np.repeat(-100.0, x.shape[0]), np.repeat(ylim[0], x.shape[0]), color = 'red', alpha = 0.25, zorder = 0)
    
    if peak_hours:
        ax.axvspan(17, 21, alpha=0.25, color='orange')
        ax.axvline(17, color = 'orange', zorder = 0)
        ax.axvline(21, color = 'orange', zorder = 0)

    ax.set_title(title)
    ax.grid(True)

    fmt = '%.0f%%'
    hours = pd.date_range(start='1/1/2020', periods = 24, freq='1H')
    xticks = hours.strftime('%-I %p')
    ax.set_xticks(x)
    ax.set_xticklabels(xticks, rotation= 90)
    yticks = mtick.FormatStrFormatter(fmt)
    ax.yaxis.set_major_formatter(yticks)
    ax.grid(True)
    ax.set_ylim(ylim)
    plt.autoscale(enable=True, axis='x', tight=True)
    ax.set_xlim(xlim)

    return fig, ax

#%% PV-BESS Sweep + FHE

year = '2035'
interval = 0.05
color = 'tab:blue'

title = 'Parallel High Growth Rate Adoption \nof Full House Electrification & PV Only Dominant Pathways\n2035 - Net Load Change Snapshot'
ylim = (-275, 275)
xlim = (0, 23)
figsize = (15,10)
fhe_pv_net = ComputeStats(fhe_pv_overall_net, fhe_base, year)
fhe_pv_quantiles = ComputeHourlyQuantiles(fhe_pv_net, interval)
QuantilesFanPlot(fhe_pv_quantiles, interval, color, title, ylim, xlim, True, True, figsize)

title = None
xlim = (17,21)
ylim = (-100, 100)
figsize = (5,5)
QuantilesFanPlot(fhe_pv_quantiles, interval, color, title, ylim, xlim, False, False, figsize)

title = 'Parallel High Growth Rate the Adoption \nof Full House Electrification & BESS Only Dominant Pathways\n2035 - Net Load Change Snapshot'
ylim = (-275, 275)
xlim = (0, 23)
figsize = (15,10)
fhe_bess_net = ComputeStats(fhe_bess_overall_net, fhe_base, year)
fhe_bess_quantiles = ComputeHourlyQuantiles(fhe_bess_net, interval)
QuantilesFanPlot(fhe_bess_quantiles, interval, color, title, ylim, xlim, True, True, figsize)

title = None
xlim = (17,21)
ylim = (-100, 100)
figsize = (5,5)
QuantilesFanPlot(fhe_bess_quantiles, interval, color, title, ylim, xlim, False, False, figsize)

title = 'Parallel High Growth Rate Adoption \nof Full House Electrification & PV+BESS Dominant Pathways\n2035 - Net Load Change Snapshot'
ylim = (-275, 275)
xlim = (0, 23)
figsize = (15,10)
fhe_pvbess_net = ComputeStats(fhe_pvbess_overall_net, fhe_base, year)
fhe_pvbess_quantiles = ComputeHourlyQuantiles(fhe_pvbess_net, interval)
QuantilesFanPlot(fhe_pvbess_quantiles, interval, color, title, ylim, xlim, True, True, figsize)

title = None
xlim = (17,21)
ylim = (-100, 100)
figsize = (5,5)
QuantilesFanPlot(fhe_pvbess_quantiles, interval, color, title, ylim, xlim, False, False, figsize)

# %% Electrification Sweep + PV+BESS

year = '2035'
interval = 0.05
color = 'tab:green'

title = 'Parallel High Growth Rate Adoption \nof IAQ-Minor Electrification & PV+BESS Dominant Pathways\n2035 - Net Load Change Snapshot'
ylim = (-275, 275)
xlim = (0, 23)
figsize = (15,10)
iaq_pvbess_net = ComputeStats(iaq_pvbess_overall_net, iaq_base, year)
iaq_pvbess_quantiles = ComputeHourlyQuantiles(iaq_pvbess_net, interval)
QuantilesFanPlot(iaq_pvbess_quantiles, interval, color, title, ylim, xlim, True, True, figsize)

title = None
xlim = (17,21)
ylim = (-100, 100)
figsize = (5,5)
QuantilesFanPlot(iaq_pvbess_quantiles, interval, color, title, ylim, xlim, False, False, figsize)

title = 'Parallel High Growth Rate the Adoption \nof IAQ-Moderate Electrification & BESS+PV Dominant Pathways\n2035 - Net Load Change Snapshot'
ylim = (-275, 275)
xlim = (0, 23)
figsize = (15,10)
cmin_pvbess_net = ComputeStats(cmin_pvbess_overall_net, cmin_base, year)
cmin_pvbess_quantiles = ComputeHourlyQuantiles(cmin_pvbess_net, interval)
QuantilesFanPlot(cmin_pvbess_quantiles, interval, color, title, ylim, xlim, True, True, figsize)

title = None
xlim = (17,21)
ylim = (-100, 100)
figsize = (5,5)
QuantilesFanPlot(cmin_pvbess_quantiles, interval, color, title, ylim, xlim, False, False, figsize)

title = 'Parallel High Growth Rate Adoption \nof Full House Electrification & PV+BESS Dominant Pathways\n2035 - Net Load Change Snapshot'
ylim = (-275, 275)
xlim = (0, 23)
figsize = (15,10)
fhe_pvbess_net = ComputeStats(fhe_pvbess_overall_net, fhe_base, year)
fhe_pvbess_quantiles = ComputeHourlyQuantiles(fhe_pvbess_net, interval)
QuantilesFanPlot(fhe_pvbess_quantiles, interval, color, title, ylim, xlim, True, True, figsize)

title = None
xlim = (17,21)
ylim = (-100, 100)
figsize = (5,5)
QuantilesFanPlot(fhe_pvbess_quantiles, interval, color, title, ylim, xlim, False, False, figsize)

# %%
