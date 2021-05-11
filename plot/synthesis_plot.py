#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 07:14:53 2020

@author: edf
"""

#%% Package Imports

import itertools as it
import pickle
import os
os.chdir('/Users/edf/repos/cec_ng/pathways/')    

import pkg.plot as plot

#%% Generate Reader Function

def ExtractAnnualBaselineDeviations(root, paths):
    '''Helper function to extract and combine the baseline hourly load profile
    and composite hourly load profile from a single simulation run into a set
    of annual total deviations from the baseline laod. This function helps 
    avoid the need to maintain an entire simulation object's contents in 
    memory for the synthesis analysis routines'''
    
    data_dict = {}
    
    for p in paths:
    
        pathway = pickle.load(open(root + p, 'rb'))
        data = pathway.composite_load_profile - pathway.baseline_load_profile
        data = data.resample('Y').agg('sum').divide(1000000)
        data_dict[p[:-4]] = data
    
    baseline = data_dict[p[:-4]].copy(deep = True)
    baseline.loc[:] = 0
    data_dict['baseline'] = baseline

    return data_dict

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
        data = pathway.composite_load_profile
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

der_annual = ExtractAnnualBaselineDeviations(der_root, der_paths)
der_peak = ExtractMonthlyPeakBaselineDeviations(der_root, der_paths)

a = list(der_annual.keys())

#%% Read in Electrification Data

elec_root = '/users/edf/repos/cec_ng/pathways/data/sf_electrification/elec/'

elec_paths = ['cmin_low.pkl', 'cmin_med.pkl','cmin_high.pkl',
              'iaq_low.pkl', 'iaq_med.pkl', 'iaq_high.pkl',
              'fhe_low.pkl', 'fhe_med.pkl', 'fhe_high.pkl']

elec_annual = ExtractAnnualBaselineDeviations(elec_root, elec_paths)
elec_peak = ExtractMonthlyPeakBaselineDeviations(elec_root, elec_paths)

b = list(elec_annual.keys())

#%% Read in EV Data

ev_root = '/users/edf/repos/cec_ng/pathways/data/sf_ev/'

ev_paths = ['ev_low.pkl','ev_med.pkl',  'ev_high.pkl']

ev_annual = ExtractAnnualBaselineDeviations(ev_root, ev_paths)
ev_peak = ExtractMonthlyPeakBaselineDeviations(ev_root, ev_paths)

c = list(ev_annual.keys())

#%% Generate Combinations

combs = CartesianProduct(a, b, c)

#%% PVBESS-High

p = ['pvbess_high']
c = 'tab:cyan'
t = 'High Growth PV+BESS Dominant\n Composite Pathways'
fig, ax = plot.PathwaysFocusPlot(combs, 
                                 der = der_annual, 
                                 elec = elec_annual, 
                                 ev = ev_annual, 
                                 match = p, 
                                 color = c, 
                                 title = t,
                                 style = 'annual')

#%% FHE-High

p = ['fhe_high']
c = 'tab:orange'
t = 'High Growth Full Home Electrification Dominant\n Composite Pathways'
fig, ax = plot.PathwaysFocusPlot(combs, 
                                 der = der_annual, 
                                 elec = elec_annual, 
                                 ev = ev_annual, 
                                 match = p, 
                                 color = c, 
                                 title = t,
                                 style = 'annual')

#%% EV-High

p = ['ev_high']
c = 'tab:purple'
t = 'High Growth EV Adoption\n Composite Pathways'
fig, ax = plot.PathwaysFocusPlot(combs, 
                                 der = der_annual, 
                                 elec = elec_annual, 
                                 ev = ev_annual, 
                                 match = p, 
                                 color = c, 
                                 title = t,
                                 style = 'annual')


#%% Maximum Pathway

p = [('baseline', 'fhe_high', 'ev_high'),
     ('pv_high', 'cmin_low', 'ev_low')]
c = [ 'tab:pink','tab:olive']
t = 'Extreme High and Low Load Growth\n Composite Pathways'
fig, ax = plot.PathwaysFocusPlot(combs, 
                                 der = der_annual, 
                                 elec = elec_annual, 
                                 ev = ev_annual, 
                                 select = p, 
                                 color = c, 
                                 title = t,
                                 style = 'annual')

#%% Neutral Growth Pathways

p = [('pvbess_low', 'fhe_low', 'ev_low'),
     ('pvbess_med', 'fhe_med', 'ev_med'),
     ('pvbess_high', 'fhe_high', 'ev_high')]
c = ['tab:green','tab:blue','tab:red']
t = 'Parallel Growth\n Composite Pathways'
fig, ax = plot.PathwaysFocusPlot(combs, 
                                 der = der_annual, 
                                 elec = elec_annual, 
                                 ev = ev_annual, 
                                 select = p, 
                                 color = c,
                                 title = t,
                                 style = 'annual')
