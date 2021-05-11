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


#%% Define Reader Function

def Read(data_root):

    files = os.listdir(data_root)
    runs = []

    for f in files:
        if f.endswith('med.pkl') or f.endswith('low.pkl') or f.endswith('high.pkl'):
            r = pickle.load(open(data_root + f, 'rb'))
            runs.append(r)
        else:
            continue

    return runs

#%% Define Function to Combine SF-MF Outputs

def Combine(sf_run, mf_run, scalar, unit):

    '''Function to combine the hourly changes in load from the 
    baseline pathway between a pair of single-family and 
    multi-family runs'''

    sf_change = sf_run.composite_load_profile
    sf_baseline = sf_run.baseline_load_profile

    mf_change = mf_run.composite_load_profile 
    mf_baseline = mf_run.baseline_load_profile

    #NOTE: Multiply by -1.0 here to format for AVERT
    output = -1.0 * ((sf_change + mf_change) / scalar)
    baseline = -1.0 * ((sf_baseline + mf_baseline) / scalar) 

    output.name = unit
    baseline.name = unit

    return baseline, output

#%% Define Function to Extract Combined Data for a Single Output Year

def Extract(combined, year):

    '''Function to extract the total combined hourly changes in load from the baseline pathway for a specific year'''

    ind = combined.index.year == year
    output = combined.loc[ind]

    return output

#%% Read Simulation Outputs from Files for SF

sf_data_root = '/users/edf/repos/cec_ng/pathways/data/sf_electrification/elec/'

sf_runs = Read(sf_data_root)

#%% Read Simulation Outputs from Files for MF

mf_data_root = '/users/edf/repos/cec_ng/pathways/data/mf_electrification/elec/'

mf_runs = Read(mf_data_root)

#%% Zip Runs into Tuple

runs = tuple(zip(sf_runs, mf_runs))

#%% Specific Output Years to Extract

years = [2020, 2025, 2030, 2035, 2040, 2045]

#%% Combine and Extract Output

output_root = '/users/edf/gdrive/projects/cec_natural_gas/analysis/avert/input_data/'

scalar = 1000
unit = 'MW'

for sf, mf in runs:
    baseline, combined = Combine(sf, mf, scalar, unit)
    for y in years:
        combined_output = Extract(combined, y)
        baseline_output = Extract(baseline, y)
        baseline_output.to_csv('{}{}_{}.csv'.format(output_root, str(y), 'Electrification_Baseline'))
        combined_output.to_csv('{}{}_{}.csv'.format(output_root, str(y), sf.tl))


# %%
