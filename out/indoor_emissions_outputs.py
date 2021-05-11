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

#%% Read Runs From Base Directory

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

def Combine(sf_runs, mf_runs, scalar, unit):

    '''Function to combine the hourly changes in load from the 
    baseline pathway between a pair of single-family and 
    multi-family runs'''

    runs = tuple(zip(sf_runs, mf_runs))

    combined = {}

    for sf_run, mf_run in runs:

        sf_change = sf_run.composite_load_profile - sf_run.baseline_load_profile

        mf_change = mf_run.composite_load_profile - mf_run.baseline_load_profile

        diff = (sf_change + mf_change) / scalar
        diff.name = unit

        combined[sf_run.tl] = diff

    return combined

#%% Define Function to Extract Combined Data for a Single Output Year

def Extract(combined, years):

    '''Function to extract the total combined hourly changes in load from the baseline pathway for a specific year'''

    extracted = {}

    for k,v in combined.items():
        for y in years:
            ind = v.index.year == y
            extract = v.loc[ind]
            key = str(y) + '_' + k
            extracted[key] = extract

    return extracted

#%% Subtract LG Appliance Gas use from Total Gas Use for Heating Gas

def Subtract(extracted_total_gas, extracted_la_gas):
    '''Function to subtract the large appliance gas consumption fraction from the total gas time series for each extracted year and pathway.'''

    subtracted = {}
    keys = extracted_total_gas.keys()
    
    for k in keys:
        subtracted[k] = extracted_total_gas[k] - extracted_la_gas[k]

    return subtracted

#%% Writer Function to Write Extracted Outputs to Files

def Write(extracts, name, data_root):
    '''Function to pickle extracted data for external use.'''

    fh = open(data_root + name, 'wb')
    pickle.dump(extracts, fh)

    return

#%% Read Simulation Outputs from Files for SF Total Gas

sf_data_root_total_gas = '/users/edf/repos/cec_ng/pathways/data/sf_electrification/gas/total/'

sf_runs_total_gas = Read(sf_data_root_total_gas)

#%% Read Simulation Outputs from Files for SF Large Appliance Gas

mf_data_root_total_gas = '/users/edf/repos/cec_ng/pathways/data/mf_electrification/gas/total/'

mf_runs_total_gas = Read(mf_data_root_total_gas)

#%% Read Simulation Outputs from Files for SF Total Gas

sf_data_root_la_gas = '/users/edf/repos/cec_ng/pathways/data/sf_electrification/gas/large_appliances/'

sf_runs_la_gas = Read(sf_data_root_la_gas)

#%% Read Simulation Outputs from Files for MF

mf_data_root_la_gas = '/users/edf/repos/cec_ng/pathways/data/mf_electrification/gas/large_appliances/'

mf_runs_la_gas = Read(mf_data_root_la_gas)

#%% Combine and Extract Total Gas

scalar = 1000000 # convert to MMbtu
unit = 'MMbtu'
combined_total_gas = Combine(sf_runs_total_gas, mf_runs_total_gas, scalar, unit)
combined_la_gas = Combine(sf_runs_la_gas, mf_runs_la_gas, scalar, unit)

#%% Extract Total and Large Appliance Gas Use for Selected Years

years = [2020, 2025, 2030, 2035, 2040, 2045]
extracted_total_gas = Extract(combined_total_gas, years)
extracted_la_gas = Extract(combined_la_gas, years)

#%% Compute Non-LA Gas Use from Total

extracted_non_la_gas = Subtract(extracted_total_gas, extracted_la_gas)

#%% Write Outputs to File

output_root = '/users/edf/gdrive/projects/cec_natural_gas/analysis/indoor/input_data/'

Write(extracted_la_gas, 'extracted_la_gas.pkl', output_root)
Write(extracted_non_la_gas, 'extracted_non_la_gas.pkl', output_root)
