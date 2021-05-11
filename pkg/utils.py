#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 13:05:53 2020

@author: edf
"""

import os
import pandas as pd
import numpy as np
        
#%% Override the Initial Selections Using an Input Array of Counts

def InitializeSelection(row, custom_init):
    '''Funtion to randomly select a subset of input row values for custom
    inititalization during the selection procedure'''
    
    inds = np.random.choice(np.arange(0,row.shape[0]), size = np.sum(custom_init), replace = False)
    inds_cs = np.cumsum(custom_init)
    inds_cs = np.hstack((0,inds_cs))
    
    for i in np.arange(0,inds_cs.shape[0]-1):
        ind_val = inds[inds_cs[i]:inds_cs[i+1]]
        row.loc[ind_val] = i+1 
    
    return row

#%% Load EV Load Profiles
    
def LoadEV(path, index):
    '''EV load profiles loader function: accepts the filepath to an EV load 
    profile that must be added to the baseline load profile to generate the 
    total load for the ev scenario household. The function requires an input
    dataframe with the corresponding baseline profiles as well as the index
    value indicating the profile to which the EV loads should be added.'''
    
    df = pd.read_csv(path, header = None)
    df.columns = [index[1]]
    df['ts'] = pd.date_range(start = '01/01/2020',
                             end = '12/31/2020',
                             freq = 'H', 
                             closed = 'left')
    df = df.set_index('ts', drop = True)
    out = df.loc[:,index[1]]
    
    return out

#%% Read Multiple EV Load Profiles from Input Directory
    
def ReadEV(directory):
    '''Read EV Function: accepts and input directory where multiple EV load
    profiles are stored as csv files and a formatted dataframe of baseline
    profiles with multi-index columns.'''
        
    A = []
    B = []
    ind = []
    names = []
    dfs = []
    
    for f in os.scandir(directory):
        if (f.path.endswith('.csv') and f.is_file()):
            a, b, c = f.name.split('-')
            A.append(int(a))
            B.append(int(b))
            ind.append((int(a),int(b)))
            names.append(c.replace('_',' ')[:-4])
            dfs.append(LoadEV(f, ind[-1]))
            
    out = pd.concat(dfs, axis=1)
    out.columns = pd.MultiIndex.from_tuples(tuple(zip(A,B)))
    out = out.sort_index(axis=1)
    
    names = pd.DataFrame(names)
    names.index = pd.MultiIndex.from_tuples(ind)
    names = names.sort_index(axis = 0)
    names.columns = ['name']
    
    return out, names
 
#%% Read DER Performance Outputs from HOMER Optimizations

def LoadREopt(path):
    '''REopt DER performance output reader function: reads in and formats a
    timestampped dataframe representing the hourly performance characteristics
    or a PV, BESS, or coupled PV+BESS system linked to with the input filepath
    '''
    
    fields = ['Electric Load (kW)',
              'PV Serving Load (kW)', 
              'PV Charging Battery (kW)',
              'PV Exporting to Grid (kW)',
              'Battery Discharging (kW)', 
              'Battery to Grid (kW)',
              'Grid Charging Battery (kW)']
    df = pd.read_csv(path, usecols = fields)
    df['ts'] = pd.date_range(start = '01/01/2020',
                       end = '12/31/2020',
                       freq = 'H', 
                       closed = 'left')
    df = df.set_index('ts', drop = True)
    df = df.fillna(0)
    df['kWh'] = df['Electric Load (kW)'] + (df['Grid Charging Battery (kW)']) - (df['PV Serving Load (kW)'] + df['Battery Discharging (kW)'] + df['Battery to Grid (kW)']) - df['PV Exporting to Grid (kW)']
    out = df['kWh']
    
    return out

#%% Function to Read in Multiple HOMER Run Files
    
def ReadREopt(directory):
    '''Read REpot function: Function ti iteratively loop through all of the
    hourly performance output files associated with a given scenario model and
    construct an output dataframe for use in the subsequent selection and
    aggregation procdures. Function requires as inputs the "directory" where
    input HOMER performance results are located.'''
    
    A = []
    B = []
    ind = []
    names = []
    dfs = []
    
    for f in os.scandir(directory):
        if (f.path.endswith('.csv') and f.is_file()):
            a, b, c = f.name.split('-')
            A.append(int(a))
            B.append(int(b))
            ind.append((int(a),int(b)))
            names.append(c.replace('_',' ')[:-4])
            dfs.append(LoadREopt(f))
            
    out = pd.concat(dfs, axis=1)
    out.columns = pd.MultiIndex.from_tuples(tuple(zip(A,B)))
    out = out.sort_index(axis=1)
    
    names = pd.DataFrame(names)
    names.index = pd.MultiIndex.from_tuples(ind)
    names = names.sort_index(axis = 0)
    names.columns = ['name']
    
    return out, names

#%% Read Load Profile Data from B-Opt Simulations
    
def LoadBEopt(path, field, unit):
    
    '''Load Profile Reader Function: reads in and formats a timestampped 
    dataframe representing the hourly load profile associated with the B-Opt
    prototype model linked to with the input filepath.'''
    
    columns = pd.read_csv(path, nrows=0).columns
    ind = columns.str.contains(field, regex=False)
    if ind.any():
        col = columns[ind].values[0]
        df = pd.read_csv(path, usecols = [col], low_memory = False)
        df = df.loc[1:]
        df = pd.to_numeric(df[col]).to_frame()
        df['ts'] = pd.date_range(start = '01/01/2020',
                       end = '12/31/2020',
                       freq = 'H', 
                       closed = 'left')
        out = df.set_index('ts', drop = True)
        out.columns = [unit]
    else:
        ts = pd.date_range(start = '01/01/2020',
                       end = '12/31/2020',
                       freq = 'H', 
                       closed = 'left')
        out = pd.DataFrame(columns = ['btu'], index = ts, dtype=float)
        out.loc[:,[unit]] = 0.0
    
    return out
    
#%% Read B-Opt Simulated Load Profile Data
    
def ReadBEopt(directory, field, unit):
    '''ReadBEopt Function: Function to iteratively loop through all of the 
    hourly load profiles associated with a given simulation run and construct 
    an output dataframe for use in the subsequent selection and aggregation 
    procedures. Function requires as inputs the "directory" where the input
    load profile CSV files are located and the "field" corresponding to the 
    hourly kWh usage in each file, and a "unit" variable corresponding to the
    energy unit associated with the previous input field'''

    A = []
    B = []
    ind = []
    names = []
    dfs = []
    
    for f in os.scandir(directory):
        if (f.path.endswith('.csv') and f.is_file()):
            a, b, c = f.name.split('-')
            A.append(int(a))
            B.append(int(b))
            ind.append((int(a),int(b)))
            names.append(c.replace('_',' ')[:-4])
            dfs.append(LoadBEopt(f, field, unit))
            
    out = pd.concat(dfs, axis=1)
    out.columns = pd.MultiIndex.from_tuples(tuple(zip(A,B)))
    out = out.sort_index(axis=1)
    
    names = pd.DataFrame(names)
    names.index = pd.MultiIndex.from_tuples(ind)
    names = names.sort_index(axis = 0)
    names.columns = ['name']
    
    return out, names
# %%
