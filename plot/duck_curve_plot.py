#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 08:02:49 2020

@author: edf
"""

#%% Package Imports

import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
os.chdir('/Users/edf/repos/cec_ng/pathways/')    

import pkg.plot as plot

#%% Read In Full Home Electrification Data

sf_root = '/users/edf/repos/cec_ng/pathways/data/sf_electrification/'
mf_root = '/users/edf/repos/cec_ng/pathways/data/mf_electrification/'

stem = 'fhe_high.pkl'

sf = pd.read_pickle(sf_root + stem)
mf = pd.read_pickle(mf_root + stem)

#%% Compute Hourly Peak Loads Function

def ComputeHourlyPeakLoads(sim, year):
    '''Function to compute the hourly peak loads for a specific month
    and year in anticipation of handing off to a dedicated plot
    routine.'''
    
    ind = sim.composite_load_profile.index.year == year
    data = sim.composite_load_profile.loc[ind]

    groups = [data.index.month, 
            data.index.hour]
    grouped_peaks = data.groupby(groups).agg('max')
    hourly_peak_loads = grouped_peaks.unstack(level=1)
    
    return hourly_peak_loads

#%% Generate Peak Load Plotting Function

def PlotCompositeHourlyPeakLoads(sim, **kwargs):
    '''Function to plot the annual peak hourly composite load for the 
    entire commmunity for each hour of the year and for each year of the 
    simulation time horizon.'''
    
    root = kwargs.get('root', None)
    vlim_hour_peak = kwargs.get('vlim_hour_peak', None)
        
    plt_name = 'Composite_Hourly_Peak_Loads'   
    fig, ax = plt.subplots(1, 1, figsize = (15,8))
    
    raw = ComputeHourlyPeakLoads(sim, 2045)  
    t_start = 1
    t_stop = 12
    data = raw.loc[t_start:t_stop,:].divide(1000)
    col = pd.date_range(start = '01/01/2020', 
                        end = '01/02/2020', freq = 'H', closed = 'left')
    ind = pd.date_range(start = '01/01/2020', 
                        end = '01/01/2021', freq = 'M', closed = 'right')
    data.set_index(ind, inplace = True)
    data.columns = col
    
    x_ticks = [x.strftime('%-I %p') for x in data.columns]
    ax.tick_params(axis='both', direction='out')
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels(x_ticks, rotation = -90)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels(data.index.month_name())
    im = ax.imshow(data, vmin=vlim_hour_peak[0], vmax=vlim_hour_peak[1], 
                    cmap = 'viridis', interpolation='nearest', 
                    aspect='auto')
    cbar = plt.colorbar(im, ax = ax)
    cbar.set_label('\nPeak Load (MW)', rotation = 90)
    ax.set_ylabel('Forecast Month')
    ax.set_xlabel('Hour')
    ax.set_title('2045 - ' + sim.tl.replace('_',', ') + '\n' + plt_name.replace('_',' ') + ' by Month')
    
    if root is not None:
        
        fig.savefig('{}{}_{}_{}.png'.format(root, sim.tl, plt_name, '2045_monthly_focus'), 
                    dpi = 150, 
                    bbox_inches='tight')
        plt.close(fig)
        
    return fig, ax

# %% Set Plotting Parameters and Generate Plot

vlim_hour_peak = (np.round(sf.hourly_peak_loads.min().min() / 1000), 
                  np.round(sf.hourly_peak_loads.max().max() / 1000))

plot_root = '/users/edf/repos/cec_ng/pathways/fig/sf_electrification/'

fig, ax = PlotCompositeHourlyPeakLoads(sf,
                                    root = plot_root,
                                    vlim_hour_peak = vlim_hour_peak)

# %%
