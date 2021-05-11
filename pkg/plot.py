#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 08:10:01 2020

@author: edf
"""

#%% Package Imports 

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib
import pandas as pd
import numpy as np
from colour import Color
from itertools import chain

#%% Specify Default Plot Parameters

font = {'size' : 18}
matplotlib.rc('font', **font)

#%% Name Colors for Composition Plot
    
def NameColors(df):
    '''Name Colors: Function to generate a list of color names for the index
    values in the input dataframe. The objective of this function is to bundle
    prototypes into similar color groups for the purpose of generating the 
    composition plot.'''
    
    ind = df.index.get_level_values(0).unique().values 
    red = Color('red')
    blue = Color('blue')
    bases = list(blue.range_to(red, ind.shape[0]+1))
    colors = []
        
    for i in ind:
        levels = df.loc(axis=0)[i].index.get_level_values(0).values
        c = list(bases[i].range_to(bases[i+1], levels.shape[0]+1))
        colors.append(c[:-1])
    
    colors_list = list(chain.from_iterable(colors))
    out = [c.rgb for c in colors_list]
            
    return out


#%% Generate Single Year Load Profile Comparison Plot
    
def CompareInputProfiles(profiles, names, units, **kwargs):
    '''Function to provide a general overview comparison of the input single
    year annual load profiles used to generate the subsequent pathway 
    scenarios.'''
    
    figsize = kwargs.get('figsize', None)
    root = kwargs.get('root', None)
    n = len(profiles.keys())
    
    plt_name = 'Comparison_of_Input_Prototype_Load_Profiles'
    fig, ax = plt.subplots(n, 1, figsize = figsize, sharex=True, sharey=True)
    ax = ax.ravel()
    
    names = names.unstack(level=0).stack(level=-1)

    for i, k in enumerate(names.index.values):
        
        j = k[::-1]
                
        ax[i].plot(profiles.loc[:,j])
        ax[i].set_xlim([pd.to_datetime('1/1/2020'), pd.to_datetime('1/1/2021')])
        ax[i].annotate(names.loc[k,'name'].replace('_',', '), 
                       xy = (1.01, 0.5),
                       xycoords = 'axes fraction')
        ax[i].set_ylabel(units)
    
    ax[names.index.values.shape[0]-1].set_xticks(pd.date_range(start = '1/1/2020', end = '1/1/2021', freq = 'MS'))
    ax[names.index.values.shape[0]-1].set_xticklabels(profiles.index.month_name().unique().values, rotation = 90)
    fig.tight_layout()
        
    if root is not None:
        
        fig.savefig('{}{}.png'.format(root, plt_name), 
                    dpi = 150, 
                    bbox_inches='tight')   
        plt.close(fig)
        
    return fig, ax
    
#%% Generate Composition Comparison Plot 
    
def CompareCompositions(runs, **kwargs):
    '''Function to plot the time series evolution in household type counts
    over the simulation time horizon for multiple pathway scenarios.'''
    
    root = kwargs.get('root', None)
    n = kwargs.get('n', None)
    m = kwargs.get('m', None)
    
    plt_name = 'Comparison_of_Community_Compositions'   
    fig, ax = plt.subplots(n, m, figsize = (40,35))
    ax = ax.ravel()
    
    for i, r in enumerate(runs):
        
        if r is None:
            continue

        c_p = r.counts.copy(deep=True)
        c_p[c_p.isna()] = 0.0
        data = c_p.divide(c_p.sum(axis=1), axis=0)
        x = data.index.get_level_values(0).values
        y = data.values.T.astype(float)
        
        colors = NameColors(r.nm)
    
        ax[i].stackplot(x, y, 
                        colors = colors, 
                        edgecolor = 'black',
                        linewidth = 0.25)
        ax[i].set_xlim([pd.to_datetime('1/1/2020'), pd.to_datetime('1/1/2045')])
        ax[i].set_ylim(0,1)
        ax[i].set_ylabel('Percentage of Households')
        ax[i].set_xlabel('Time (Years)')
        ax[i].grid(True)
        ax[i].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax[i].set_title(r.tl.replace('_',', '))
    
    if root is not None:
        
        fig.savefig('{}{}.png'.format(root, plt_name), 
                    dpi = 150, 
                    bbox_inches='tight')   
        plt.close(fig)
    
    return fig, ax

#%% Generate Composite Hourly Profile Plot
def CompareCompositeAnnualProfiles(runs, **kwargs):
    '''Function to plot the composite annual load profiles for all households
    over the simulation time horizon, simultaneously, for an input list
    corresponding to multiple simulation runs.'''
    
    root = kwargs.get('root', None)

    plt_name = 'Comparison_of_Composite_Annual_Profiles'        
    fig, ax = plt.subplots(1, 1, figsize = (15,8))
    
    raw = []
    col = []
    
    for r in runs:
        
        if r is None:
            continue
        
        if r.fuel == 'elec':
            raw.append(r.composite_load_profile.resample('Y').agg('sum').divide(1000000))
            col.append(r.tl.replace('_', ', '))
            ax.set_ylabel('GWh / Year')
        else:
            raw.append(r.composite_load_profile.resample('Y').agg('sum').divide(1000000000))
            col.append(r.tl.replace('_', ', '))
            ax.set_ylabel('Gbtu / Year')

    data = pd.concat(raw, axis = 1)
    data.columns = col
    
    data.plot(ax = ax, linewidth = 3.0)
    ax.set_xlim(['1/1/2020','1/1/2045'])
    ax.legend(col,
              bbox_to_anchor = (1.05,0.5), 
              loc = "center left", 
              borderaxespad = 0)
    ax.set_xlabel('Time (Years)')
    ax.grid(True)
    ax.set_title(plt_name.replace('_',' '))
    ax.set_ylim((np.floor(data.min().min()), np.ceil(data.max().max())))

    if root is not None:
        
        fig.savefig('{}{}.png'.format(root, plt_name), 
                    dpi = 150, 
                    bbox_inches='tight')
        plt.close(fig)
    
    return fig, ax

#%% Generate Annual Baseline Deviation Plot
def CompareCompositeAnnualBaselineChanges(baseline, runs, **kwargs):
    '''Function to plot the annual change in the composite load from the
    baseline scenario for multiple input sumulation runs, simultaneously.'''
    
    root = kwargs.get('root', None)
    
    plt_name = 'Comparison_of_Composite_Annual_Baseline_Changes'
    fig, ax = plt.subplots(1, 1, figsize = (15,8))
    
    raw = []
    col = []
    
    for r in runs:
        
        if r is None:
            continue

        if r.fuel == 'elec':
            comp_load_annual = r.composite_load_profile.resample('Y').agg('sum').divide(1000000)
            base_load_annual = baseline.composite_load_profile.resample('Y').agg('sum').divide(1000000)
            ax.set_ylabel('GWh / Year')
        else: 
            comp_load_annual = r.composite_load_profile.resample('Y').agg('sum').divide(1000000000)
            base_load_annual = baseline.composite_load_profile.resample('Y').agg('sum').divide(1000000000)
            ax.set_ylabel('Gbtu / Year')
        
        b_dev_annual = comp_load_annual - base_load_annual
        raw.append(b_dev_annual)
        col.append(r.tl.replace('_', ', '))
        
    data = pd.concat(raw, axis = 1)
    data.columns = col
    
    data.plot(ax = ax, linewidth = 3.0)
    ax.set_xlim(['1/1/2020','1/1/2045'])
    ax.legend(col,                 
              bbox_to_anchor = (1.05,0.5), 
              loc = "center left", 
              borderaxespad = 0)
    ax.set_xlabel('Time (Years)')
    ax.grid(True)
    ax.set_title(plt_name.replace('_',' '))
    ax.set_ylim((np.floor(data.min().min()), np.ceil(data.max().max())))
    
    if root is not None:
        
        fig.savefig('{}{}.png'.format(root, plt_name), 
                    dpi = 150, 
                    bbox_inches='tight')
        plt.close(fig)
    
    return fig, ax

#%% Generate Hourly Baseline Change Composite Plot

def CompareCompositeHourlyMeanBaselineChanges(runs, **kwargs):
    '''Function to the plot the average hourly deviations for a set of
    simulation runs' composite load profiles from the baseline 
    on an hourly basis.'''
    
    root = kwargs.get('root', None)
    vlim_hour = kwargs.get('vlim_hour', None)
    n = kwargs.get('n', None)
    m = kwargs.get('m', None)
    pct = kwargs.get('pct', False)
    
    fig, ax = plt.subplots(n, m, figsize=(40,35))
    ax = ax.ravel()
    
    for i, r in enumerate(runs):
        
        if r is None:
            continue
        
        if r.fuel == 'elec':
            raw = r.mean_hourly_baseline_change.divide(1000)
        else: 
            raw = r.mean_hourly_baseline_change.divide(1000000)

        t_start = 2020
        t_stop = 2045
        cmap = 'jet'
        
        if pct: 
            raw = r.mean_hourly_baseline_change_pct
            data = raw.loc[t_start:t_stop,:]
            plt_name = 'Comparison_of_Composite_Hourly_Mean_Baseline_Change_Percentages'
        else:
            data = raw.loc[t_start:t_stop,:]
            plt_name = 'Comparison_of_Composite_Hourly_Mean_Baseline_Changes'
        
        col = pd.date_range(start = '01/01/2020', 
                            end = '01/02/2020', freq = 'H', closed = 'left')
        ind = pd.date_range(start = '01/01/2020', 
                            end = '01/01/2046', freq = 'Y', closed = 'right')
        data.set_index(ind, inplace = True)
        data.columns = col
        
        x_ticks = [x.strftime('%-I %p') for x in data.columns]
        ax[i].tick_params(axis='both', direction='out')
        ax[i].set_xticks(range(len(data.columns)))
        ax[i].set_xticklabels(x_ticks, rotation = -90)
        ax[i].set_yticks(range(len(data.index)))
        ax[i].set_yticklabels(data.index.year)
        ax[i].imshow(data, vmin=vlim_hour[0], vmax=vlim_hour[1], 
                       cmap = cmap , interpolation='nearest', 
                       aspect='auto')
        ax[i].set_ylabel('Forecast Year')
        ax[i].set_xlabel('Hour')
        ax[i].set_title(r.tl.replace('_',', '))
        
    if root is not None:
        
        fig.savefig('{}{}.png'.format(root, plt_name), 
                    dpi = 150, 
                    bbox_inches='tight')
        plt.close(fig)

    return fig, ax

#%% Generate Monthly Baseline Change Comparison Plot
    
def CompareCompositeMonthlyBaselineChanges(runs, **kwargs):
    '''Function to the plot the deviations of a simulation's composite
    load profile from the baseline on a monthly basis.'''
    
    root = kwargs.get('root', None)
    vlim_month = kwargs.get('vlim_month', None)
    n = kwargs.get('n', None)
    m = kwargs.get('m', None)
    
    plt_name = 'Comparison_of_Composite_Monthly_Baseline_Changes'
    fig, ax = plt.subplots(n, m, figsize=(40,35))
    ax = ax.ravel()
    
    for i, r in enumerate(runs):
        
        if r is None:
            continue
    
        raw = r.monthly_baseline_change
        t_start = 2020
        t_stop = 2045
        data = raw.loc[t_start:t_stop,:]
        col = pd.date_range(start = '01/01/2020', 
                            end = '01/01/2021', freq = 'M', closed = 'left')
        ind = pd.date_range(start = '01/01/2020', 
                            end = '01/01/2046', freq = 'Y', closed = 'right')
        data.set_index(ind, inplace = True)
        data.columns = col

        if r.fuel == 'elec':
            data = data.divide(1000000)
        else: 
            data = data.divide(1000000000)
        
        x_ticks = [x.strftime('%b') for x in data.columns]
        ax[i].tick_params(axis='both', direction='out')
        ax[i].set_xticks(range(len(data.columns)))
        ax[i].set_xticklabels(x_ticks, rotation = -90)
        ax[i].set_yticks(range(len(data.index)))
        ax[i].set_yticklabels(data.index.year)
        ax[i].imshow(data, vmin=vlim_month[0], vmax=vlim_month[1], 
                       cmap = 'Spectral_r', interpolation='nearest', 
                       aspect='auto')
        ax[i].set_ylabel('Forecast Year')
        ax[i].set_xlabel('Month')
        ax[i].set_title(r.tl.replace('_',', '))
    
    if root is not None:
        
        fig.savefig('{}{}.png'.format(root, plt_name), 
                    dpi = 150, 
                    bbox_inches='tight')
        plt.close(fig)

    return fig, ax

#%% Function to compare the composit Hourly Peak Load across runs
def CompareCompositeHourlyPeakLoads(runs, **kwargs):
    '''Function to compare the annual peak hourly composite loads for the 
    entire commmunity for each hour of the year and for each year of the 
    simulation time horizon across multiple simulation runs.'''

    root = kwargs.get('root', None)
    vlim_hour_peak = kwargs.get('vlim_hour_peak', None)
    n = kwargs.get('n', None)
    m = kwargs.get('m', None)
        
    plt_name = 'Comparison_of_Composite_Hourly_Peak_Loads'   
    fig, ax = plt.subplots(n, m, figsize = (40,35))
    ax = ax.ravel()
    
    for i, r in enumerate(runs):
        
        if r is None:
            continue
    
        raw = r.hourly_peak_loads
        t_start = 2020
        t_stop = 2045

        if r.fuel == 'elec':
            data = raw.loc[t_start:t_stop,:].divide(1000)
        else:
            data = raw.loc[t_start:t_stop,:].divide(1000000)

        col = pd.date_range(start = '01/01/2020', 
                            end = '01/02/2020', freq = 'H', closed = 'left')
        ind = pd.date_range(start = '01/01/2020', 
                            end = '01/01/2046', freq = 'Y', closed = 'right')
        data.set_index(ind, inplace = True)
        data.columns = col
        
        x_ticks = [x.strftime('%-I %p') for x in data.columns]
        ax[i].tick_params(axis='both', direction='out')
        ax[i].set_xticks(range(len(data.columns)))
        ax[i].set_xticklabels(x_ticks, rotation = -90)
        ax[i].set_yticks(range(len(data.index)))
        ax[i].set_yticklabels(data.index.year)
        ax[i].imshow(data, vmin=vlim_hour_peak[0], vmax=vlim_hour_peak[1], 
                       cmap = 'viridis', interpolation='nearest', 
                       aspect='auto')
        ax[i].set_ylabel('Forecast Year')
        ax[i].set_xlabel('Hour')
        ax[i].set_title(r.tl.replace('_',', '))
    
    if root is not None:
        
        fig.savefig('{}{}.png'.format(root, plt_name), 
                    dpi = 150, 
                    bbox_inches='tight')
        plt.close(fig)
        
    return fig, ax

#%% Function to compare the composite Monthly Load Factors across runs
def CompareCompositeMonthlyLoadFactors(runs, **kwargs):
    '''Function to compare the composite monthly load factor for the 
    entire commmunity for each month of the year and for each year of the 
    simulation time horizon for multiple simulation runs.'''

    root = kwargs.get('root', None)
    vlim_month_lf = kwargs.get('vlim_month_lf', None)
    n = kwargs.get('n', None)
    m = kwargs.get('m', None)
        
    plt_name = 'Comparison_of_Composite_Monthly_Load_Factors'   
    fig, ax = plt.subplots(n, m, figsize = (40,35))
    ax = ax.ravel()
    
    for i, r in enumerate(runs):
        
        if r is None:
            continue
    
        raw = r.monthly_load_factors
        t_start = 2020
        t_stop = 2045
        data = raw.loc[t_start:t_stop,:]
        col = pd.date_range(start = '01/01/2020', 
                            end = '01/01/2021', freq = 'M', closed = 'left')
        ind = pd.date_range(start = '01/01/2020', 
                            end = '01/01/2046', freq = 'Y', closed = 'right')
        data.set_index(ind, inplace = True)
        data.columns = col
        
        x_ticks = [x.strftime('%b') for x in data.columns]
        ax[i].tick_params(axis='both', direction='out')
        ax[i].set_xticks(range(len(data.columns)))
        ax[i].set_xticklabels(x_ticks, rotation = -90)
        ax[i].set_yticks(range(len(data.index)))
        ax[i].set_yticklabels(data.index.year)
        ax[i].imshow(data, vmin=vlim_month_lf[0], vmax=vlim_month_lf[1], 
                       cmap = 'plasma_r', interpolation='nearest', 
                       aspect='auto')
        ax[i].set_ylabel('Forecast Year')
        ax[i].set_xlabel('Month')
        ax[i].set_title(r.tl.replace('_',', '))
    
    if root is not None:
        
        fig.savefig('{}{}.png'.format(root, plt_name), 
                    dpi = 150, 
                    bbox_inches='tight')
        plt.close(fig)
        
    return fig, ax

#%%  Pathways Focus Plot

def PathwaysFocusPlot(combs, **kwargs):
    '''Function to generate an output plot highlighting a single combination
    of pathways in terms of their overall combined deviations from the baseline
    pathway runs'''
    
    der_dict = kwargs.get('der', None)
    elec_dict = kwargs.get('elec', None)
    ev_dict = kwargs.get('ev', None)
    match_scenarios = kwargs.get('match', None)
    select_scenarios = kwargs.get('select', None)
    color = kwargs.get('color', None)
    title = kwargs.get('title', None)
    style = kwargs.get('style', None)

    fig, ax = plt.subplots(1, 1, figsize = (10,10))
    i = 0

    for c in combs:
    
        data = der_dict[c[0]] + elec_dict[c[1]] + ev_dict[c[2]]
        
        if match_scenarios is not None:
            
            if c == ('baseline', 'baseline', 'baseline'):
                ax.plot(data, color = 'black', linewidth = 3.0)
            elif all(s in c for s in match_scenarios):
                ax.plot(data, color = color, linewidth = 3.0, alpha = 0.75)
            else:
                ax.plot(data, color = 'grey', linewidth = 0.25)
                
        elif select_scenarios is not None:

            if c == ('baseline', 'baseline', 'baseline'):
                ax.plot(data, color = 'black', linewidth = 3.0)
            elif c in select_scenarios:
                ax.plot(data, color = color[i], linewidth = 3.0, alpha = 1.0)
                i+=1
            else:
                ax.plot(data, color = 'grey', linewidth = 0.25)
            
    ax.set_xlim([pd.to_datetime('2021'),pd.to_datetime('2045')])
    ax.set_xlabel('Time (Years)')
    if style == 'annual':
        ax.set_ylabel('Deviations from Baseline\n Annual Total Loads (GWh)')
    elif style == 'peak_pct':
        ax.set_ylabel('Deviations from Baseline\n Median Daily Peak Loads (%)')
        fmt = '%.0f%%'
        yticks = mtick.FormatStrFormatter(fmt)
        ax.yaxis.set_major_formatter(yticks)

    ax.grid(True)
    ax.set_title(title)

    return fig, ax   
    