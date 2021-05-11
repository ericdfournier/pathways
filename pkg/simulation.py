#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 10:02:19 2020

@author: edf
"""

#%% Package Imports 

import os
import numpy as np
import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl

os.chdir('/Users/edf/repos/cec_ng/pathways/')    
from pkg.plot import NameColors
from pkg.utils import InitializeSelection

#%% Seed Random Number Generator for Determininstic Simulation Outputs

np.random.seed(0)

#%% Specify Default Plot Parameters

font = {'size' : 18}
mpl.rc('font', **font)

#%% Create Simulation Class

class Simulation:
    
    # Initiate simulation parameters
    def __init__(self, tl, p, nm, w0, w1, n, k, rw, t_start, t_stop, t_end):
        
        self.tl = tl
        self.p = p 
        self.nm = nm
        self.w0 = w0
        self.w1 = w1
        self.n = n
        self.k = k
        self.rw = rw
        self.t_start = t_start
        self.t_stop = t_stop
        self.t_max = pd.date_range(start = self.t_start, 
                       end = self.t_stop,
                       freq = 'YS').shape[0]
        self.t_end = t_end
        self.selections = None
        self.counts = None
        self.individual_load_profiles = None
        self.composite_load_profile = None
        self.baseline_load_profile = None
        self.mean_hourly_baseline_change = None
        self.mean_hourly_baseline_change_pct = None
        self.monthly_baseline_change = None
        self.hourly_peak_loads = None
        self.annual_peak_loads = None
        self.monthly_load_factors = None
        self.fuel = None
        
    # Selection Receiver Dataframe Generator Function
    def Generate(self):
        '''Empty Receiver Dataframe Generator Function: to create the empty 
        receiver dataframe that is used as input to the selector function S. This 
        function takes as its inputs "t_start" the start year and "t_end" the end 
        year of the simulation time period and "n" the number of individuals in 
        the population to be simulated.'''
        
        ts = pd.date_range(start = self.t_start, 
                       end = self.t_stop,
                       freq = 'YS')
        s = np.arange(0, self.n, 1) 
        out = pd.DataFrame(index = ts, columns = s)
    
        return out
        
    # Probability Distribution Generator Function
    def Probability(self):
        '''Probability Generator Function: This function accepts four input
        parameters "p" a dataframe of hourly load profiles associated with each
        category, "w0" the relative weightings of the individual non-baseline 
        categories, "k" - the tunable sigmoid function parameter, and "t_max" - 
        the time horizon length parameter. The function outputs a time series of 
        transition probabilities for each of the non-baseline categories over the 
        time horizon [0,t_max].'''
        
        cp = self.p.columns.get_level_values(0).drop_duplicates().values[1:]
        x = np.linspace(0.0, 1.0, self.t_max)
        k_vec = np.repeat(self.k, self.t_max)
        p_cb = np.expand_dims((x - (k_vec * x)) / (k_vec - (2.0 * k_vec * np.abs(x)) + 1.0).T, 
                              axis=0)
        p_cp = np.repeat(((1.0 - p_cb) / len(cp)), len(cp), axis=0)
        p_cpw = p_cp + np.multiply(p_cp, np.expand_dims(self.w0, axis=0).T)
        p_c =  np.fliplr(np.vstack((p_cb, p_cpw))).T
    
        return p_c
    
    # Category Selector Function
    def Select(self, df, p_c, **kwargs):
        '''Category Selection Function: This function accepts three input
        parameters "df" - an empty receiver dataframe with a yearly timestamp index
        ranging from [0,t_max] and columns for each simulated household unit, "c"
        a list of potential categories with the zero index value as the baseline, 
        and "p_c" the time series of transition probabilities to each category as
        each time step generated as the output of the function P(k, t_max, c)'''
        
        custom_init = kwargs.get('custom_init', None)
                
        c0 = self.p.columns.get_level_values(0).drop_duplicates().values
        c1 = self.p.columns.get_level_values(1).drop_duplicates().values
        out = pd.DataFrame(index = df.index, columns = df.columns)
        out.loc[:,:] = [[(np.nan, np.nan)] * out.shape[1]] * out.shape[0]
                
        for n, i in enumerate(out.index):
                            
            if n == 0:
                row = pd.DataFrame(index = out.columns, columns = ['c0', 'c1'])
                row['c0'] = np.random.choice(c0, 
                                             size = out.shape[1], 
                                             replace = True, 
                                             p = p_c[n,:]).astype(float)                
                row['c1'] = np.random.choice(c1, 
                                             size = out.shape[1],
                                             p = self.w1,
                                             replace = True).astype(float)  
                if custom_init is not None:
                    r = row['c0'].copy(deep = True)
                    row['c0'] = InitializeSelection(r, custom_init)

                choices = row[['c0', 'c1']].apply(tuple, axis=1)
                out.loc[i,:] = choices
                                
            else:
                prev = out.iloc[n-1,:]
                prev_c0 = prev.apply(lambda x: float(x[0]))
                prev_c1 = prev.apply(lambda x: float(x[1]))
                pool = prev_c0 == 0
                row = pd.DataFrame(index = out.columns, columns = ['c0', 'c1'])
                row.loc[pool,'c0'] = np.random.choice(c0, 
                                             size = pool.sum(), 
                                             replace = True, 
                                             p = p_c[n,:]).astype(float)
                row.loc[pool,'c1'] = prev_c1.loc[pool]
                choices = row[['c0', 'c1']].apply(tuple, axis=1)
                out.loc[i,pool] = choices
            
        for col in out.columns:
            
            nan_ind = out[col] == (np.nan, np.nan)
            v_ind = nan_ind.replace(True, np.nan).last_valid_index()
            out.loc[nan_ind,col] = [out.loc[v_ind,col]] * nan_ind.sum()
            
        return out

    # Category Counts Aggregator Function 
    def Count(self, df):
        '''Category Counts Aggregator Function: This Function computes the number 
        of individuals that have been selected within a set of categories at each 
        timestep in the input selection dataframe. The function takes in as an 
        input "df" which corresponds to the input selections and "c" which 
        corresponds to the coding of selection categories.'''
        
        out = pd.DataFrame(index = df.index, columns = self.p.columns)
        
        for i, r in df.iterrows():
            c = r.value_counts()
            out.loc[i,c.index] = c.values
            
        return out

    # Load Profile Aggregation Function 
    def Aggregate(self, df):
        '''Aggregate Load Profiles Function: accepts as inputs a "counts"
        dataframe, corresponding to the counts of each category of household in
        each year of the time horizon, and a "profiles" dictionary which provides
        the average annual load profile associated with each household category.
        The output of the function is a single dataframe with the total community 
        load for each hour in the forecast time horizon disaggregated by
        household category.'''
        
        ts = pd.date_range(start = df.index.values[0], 
                           end = df.index.values[-1],
                           freq = 'H',
                           closed = 'left')  
        cols = df.columns
        out = pd.DataFrame(index = ts, columns = cols, dtype = float)
            
        # Drop leap years
        y_ind = out.index.is_leap_year
        m_ind = out.index.month == 2
        d_ind = out.index.day == 29
        ly_ind = y_ind & m_ind & d_ind
        out = out.loc[~ly_ind]
        
        for i, row in df.iterrows():
                    
            ind = out.index.year == i.year
            
            for c in cols:
                
                prof = self.p.loc[:,c]
                count = row.loc[c]
                out.loc[ind,c] = (prof * count).values
        
        out[out.isna()] = 0.0
        
        return out

    # Smooth Aggregate Hourly Load Profile Using Moving-Windowed Average 
    def Smooth(self, i_p):
        '''Smooth Function: Function to apply a moving window average of a
        specified size to the individual load profiles generated from the 
        simulation procedure. The function requires as inputs "i_p" a 
        the individual load profile dataframe.'''
        
        # Generate Output Smoothed Dataframe
        out = i_p.rolling(window = self.rw).mean()
        out = out.fillna(out.mean())
        
        return out
    
    # Trim Simulation Outputs to a Designated Time Horizon Endpoint
    def Trim(self):
        '''Function to trim an input simulation run outputs to a given input
        t-end for the end of the desired simulation time horizon.(Useful for
        plot legibility)'''
                
        annual_ind = self.counts.index < self.t_end
        hourly_ind = self.individual_load_profiles.index < self.t_end
        num_ind = self.hourly_peak_loads.index < pd.to_datetime(self.t_end).year
        
        self.counts = self.counts.loc[annual_ind,:]
        self.selections = self.selections.loc[annual_ind,:]
        
        self.individual_load_profiles = self.individual_load_profiles.loc[hourly_ind,:]
        self.composite_load_profile = self.composite_load_profile.loc[hourly_ind]
        
        if self.baseline_load_profile is not None:
            base_ind = self.baseline_load_profile.index < self.t_end
            self.baseline_load_profile = self.baseline_load_profile.loc[base_ind]
            
        if self.mean_hourly_baseline_change is not None:
            self.mean_hourly_baseline_change = self.mean_hourly_baseline_change.loc[num_ind,:]
            self.mean_hourly_baseline_change_pct = self.mean_hourly_baseline_change_pct.loc[num_ind, :]
            
        if self.monthly_baseline_change is not None:
            self.monthly_baseline_change = self.monthly_baseline_change.loc[num_ind,:]
            
        self.hourly_peak_loads = self.hourly_peak_loads.loc[num_ind]
        self.annual_peak_loads = self.annual_peak_loads.loc[num_ind]
        self.monthly_load_factors = self.monthly_load_factors.loc[num_ind]
        
        return

    # Run Simulation
    def Run(self, **kwargs):
        '''Run Function: Function to automate the process of generating
        category specific load profiles for some predefined time horizon based
        upon a set of input simulation parameters: "c" an input vector of integer
        categories, "w0" an input vector of category relative probability weights
        for the non-baseline first order categories, "w1" an input vector of 
        weights for all second order categories, "n" a scalar with the number of 
        individual households in the simulation, "k" the logistic growth function 
        scaling parameter [-1,1], and "t_start" and "t_stop" the yearly timestamps 
        of the simulation time horizon.'''

        # Extract baseline if exists
        baseline = kwargs.get('baseline', None)
        
        # Extract custom initialization
        custom_init = kwargs.get('custom_init', None)
                            
        # Construct Empty Selection Dataframe
        empty_selections = self.Generate()
        
        # Compute Probabilities
        probabilities = self.Probability()
    
        # Compute Selections
        if custom_init is not None:
            self.selections = self.Select(empty_selections, probabilities, custom_init = custom_init)
        else:
            self.selections = self.Select(empty_selections, probabilities) 
        
        # Compute Selection Category Counts
        self.counts = self.Count(self.selections)
    
        # Aggregate Profiles From Selection Counts
        individual_load_profiles_raw = self.Aggregate(self.counts)
            
        # Smooth Individually Aggregated Load Profiles
        self.individual_load_profiles = self.Smooth(individual_load_profiles_raw)
                
        # Sum for Composite Load
        self.composite_load_profile = self.individual_load_profiles.sum(axis=1)
        
        # Generate Baseline Hourly Load
        self.Stats(baseline = baseline)
        
        # Trim Outputs to T-end
        self.Trim()
        
        return
    
    # Assign Baseline Load Profile
    def StatsAssignBaselineLoadProfile(self, baseline):
        '''Function to assign the hourly baseline load profile from an input 
        baseline simulation run.'''
        
        self.baseline_load_profile = baseline.composite_load_profile
        
        return
    
    # Compute Mean Hourly Change from Baseline
    def StatsMeanHourlyBaselineChange(self, baseline):
        '''Function to compute the mean hourly change in the baseline composite
        load profile and the simulation composite load profile.'''
        
        hourly_diffs = self.composite_load_profile - baseline.composite_load_profile
        hourly_diffs_pct = hourly_diffs.divide(baseline.composite_load_profile).multiply(100.0)
        
        groups = [hourly_diffs.index.year, hourly_diffs.index.hour]
        
        grouped_diffs = hourly_diffs.groupby(groups).agg('mean')
        grouped_diffs_pct = hourly_diffs_pct.groupby(groups).agg('mean')
        
        unstacked_diffs = grouped_diffs.unstack(level=1)
        unstacked_diffs_pct = grouped_diffs_pct.unstack(level=1)
        
        self.mean_hourly_baseline_change = unstacked_diffs
        self.mean_hourly_baseline_change_pct = unstacked_diffs_pct
        
        return
    
    # Compute Monthly Change from Baseline
    def StatsMonthlyBaselineChange(self, baseline):
        '''Function to compute the total monthly change in the baseline
        composite load profile and the simulation composite load profile.'''
        
        hourly_diffs = self.composite_load_profile - baseline.composite_load_profile
        groups = [hourly_diffs.index.year, 
                  hourly_diffs.index.month]
        grouped_diffs = hourly_diffs.groupby(groups).agg('sum')
        unstacked_diffs = grouped_diffs.unstack(level=1)
        self.monthly_baseline_change = unstacked_diffs
    
    # Compute Hourly Peak Loads
    def StatsHourlyPeakLoads(self):
        '''Function to compute the absolute annual peak load for each hour
        of the day for each year within the simulation.'''
        
        groups = [self.composite_load_profile.index.year, 
                  self.composite_load_profile.index.hour]
        grouped_peaks = self.composite_load_profile.groupby(groups).agg('max')
        unstacked_peaks = grouped_peaks.unstack(level=1)
        self.hourly_peak_loads = unstacked_peaks

        return        
    
    # Compute Annual Peak Loads
    def StatsAnnualPeakLoads(self):
        '''Function to compute the absolute hourly peak load each year in 
        the simulation time horizon.'''
        
        self.annual_peak_loads = self.hourly_peak_loads.max(axis = 1)
        
        return
    
    # Compute Monthly Load Factors
    def StatsMonthlyLoadFactors(self):
        '''Function to compute the monthly load factors - as the ratio of the
        average monthly load to the peak monthly load - for each month and 
        each year of the simulation time horizon.'''
        
        groups = [self.composite_load_profile.index.year,
                  self.composite_load_profile.index.month]
        grouped_mean_loads = self.composite_load_profile.groupby(groups).agg('mean')
        grouped_max_peaks = self.composite_load_profile.groupby(groups).agg('max')
        grouped_load_factors = grouped_mean_loads / grouped_max_peaks
        unstacked_load_factors = grouped_load_factors.unstack(level=1)
        self.monthly_load_factors = unstacked_load_factors
        
        return
    
    # Compute Stats Subroutines
    def Stats(self, **kwargs):
        '''Function to automate the execution of the various stats calculation
        subroutines.'''
        
        baseline = kwargs.get('baseline', None)
        
        self.StatsHourlyPeakLoads()
        self.StatsAnnualPeakLoads()
        self.StatsMonthlyLoadFactors()
        
        if baseline is not None:
            self.StatsAssignBaselineLoadProfile(baseline)
            self.StatsMeanHourlyBaselineChange(baseline)
            self.StatsMonthlyBaselineChange(baseline)

        return
    
    # Generate Composition Plot 
    def PlotComposition(self, **kwargs):
        '''Function to plot the time series evolution in household type counts
        over the simulation time horizon.'''
        
        root = kwargs.get('root', None)

        c_p = self.counts.copy(deep=True)
        c_p[c_p.isna()] = 0.0
        data = c_p.divide(c_p.sum(axis=1), axis=0)
        x = data.index.get_level_values(0).values
        y = data.values.T.astype(float)
        
        colors = NameColors(self.nm)
        
        plt_name = 'Community_Composition'   
        fig, ax = plt.subplots(1, 1, figsize = (15,8))
    
        ax.stackplot(x, y, 
                     colors = colors, 
                     edgecolor = 'black',
                     linewidth = 0.25)
        ax.set_xlim([pd.to_datetime('1/1/2020'), pd.to_datetime('1/1/2045')])
        ax.set_ylim(0,1)
        ax.set_ylabel('Percentage of Households')
        ax.set_xlabel('Time (Years)')
        ax.grid(True)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        plt.xticks(rotation=45)
        ax.set_title(self.tl.replace('_',', ') + '\n' + plt_name.replace('_',' '))
        ax.legend(self.nm.values.flatten(), 
                  bbox_to_anchor = (1.05,0.5), 
                  loc = "center left", 
                  borderaxespad = 0)
        
        if root is not None:
            
            fig.savefig('{}{}_{}.png'.format(root, self.tl, plt_name), 
                        dpi = 150, 
                        bbox_inches='tight')   
            plt.close(fig)
        
        return fig, ax

    # Generate Individual Hourly Profiles Plot
    def PlotIndividualHourlyProfiles(self, **kwargs):
        '''Function to plot the hourly load profiles for each individual household
        type over the simulation time horizon.'''
        
        root = kwargs.get('root', None)
                
        plt_name = 'Individual_Hourly_Profiles'   
        fig, ax = plt.subplots(1, 1, figsize = (15,8))
        colors = NameColors(self.nm)

        if self.fuel is 'elec':
            data = self.individual_load_profiles.divide(1000)
            ax.set_ylabel('MW')
        else:
            data = self.individual_load_profiles.divide(1000000)
            ax.set_ylabel('MMbtu')

        data.plot(ax = ax, 
                  color = colors,
                  alpha = 0.75,
                  linewidth = 0.25)
        ax.set_xlim(['1/1/2020','1/1/2045'])
        ax.legend(self.nm.values.flatten(), 
                  bbox_to_anchor = (1.05,0.5), 
                  loc = "center left", 
                  borderaxespad = 0)

        ax.set_xlabel('Time (Hours)')
        ax.grid(True)
        ax.set_title(self.tl.replace('_',', ') + '\n' + plt_name.replace('_',' '))
        
        if root is not None:
            
            fig.savefig('{}{}_{}.png'.format(root, self.tl, plt_name), 
                        dpi = 150, 
                        bbox_inches='tight')
            plt.close(fig)
        
        return fig, ax

    # Generate Individual Annual Profiles Plot
    def PlotIndividualAnnualProfiles(self, **kwargs):
        '''Function to plot the annual load profiles for each individual household
        type over the simulation time horizon.'''
        
        root = kwargs.get('root', None)
        
        plt_name = 'Individual_Annual_Profiles'   
        fig, ax = plt.subplots(1, 1, figsize = (15,8))

        if self.fuel is 'elec':
            i_p_annual = self.individual_load_profiles.resample('Y').agg('sum').divide(1000000)
            ax.set_ylabel('GWh / Year')
        else: 
            i_p_annual = self.individual_load_profiles.resample('Y').agg('sum').divide(1000000000)
            ax.set_ylabel('Gbtu / Year')
        
        colors = NameColors(self.nm)
        i_p_annual.plot(ax = ax,
                        color = colors,
                        alpha = 1.0,
                        linewidth = 3.0)
        ax.set_xlim(['1/1/2020','1/1/2045'])
        ax.legend(self.nm.values.flatten(),
                  bbox_to_anchor = (1.05,0.5), 
                  loc = "center left", 
                  borderaxespad = 0)
        ax.set_xlabel('Time (Years)')
        ax.grid(True)
        ax.set_title(self.tl.replace('_',', ') + '\n' + plt_name.replace('_',' '))
        
        if root is not None:
            
            fig.savefig('{}{}_{}.png'.format(root, self.tl, plt_name), 
                        dpi = 150, 
                        bbox_inches='tight')
            plt.close(fig)
        
        return fig, ax

    # Generate Composite Hourly Profile Plot
    def PlotCompositeHourlyProfile(self, **kwargs):
        '''Function to plot the composite hourly load profile for all households
        over the simulation time horizon.'''
        
        root = kwargs.get('root', None)
                
        plt_name = 'Composite_Hourly_Profile'   
        fig, ax = plt.subplots(1, 1, figsize = (15,8))
        
        if self.fuel is 'elec':
            data = self.composite_load_profile.divide(1000)
            ax.set_ylabel('MW')
        else: 
            data = self.composite_load_profile.divide(1000000)
            ax.set_ylabel('MMbtu')

        data.plot(ax = ax, linewidth = 0.25)
        ax.set_xlim(['1/1/2020','1/1/2045'])
        ax.legend(['Combined Load'],                   
                  bbox_to_anchor = (1.05,0.5), 
                  loc = "center left", 
                  borderaxespad = 0)
        ax.set_xlabel('Time (Hours)')
        ax.grid(True)
        ax.set_title(self.tl.replace('_',', ') + '\n' + plt_name.replace('_',' '))

        if root is not None:
            
            fig.savefig('{}{}_{}.png'.format(root, self.tl, plt_name), 
                        dpi = 150, 
                        bbox_inches='tight')
            plt.close(fig)
        
        return fig, ax
    
    # Function to plot the composit Hourly Peak Load
    def PlotCompositeHourlyPeakLoads(self, **kwargs):
        '''Function to plot the annual peak hourly composite load for the 
        entire commmunity for each hour of the year and for each year of the 
        simulation time horizon.'''
    
        root = kwargs.get('root', None)
        vlim_hour_peak = kwargs.get('vlim_hour_peak', None)
            
        plt_name = 'Composite_Hourly_Peak_Loads'   
        fig, ax = plt.subplots(1, 1, figsize = (15,8))
        
        raw = self.hourly_peak_loads
        t_start = 2020
        t_stop = 2045

        if self.fuel is 'elec':
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
        ax.tick_params(axis='both', direction='out')
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(x_ticks, rotation = -90)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index.year)
        im = ax.imshow(data, vmin=vlim_hour_peak[0], vmax=vlim_hour_peak[1], 
                       cmap = 'viridis', interpolation='nearest', 
                       aspect='auto')
        cbar = plt.colorbar(im, ax = ax)

        if self.fuel is 'elec':
            cbar.set_label('\nPeak Load (MW)', rotation = 90)
        else:
            cbar.set_label('\nPeak Load (MMbtu)', rotation = 90)

        ax.set_ylabel('Forecast Year')
        ax.set_xlabel('Hour')
        ax.set_title(self.tl.replace('_',', ') + '\n' + plt_name.replace('_',' '))
        
        if root is not None:
            
            fig.savefig('{}{}_{}.png'.format(root, self.tl, plt_name), 
                        dpi = 150, 
                        bbox_inches='tight')
            plt.close(fig)
            
        return fig, ax
    
    # Function to plot the composite Monthly Load Factor
    def PlotCompositeMonthlyLoadFactors(self, **kwargs):
        '''Function to plot the composite monthly load factor for the 
        entire commmunity for each month of the year and for each year of the 
        simulation time horizon.'''
    
        root = kwargs.get('root', None)
        vlim_month_lf = kwargs.get('vlim_month_lf', None)
            
        plt_name = 'Composite_Monthly_Load_Factors'   
        fig, ax = plt.subplots(1, 1, figsize = (15,8))
        
        raw = self.monthly_load_factors
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
        ax.tick_params(axis='both', direction='out')
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(x_ticks, rotation = -90)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index.year)
        im = ax.imshow(data, vmin=vlim_month_lf[0], vmax=vlim_month_lf[1], 
                       cmap = 'plasma_r', interpolation='nearest', 
                       aspect='auto')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('\nLoad Factor (Unitless)', rotation = 90)
        ax.set_ylabel('Forecast Year')
        ax.set_xlabel('Month')
        ax.set_title(self.tl.replace('_',', ') + '\n' + plt_name.replace('_',' '))
        
        if root is not None:
            
            fig.savefig('{}{}_{}.png'.format(root, self.tl, plt_name), 
                        dpi = 150, 
                        bbox_inches='tight')
            plt.close(fig)
            
        return fig, ax
    
    # Generate Composite Hourly Profile Plot
    def PlotCompositeAnnualProfile(self, **kwargs):
        '''Function to plot the composite annual load profile for all households
        over the simulation time horizon.'''
        
        root = kwargs.get('root', None)

        plt_name = 'Composite_Annual_Profile'        
        fig, ax = plt.subplots(1, 1, figsize = (15,8))
        
        if self.fuel is 'elec':
            c_p_annual = self.composite_load_profile.resample('Y').agg('sum').divide(1000000)
            ax.set_ylabel('GWh / Year')
        else:
            c_p_annual = self.composite_load_profile.resample('Y').agg('sum').divide(1000000000)
            ax.set_ylabel('Gbtu / Year')

        c_p_annual.plot(ax = ax, linewidth = 3.0)
        ax.set_xlim(['1/1/2020','1/1/2045'])
        ax.legend(['Combined Load'],
                  bbox_to_anchor = (1.05,0.5), 
                  loc = "center left", 
                  borderaxespad = 0)
        ax.set_xlabel('Time (Years)')
        ax.grid(True)
        ax.set_title(self.tl.replace('_',', ') + '\n' + plt_name.replace('_',' '))

        if root is not None:
            
            fig.savefig('{}{}_{}.png'.format(root, self.tl, plt_name), 
                        dpi = 150, 
                        bbox_inches='tight')
            plt.close(fig)
        
        return fig, ax
    
    # Generate Hourly Baseline Deviation Plot
    def PlotCompositeHourlyBaselineChange(self, **kwargs):
        '''Function to plot the hourly change in the composite hourly load profles
        from the baseline scenario.'''
        
        root = kwargs.get('root', None)
                
        plt_name = 'Composite_Hourly_Baseline_Change'
        fig, ax = plt.subplots(1, 1, figsize = (15,8))
        
        b_dev_hourly = self.composite_load_profile - self.baseline_load_profile

        if self.fuel is 'elec':
            b_dev_hourly = b_dev_hourly.divide(1000)
            ax.set_ylabel('MW')
        else:
            b_dev_hourly = b_dev_hourly.divide(1000000)
            ax.set_ylabel('MMbtu')
        
        b_dev_hourly.plot(ax = ax, linewidth = 0.25)
        ax.set_xlim(['1/1/2020','1/1/2045'])
        ax.legend(['Change in Combined Load'],                   
                  bbox_to_anchor = (1.05,0.5), 
                  loc = "center left", 
                  borderaxespad = 0)
        ax.set_xlabel('Time (Hours)')
        ax.grid(True)
        ax.set_title(self.tl.replace('_',', ') + '\n' + plt_name.replace('_',' '))
        
        if root is not None:
            
            fig.savefig('{}{}_{}.png'.format(root, self.tl, plt_name), 
                        dpi = 150, 
                        bbox_inches='tight')
            plt.close(fig)

        return fig, ax
    
    # Generate Annual Baseline Deviation Plot
    def PlotCompositeAnnualBaselineChange(self, **kwargs):
        '''Function to plot the annual change in the composite load from the
        baseline scenario.'''
        
        root = kwargs.get('root', None)
        
        plt_name = 'Composite_Annual_Baseline_Change'
        fig, ax = plt.subplots(1, 1, figsize = (15,8))

        if self.fuel is 'elec':        
            comp_load_annual = self.composite_load_profile.resample('Y').agg('sum').divide(1000000)
            base_load_annual = self.baseline_load_profile.resample('Y').agg('sum').divide(1000000)
            ax.set_ylabel('GWh / Year')
        else:
            comp_load_annual = self.composite_load_profile.resample('Y').agg('sum').divide(1000000000)
            base_load_annual = self.baseline_load_profile.resample('Y').agg('sum').divide(1000000000)
            ax.set_ylabel('Gbtu / Year')

        b_dev_annual = comp_load_annual - base_load_annual
        
        b_dev_annual.plot(ax = ax, linewidth = 3.0)
        ax.set_xlim(['1/1/2020','1/1/2045'])
        ax.legend(['Change in Combined Load'],                 
                  bbox_to_anchor = (1.05,0.5), 
                  loc = "center left", 
                  borderaxespad = 0)
        ax.set_xlabel('Time (Years)')
        ax.grid(True)
        ax.set_title(self.tl.replace('_',', ') + '\n' + plt_name.replace('_',' '))
        
        if root is not None:
            
            fig.savefig('{}{}_{}.png'.format(root, self.tl, plt_name), 
                        dpi = 150, 
                        bbox_inches='tight')
            plt.close(fig)
        
        return fig, ax

    # Function to plot the composite hourly mean load change from the baseline
    def PlotCompositeHourlyMeanBaselineChange(self, **kwargs):
        '''Function to the plot the average hourly deviations of a simulation's
        composite load profile from the baseline on an hourly basis.'''
        
        root = kwargs.get('root', None)
        vlim_hour = kwargs.get('vlim_hour', None)
        pct = kwargs.get('pct', False)
        
        plt_name = 'Composite_Hourly_Mean_Baseline_Change'
        fig, ax = plt.subplots(1,1,figsize=(15,10))
        
        if self.fuel is 'elec':
            raw = self.mean_hourly_baseline_change.divide(1000)
            cb_label = '\nMean Deviation from Baseline (MW)'
        else: 
            raw = self.mean_hourly_baseline_change.divide(1000000)
            cb_label = '\nMean Deviation from Baseline (MMbtu)'

        t_start = 2020
        t_stop = 2045
        cmap = 'jet'
        
        if pct: 
            raw = self.mean_hourly_baseline_change_pct
            data = raw.loc[t_start:t_stop,:]
            plt_name = 'Comparison_of_Composite_Hourly_Mean_Baseline_Change_Percentages'
            cb_label = '\nMean Deviation from Baseline (%)'
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
        ax.tick_params(axis='both', direction='out')
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(x_ticks, rotation = -90)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index.year)
        im = ax.imshow(data, vmin=vlim_hour[0], vmax=vlim_hour[1], 
                       cmap = cmap, interpolation='nearest', 
                       aspect='auto')
        cbar = plt.colorbar(im, ax = ax)
        cbar.set_label(cb_label, rotation = 90)
        ax.set_ylabel('Forecast Year')
        ax.set_xlabel('Hour')
        ax.set_title(self.tl.replace('_',', ') + '\n' + plt_name.replace('_',' '))
        
        if root is not None:
            
            fig.savefig('{}{}_{}.png'.format(root, self.tl, plt_name), 
                        dpi = 150, 
                        bbox_inches='tight')
            plt.close(fig)
    
        return fig, ax
    
    # Function to plot the composite monthly total load change from baseline
    def PlotCompositeMonthlyBaselineChange(self, **kwargs):
        '''Function to the plot the deviations of a simulation's composite
        load profile from the baseline on a monthly basis.'''
        
        root = kwargs.get('root', None)
        vlim_month = kwargs.get('vlim_month', None)
        
        plt_name = 'Composite_Monthly_Baseline_Change'
        fig, ax = plt.subplots(1,1,figsize=(15,10))
        
        raw = self.monthly_baseline_change
        t_start = 2020
        t_stop = 2045
        data = raw.loc[t_start:t_stop,:]
        col = pd.date_range(start = '01/01/2020', 
                            end = '01/01/2021', freq = 'M', closed = 'left')
        ind = pd.date_range(start = '01/01/2020', 
                            end = '01/01/2046', freq = 'Y', closed = 'right')
        data.set_index(ind, inplace = True)
        data.columns = col

        if self.fuel is 'elec':
            data = data.divide(1000000)
        else: 
            data = data.divide(1000000000)
        
        x_ticks = [x.strftime('%b') for x in data.columns]
        ax.tick_params(axis='both', direction='out')
        ax.set_xticks(range(len(data.columns)))
        ax.set_xticklabels(x_ticks, rotation = -90)
        ax.set_yticks(range(len(data.index)))
        ax.set_yticklabels(data.index.year)
        im = ax.imshow(data, vmin=vlim_month[0], vmax=vlim_month[1], 
                       cmap = 'Spectral_r', interpolation='nearest', 
                       aspect='auto')
        cbar = fig.colorbar(im, ax=ax)

        if self.fuel is 'elec':
            cbar.set_label('\nDeviation from Baseline (GWh)', rotation = 90)
        else: 
            cbar.set_label('\nDeviation from Baseline (Gbtu)', rotation = 90)

        ax.set_ylabel('Forecast Year')
        ax.set_xlabel('Month')
        ax.set_title(self.tl.replace('_',', ') + '\n' + plt_name.replace('_',' '))
        
        if root is not None:
            
            fig.savefig('{}{}_{}.png'.format(root, self.tl, plt_name), 
                        dpi = 150, 
                        bbox_inches='tight')
            plt.close(fig)
    
        return fig, ax        
    
    def Plots(self, **kwargs):
        '''Function to generate all of the plot routines for a given
        simulation run's outputs without returning the individual figure and
        axes objects but instead writing the output plots to disk.'''
        
        baseline = kwargs.get('baseline', None)
        root = kwargs.get('root', None)
        vlim_hour = kwargs.get('vlim_hour', None)
        vlim_hour_peak = kwargs.get('vlim_hour_peak', None)
        vlim_month = kwargs.get('vlim_month', None)
        vlim_month_lf = kwargs.get('vlim_month_lf', None)
        
        _, _ = self.PlotComposition(root = root)
        _, _ = self.PlotIndividualHourlyProfiles(root = root)
        _, _ = self.PlotIndividualAnnualProfiles(root = root)
        _, _ = self.PlotCompositeHourlyProfile(root = root)
        _, _ = self.PlotCompositeAnnualProfile(root = root)
        _, _ = self.PlotCompositeHourlyPeakLoads(root = root,
                                                 vlim_hour_peak = vlim_hour_peak)
        _, _ = self.PlotCompositeMonthlyLoadFactors(root = root,
                                                    vlim_month_lf = vlim_month_lf)
        
        if baseline is not None:
            
            _, _ = self.PlotCompositeHourlyBaselineChange(root = root)
            _, _ = self.PlotCompositeAnnualBaselineChange(root = root)
            _, _ = self.PlotCompositeHourlyMeanBaselineChange(root = root,
                                                              vlim_hour = vlim_hour,
                                                              pct = False)
            _, _ = self.PlotCompositeMonthlyBaselineChange(root = root,
                                                           vlim_month = vlim_month)
        
        return
    
    def Pickle(self, root):
        '''Function to generate a pickled version of the output simulation 
        data structure generated from a simulation run that can be saved to 
        disc for ready import and subsequent reanalysis.'''
        
        file = open(root, 'wb')
        pk.dump(self, file)
        file.close()
        
        return
        
        
# %%
