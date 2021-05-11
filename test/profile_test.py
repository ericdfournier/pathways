#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 13:51:35 2020

@author: edf
"""

#%% Package Imports

import os
os.chdir('/Users/edf/repos/cec_ng/pathways/')    

import pkg.utils as utils
import pandas as pd
import matplotlib.pyplot as plt

#%% Read in Building Model Load Profiles
    
d = '/Users/edf/repos/cec_ng/pathways/prototypes/sf_electrification/'
f = 'Site Energy|Total (E)'
p, nm = utils.Read(d, f)

t_d = '/Users/edf/repos/cec_ng/pathways/prototypes/iaq_test/'
t_f = 'Site Energy|Total (E)'
t_p, t_nm = utils.Read(t_d, t_f)

#%% Extract Data and Plot

i = (2,1)
test = pd.merge(p[i], t_p[i], left_index = True, right_index = True)
test.columns = ['v1', 'v2']

fig, ax = plt.subplots(3,1,figsize=(30,10), sharex = True, sharey = True)
ax[0].plot(p[i], color = 'tab:blue')
ax[1].plot(t_p[i], color = 'tab:orange')
ax[2].plot(test['v1'] - test['v2'], color = 'tab:red')
fig.tight_layout()