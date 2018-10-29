#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
       _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Mon Oct 15 17:46:40 2018

@author: chrisunderwood
"""


import numpy as np
import os
import matplotlib.pyplot as plt


import sys
sys.path.insert(0, '/Users/chrisunderwood/Documents/Python')
import CUnderwood_Functions as func
import pandas as pd

def Convert_momentumToMeV(arr):
#    arr = arr *1.6e-19
    arr = (arr ** 2) / (2 * 9.11e-31)
    arr /= 1.6e-19
    return arr


def createPlot_dist_evo(allPx_integrated, all_xaxis, yaxis, Time = True):
    import matplotlib.colors as colors
    
    yaxis = Convert_momentumToMeV(yaxis)

    cmap = plt.cm.jet
    cmap.set_under(color='white')
    minDensityToPlot = 1e4
    
    if Time:
        xAxis_picoSec = all_xaxis * 1e12 #Turn into picoseconds
        plt.pcolormesh(xAxis_picoSec, yaxis,  allPx_integrated.T, norm=colors.LogNorm(), cmap = cmap, vmin = minDensityToPlot)
        xmin = xAxis_picoSec[0]; xmax=xAxis_picoSec[-1];
        plt.xlabel("Time (ps)")
    else:
        xAxis_micron = all_xaxis * 1e6 #Turn into mu m
        plt.pcolormesh(xAxis_micron, yaxis,  allPx_integrated.T, norm=colors.LogNorm(), cmap = cmap, vmin = minDensityToPlot)
        xmin = xAxis_micron[0]; xmax=xAxis_micron[-1];
        plt.xlabel(r"Distance $(\mu m)$")
        
    ymin= yaxis[0] ; ymax=yaxis[- 1]


    ymax = 0.7e7  
    
    cbar = plt.colorbar()
    cbar.set_label("Density (nparticles/cell)", rotation=270)
    plt.axis([xmin,xmax,ymin, ymax])
    plt.ylabel(r"Momentum ($kg.ms^{-1}$)")
    plt.show()

#==============================================================================
# Constants
#==============================================================================
q_e = 1.6e-19


hdrive = '/Volumes/CIDU_passport/2018_Epoch_vega_1/'
folder = '1010_SlurmJob/'


folderList =  func.listFolders(hdrive+folder)
name = []
rho = []
cellsPerL = []
electronsPerC = []
for f in folderList:
    name.append(f.split('/')[-2])
    for i in f.split('/')[-2].split('_'):
        if i.startswith('t'):
            rho.append(float(i[1:]))
        if i.startswith('l'):
            cellsPerL.append(float(i[1:]))
        if i.startswith('e'):
            electronsPerC.append(float(i[1:]))
d = {'name':name,'folder':folderList, 'rho':rho, 'cellsPerL':cellsPerL, 
     'electronsPerC':electronsPerC}

df = pd.DataFrame(d) 

print df           

maxEnergyPerSimulation = []
maxEnergyPos = []
electronsInLastTS = []
for fp in df['folder']:
    fp += 'Dist_evo/'
    data = np.loadtxt(fp +'px_vs_t.txt')
    allpos = np.loadtxt(fp + 'xaxis_x.txt')
    yaxis = np.loadtxt(fp + 'yaxis_p_eV.txt')
    px = ((yaxis / q_e) * 2 * 9.11e-31)**0.5
#    createPlot_dist_evo(data, allpos, px)
    
    #   This is currnetly momentum
    maxEnergyPerTS = []
    for line in data:
        maxEnergyPerTS.append(px[::-1][next((i for i, x in enumerate(line[::-1]) if x), None)])
    maxEnergyPerTS = np.array(maxEnergyPerTS)
    maxEnergyPerSimulation.append(max(maxEnergyPerTS))
    maxEnergyPos.append(allpos[func.nearposn(maxEnergyPerTS, max(maxEnergyPerTS))])
#    plt.plot(allpos, maxEnergyPerTS)
#    plt.show()
#    plt.plot(yaxis,  data[0], '.')
#    plt.plot(yaxis, data[-1], '.')
#    plt.xlim([0, 0.2e-30])
#    #plt.ylim([1, 1e9])
#    plt.yscale('log')
#    plt.xlabel('Momentum')
#    plt.show()
    electronsInLastTS.append(np.sum(data[-1]))
    
se = pd.Series(maxEnergyPerSimulation)
df['MaxEnergy'] = se.values
    
se = pd.Series(electronsInLastTS)
df['ElectonsAccelerated'] = se.values

se = pd.Series(maxEnergyPos)
df['maxEnergyPos'] = se.values



df_8e23 = df
tf = df_8e23['rho'] == 7e23
df_8e23 = df_8e23.drop(df_8e23[tf].index)
df_8e23 = df_8e23.drop(df_8e23[df_8e23['cellsPerL'] > 38].index)

fig, ax = plt.subplots(nrows = 3, sharex = True, figsize=(5,10))

#Plots 0 and 1 are closely related to each other
ax[0].plot(df_8e23['cellsPerL'],df_8e23['MaxEnergy'], '.-')
ax[0].set_ylabel('Max Momentum reached in sim')

ax[1].plot(df_8e23['cellsPerL'],df_8e23['maxEnergyPos'], '.-')
ax[1].set_ylabel('maxEnergyPos')

ax[2].plot(df_8e23['cellsPerL'],df_8e23['ElectonsAccelerated'] * q_e, '.-')
ax[2].set_ylabel('Electrons Charge Accelerated')
 
ax[2].set_xlabel('cells per lambda')

locs, labels = plt.xticks()
plt.tight_layout()
plt.show()

