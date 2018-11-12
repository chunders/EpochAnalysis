#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
       _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Mon Nov 12 14:52:58 2018

@author: chrisunderwood
    Produce the spectrum of electrons from the final sdf dump
"""

import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import sdf_helper as sh
import matplotlib as mpl
#from matplotlib.colors import colorConverter
import matplotlib.colors as colors
import matplotlib.gridspec as gridspec
from matplotlib import ticker

#==============================================================================
# Define global constants
#==============================================================================
q_e = 1.6e-19
m_e = 9.11e-31

def FilesInFolder(DirectoryPath, splice):
    files = os.listdir(DirectoryPath)
    shots = []
    inputDeck = []
    for i in files:
        if not i.startswith('.') and i.endswith('.sdf'):
            shots.append(i)
        if not i.startswith('.') and i.endswith('.deck'):
            inputDeck.append(i)
    
    # Sort
    timeStep = []
    for i in range(len(shots)):
#        print shots[i], '\t',shots[i][splice[0]:splice[1]]
        timeStep.append(int(shots[i][splice[0]:splice[-1]]))
    
    timeStep = np.asarray(timeStep)
    sorted_points = sorted(zip(timeStep, shots))
    timeStep = [point[0] for point in sorted_points]
    sdf_list = [point[1] for point in sorted_points]
        
    return sdf_list, timeStep, inputDeck

def stepFolders(folderPath):
    scratchPath = '/work/scratch/scarf585/' + folderPath
    savePath = '/home/clfg/scarf585/' + folderPath
    #Create folder to save into
    if not os.path.exists(savePath):
        os.makedirs(savePath)
        
    sdf_list, simTimeSteps, inputDeck = FilesInFolder(scratchPath, [0, -4]) 
    print '    Files in folder'
    print scratchPath
    print 'File Names list:', sdf_list
    print 
    print 'Time Step ints: ', simTimeSteps           
    if len(inputDeck) == 1:
        import shutil
        shutil.copy2(scratchPath+inputDeck[0], savePath)
    
    return scratchPath, savePath, sdf_list, simTimeSteps
    
def spectrumPlot(grid, spec, time,  savePath):
    plt.plot(grid, spec)
    plt.xlabel('Momentum')
    plt.ylabel('Number of Electrons')
    plt.title('Simulation time {:.2f}ps'.format(time*1e12))
    plt.savefig(savePath + 'FinalMomentumSpec.png')
    plt.clf()    
    
    plt.plot(((grid**2 / (2 * m_e)) / q_e) * 1e-6, spec)
    plt.title('Simulation time {:.2f}ps'.format(time*1e12))
    plt.xlabel('Energy (MeV)')
    plt.ylabel('Number of Electrons')
    plt.savefig(savePath + 'FinalEnergySpec.png')
    plt.clf()    
    
#==============================================================================
# Get the file input from the cmd line
#==============================================================================
parser = argparse.ArgumentParser()                                               
parser.add_argument("--folder", "-f", type=str, required=True)


folderPath = str(parser.parse_args().folder)   
scratchPath, savePath, sdf_list, simTimeSteps = stepFolders(folderPath)
    
# Load the last sdf file
print scratchPath + sdf_list[-1]
d = sh.getdata(scratchPath + sdf_list[-1])
px = d.dist_fn_x_px_electron.data
grid = d.dist_fn_x_px_electron.grid.data
t = d.Header['time']

if len(np.shape(px)) == 3:
    print 'ERROR IN PX FILE ---- Too many dimensions'

spectrum = np.sum(px, axis = 0)
spectrumPlot(grid[1] , spectrum, t, savePath)
np.savetxt(savePath + 'spectrum.txt',np.c_[grid[1], spectrum])

