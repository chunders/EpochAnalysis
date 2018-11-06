#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
       _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Tue Oct 30 11:00:05 2018

@author: chrisunderwood
    Epoch Analysis for the 1D data
"""

import numpy as np
import matplotlib.pyplot as plt
import sdf_helper as sh
import os
import matplotlib.colors as colors


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


class Load_1D_data():
    def __init__(self, path):
        self.d = sh.getdata(path)

    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    def assignDistFuncGrid(self):
        self.xGrid = self.d.dist_fn_x_px_electron.grid.data[0]
        self.yGrid_Momentum = self.d.dist_fn_x_px_electron.grid.data[1]
        self.distFncData = self.d.dist_fn_x_px_electron.data[:,:,0].T

        
    
    def DisplayDistributionFunc_correctAxis(self):
    #Distribution Funciton display
        self.assignDistFuncGrid()
        X, Y = np.meshgrid(self.xGrid, self.yGrid_Momentum)
        plt.pcolor(X, Y, self.distFncData)
        plt.colorbar()
        plt.show()
        
    def DisplayDistributionFunc_imshow(self):
    #Distribution Funciton display
        plt.imshow(self.d.dist_fn_x_px_electron.data[:,:,0].T)
        plt.colorbar()
        plt.show()        
    
    def numberDensity_and_Efields(self, smoothBoxSize = 20):
    #Plot the number density and the electric field on top of each other
    #Smooth the number density to see the shape better    
            
        NumberDensity = self.d.Derived_Number_Density.data
        NDSmooth = self.smooth(NumberDensity, smoothBoxSize)
        
        grid = (self.d.Electric_Field_Ex.grid.data[0][1:] + self.d.Electric_Field_Ex.grid.data[0][:-1]) * 0.5
        
        ax = plt.subplot()
        ax.plot(grid, self.d.Electric_Field_Ex.data, label = 'X')
        ax.plot(grid, self.d.Electric_Field_Ey.data , label = 'Y')
        ax.plot(grid, self.d.Electric_Field_Ez.data, label = 'Z')
        ax.legend();
        ax1 = ax.twinx()
        ax1.plot(grid, NDSmooth)
        plt.show()
        
    def SpectrumFromDistFunction(self, plotting=False):
        self.assignDistFuncGrid()
        self.sumFuncData = self.distFncData.sum(axis=1)
        output = np.c_[self.yGrid_Momentum, self.sumFuncData]
        if plotting:
            plt.plot(self.yGrid_Momentum, self.sumFuncData)
            plt.show()
        return output
    
    def time(self):
        return self.d.Header.values()[10]
        
sdf_list, simTimeSteps, inputDeck = FilesInFolder('.', [0, -4])
momentumEvo = []
posAndTime = []
for sdf in sdf_list[10:110]:
    oneD = Load_1D_data(sdf)
#    oneD.numberDensity_and_Efields()
#    oneD.DisplayDistributionFunc_correctAxis()
#    oneD.DisplayDistributionFunc_imshow()
    timeSpectrum = oneD.SpectrumFromDistFunction()
    momentumEvo.append(timeSpectrum[:,1])
    posAndTime.append([np.average(oneD.xGrid), oneD.time()])

momentumEvo = np.array(momentumEvo)
posAndTime = np.array(posAndTime)

plt.pcolormesh(posAndTime[:,1], oneD.yGrid_Momentum, momentumEvo.T, norm = colors.LogNorm())
plt.colorbar()

    

