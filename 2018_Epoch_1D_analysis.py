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

import argparse


def FilesInFolder(DirectoryPath, splice):
#==============================================================================
# Create a list of all the files that are important from the epoch sim    
#==============================================================================
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
    def __init__(self, path, savePath):
        self.d = sh.getdata(path)
        self.savePath = savePath
        self.count = int(path[-8:-4])
        #Create folder to save into
        if not os.path.exists(self.savePath):
            os.makedirs(self.savePath)
        
    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    def assignDistFuncGrid(self):
        
        self.xGrid = self.d.dist_fn_x_px_electron.grid.data[0]
        self.yGrid_Momentum = self.d.dist_fn_x_px_electron.grid.data[1]
        print ('Shape of distFunc ', np.shape(self.d.dist_fn_x_px_electron.data))
        self.distFncData = self.d.dist_fn_x_px_electron.data.T
#        self.distFncData = self.d.dist_fn_x_px_electron.data[:,:,0].T
        print ('Shape of distFunc ', np.shape(self.distFncData))
        
    
    def DisplayDistributionFunc_correctAxis(self):
        #Distribution Funciton display
        self.assignDistFuncGrid()
        X, Y = np.meshgrid(self.xGrid, self.yGrid_Momentum)
        plt.pcolormesh(X, Y, self.distFncData)
        plt.colorbar()
        if not os.path.exists(self.savePath + 'DistFunc/'):
            os.makedirs(self.savePath+ 'DistFunc/')
        plt.savefig(self.savePath + 'DistFunc/' + 'df_axis' + str(self.count).zfill(4) + '.png')
        plt.clf()
#        plt.show()
        
    def DisplayDistributionFunc_imshow(self):
        #Distribution Funciton display
        plt.imshow(self.d.dist_fn_x_px_electron.data[:,:,0].T)
        plt.colorbar()
        if not os.path.exists(self.savePath + 'DistFunc/'):
            os.makedirs(self.savePath+ 'DistFunc/')
        plt.savefig(self.savePath + 'DistFunc/' + 'df_imshow' + str(self.count).zfill(4) + '.png')
        plt.clf()
   
    def numberDensity_and_Efields(self, smoothBoxSize = 20):
        #Plot the number density and the electric field on top of each other
        #Smooth the number density to see the shape better    
        NumberDensity = self.d.Derived_Number_Density.data
        NDSmooth = self.smooth(NumberDensity, smoothBoxSize)
        
        grid = (self.d.Electric_Field_Ex.grid.data[0][1:] + self.d.Electric_Field_Ex.grid.data[0][:-1]) * 0.5
        grid = grid * 1e6
        
        fig, ax = plt.subplots(nrows = 2, sharex=True)
        ax[0].plot(grid, self.d.Electric_Field_Ex.data, label = 'X')
        ax[0].plot(grid, self.d.Electric_Field_Ey.data , label = 'Y')
        ax[0].plot(grid, self.d.Electric_Field_Ez.data, label = 'Z')
        ax[0].set_xlim([grid[0], grid[-1]])
        ax[0].legend();
        ax[1].plot(grid, NDSmooth)
        ax[1].set_xlim([grid[0], grid[-1]])     

        ax[0].set_ylabel('Electric Field')
        ax[1].set_xlabel('Position (mu m)')
        ax[1].set_ylabel('Number Density')

        if not os.path.exists(self.savePath + 'NumDen_L/'):
            os.makedirs(self.savePath+ 'NumDen_L/')
        plt.savefig(self.savePath + 'NumDen_L/' + 'numAndL' + str(self.count).zfill(4) + '.png')
        plt.clf()
#        plt.show()
        
    def SpectrumFromDistFunction(self, plotting=False):
        self.assignDistFuncGrid()
        self.sumFuncData = self.distFncData.sum(axis=1)
        output = np.c_[self.yGrid_Momentum, self.sumFuncData]
        if plotting:
            plt.clf()
            plt.plot(self.yGrid_Momentum, self.sumFuncData)
            if not os.path.exists(self.savePath + 'DistFunc/'):
                os.makedirs(self.savePath+ 'DistFunc/')
            plt.yscale('log')
            plt.savefig(self.savePath + 'DistFunc/' + 'spectrum' + str(self.count).zfill(4) + '.png')
#        plt.show()
        return output
    
    def time(self):
        return self.d.Header['time']
    
    
def GetSpacedElements(array, numElems):
    out = []
    count = 0
    for a in array:
        if count == numElems - 1:
            count = 0
            out.append(a)
        else:
            count +=1
    return out    
    
if __name__ == "__main__":
#==============================================================================
#     Using the Parser take in the arguemnts to run with
#==============================================================================
    parser = argparse.ArgumentParser()                            
    # Get the folder path, from scratch and then from home.                   
    parser.add_argument("--folder", "-f", type=str, required=True)
    # Plotting options    
    parser.add_argument("--options", "-o", type=str, required=True)

    parser.add_argument("--vmin", "-l", type=str, required=False)
    parser.add_argument("--vmax", "-u", type=str, required=False)
    parser.add_argument("--spacing", "-s", type=int, required=False)    
    
    #Create Folder Paths
    folderPath = str(parser.parse_args().folder)
    scratchPath = '/work/scratch/scarf585/' + folderPath
    savePath = '/home/clfg/scarf585/' + folderPath
    
    #Create plot options list from the arguemnts
    plotOptions = parser.parse_args().options.split('_')
    spacing = parser.parse_args().spacing
    
    if type(spacing) == int:
        print 'Spacing is an int'
        print spacing
    else:
        spacing = 1
    
    numDensLaser = False
    distPlot_imshow = False
    distPlot_axis = False
    for opt in plotOptions:
        if opt  == 'l':
            numDensLaser = True
        if opt  == 'd':
            distPlot_imshow = True
        if opt == 'da':
            distPlot_axis = True

    
#==============================================================================
#     Load all the data that is needed to be analysed
#==============================================================================
    sdf_list, simTimeSteps, inputDeck = FilesInFolder(scratchPath, [0, -4])
    
    momentumEvo = []
    posAndTime = []
#    sdf_spaced = sdf_list
    sdf_spaced = GetSpacedElements(sdf_list, spacing)
    for sdf in sdf_spaced:
        oneD = Load_1D_data(scratchPath + sdf, savePath)
        if numDensLaser:
            oneD.numberDensity_and_Efields()
        if distPlot_imshow:
            oneD.DisplayDistributionFunc_imshow()
        if distPlot_axis:            
            oneD.DisplayDistributionFunc_correctAxis()

        if sdf == sdf_spaced[-1]:
            #Save the last plot as it is the spectrum
            timeSpectrum = oneD.SpectrumFromDistFunction(True)
        else:
            timeSpectrum = oneD.SpectrumFromDistFunction(False)

        momentumEvo.append(timeSpectrum[:,1])
        posAndTime.append([np.average(oneD.xGrid), oneD.time()])
    
    momentumEvo = np.array(momentumEvo)
    posAndTime = np.array(posAndTime)
    
    if not os.path.exists(savePath + 'DistEvo/'):
        os.makedirs(savePath+ 'DistEvo/')    
   
    print ('Saving the arrays to file')
    np.savetxt(savePath + 'DistEvo/xAxis.txt', posAndTime)
    np.savetxt(savePath + 'DistEvo/yMomentum.txt', oneD.yGrid_Momentum)
    np.savetxt(savePath + 'DistEvo/momentumEvo.txt', momentumEvo.T)
    
    print ('Copying input.deck')
    if len(inputDeck) == 1:
        # copy the input deck so there is a record of it
        import shutil
        shutil.copy2(scratchPath+inputDeck[0], savePath)
    
    
    plt.clf()
    plt.pcolormesh(posAndTime[:,0] * 1e6, oneD.yGrid_Momentum, momentumEvo.T, norm = colors.LogNorm())
    plt.colorbar()
    plt.xlabel('Position (mu m)')
    plt.savefig(savePath + 'DistEvo/momentumEvo_POS.png')
    
    plt.clf()
    plt.pcolormesh(posAndTime[:,1] * 1e12, oneD.yGrid_Momentum, momentumEvo.T, norm = colors.LogNorm())
    plt.colorbar()
    plt.xlabel('Time (ps)')
    plt.savefig(savePath + 'DistEvo/momentumEvo_TIME.png')

    

