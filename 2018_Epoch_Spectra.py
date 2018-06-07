#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
       _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
                                                                          
███████╗██████╗  ██████╗  ██████╗██╗  ██╗                              
██╔════╝██╔══██╗██╔═══██╗██╔════╝██║  ██║                              
█████╗  ██████╔╝██║   ██║██║     ███████║                              
██╔══╝  ██╔═══╝ ██║   ██║██║     ██╔══██║                              
███████╗██║     ╚██████╔╝╚██████╗██║  ██║                              
╚══════╝╚═╝      ╚═════╝  ╚═════╝╚═╝  ╚═╝   
Created on Wed May 30 15:34:05 2018

@author: chrisunderwood

To compare the outputted spectra, as part of a parameter scan
"""


import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns


#==============================================================================
# A function that replicates os.walk with a max depth level
#==============================================================================
def walklevel(some_dir, level=1):
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

#==============================================================================
# Creates a list of the folders of interest
#==============================================================================
def listFolders(mainDir):
    listSubFolders =  [x[0] for x in walklevel(mainDir)][1:]
    folderNames = []    
    #Modify so useable path
    for i in range(len(listSubFolders)):
        
        listSubFolders[i] += '/'
        #Add the folder that I am looking into here too!
        listSubFolders[i] += 'Dist_evo/'

        folderNames.append(listSubFolders[i].split('/')[-3])
    listSubFolders = np.array(listSubFolders)
    folderNames = np.array(folderNames)
    return listSubFolders, folderNames

def nearposn(array,value):
	#Find array position of value
    posn = (abs(array-value)).argmin()
    return posn



def createPlotOfAll_e_spectra(folderPaths, folderNames, xCrop_px):
    sns.set_palette(sns.color_palette("Set1", len(folderNames)))
    sns.set_context("talk")
    sns.set_style('darkgrid')
    yLims = [1e50, 0]

    plt.figure(figsize = (10,7))
    for fp, names in zip(folderPaths, folderNames):
        fp += 'Electron_Spectrum.txt'
        
        #Assuming that the first row is currently px
        d = np.loadtxt(fp)
        px = d[:,0]
        Energy_J = (px ** 2) / (2 * 9.11e-31)
        
        Energy_eV = Energy_J / 1.6e-19
        Energy_MeV = Energy_eV * 1e-6
        xlow = nearposn(px, xCrop_px[0])
        xhigh = nearposn(px, xCrop_px[1])
    #    print xlow, xhigh
        #    xlow = 50; xhigh = 400
        intensity = d[:,1]
        cropI = intensity[xlow:xhigh]
        if cropI.min() < yLims[0]:
            yLims[0] = cropI.min()
        if cropI.max() > yLims[1]:
            yLims[1] = cropI.max()
        
    #    print fp
        if plot_MeV:
            xAxis = Energy_MeV
        else:
            xAxis = Energy_J
        plt.plot(xAxis, intensity, label = names)
    
    if plot_MeV:
        plt.xlabel('(MeV)')
    else:
        plt.xlabel('(J)')
    plt.ylabel('Intensity (arb. units)')
    
    #==============================================================================
    # Apply the plotting limits
    #==============================================================================
    #plt.xlim([-1e-14, 1e-13])
    #plt.yscale('log')
    #
    if logPlot:
        plt.ylim([yLims[1]/1e3, yLims[1]])
        plt.yscale('log')
    else:    
        plt.ylim(yLims)
        plt.xlim([xAxis[xlow],xAxis[xhigh]])
    
    plt.legend()
    plt.show()
    print 'Crop corresponds to: ', [xAxis[xlow],xAxis[xhigh]], ' MeV'
    print 'Range of inputed data is: ', Energy_MeV[0], Energy_MeV[-1]

hdrive = '/Volumes/CIDU_passport/2018_Epoch_vega_1/'
#hdrive += '0601_Gaus_for_wavebreak/'
hdrive += '0604_JumpLR/'

folderPaths, folderNames = listFolders(hdrive)
logPlot = False
plot_MeV = True
#==============================================================================
# Search for the set of folders to look at!
#==============================================================================

starts = 'SG'
starts = 'J'

fins = 'FS'
#Index_to_save = [i for i in xrange(len(folderNames)) if folderNames[i].endswith(fins)]
Index_to_save = [i for i in xrange(len(folderNames)) if folderNames[i].startswith(starts)]
#Index_to_save = [i for i in xrange(len(folderNames)) if folderNames[i].startswith(starts) and folderNames[i].endswith('23')]

#Modify the both arrays to just be the ones of interest
folderPaths = folderPaths[Index_to_save]
folderNames = folderNames[Index_to_save]
print folderNames

#xCrop_px = [0.15e-20, 1.0e-20]
xCrop_px = [0.15e-20, 8e-21]     # The top range inputted into the file


createPlotOfAll_e_spectra(folderPaths, folderNames, xCrop_px)







