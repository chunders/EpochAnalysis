#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
       _ 
      /  |     | __  _ __  _
     /   |    /  |_||_|| ||
    /    |   /   |  |\ | ||_
   /____ |__/\ . |  | \|_|\_|
   __________________________ .
   
Created on Thu May 31 10:55:36 2018

@author: chrisunderwood
                                                                       
███████╗██████╗  ██████╗  ██████╗██╗  ██╗                              
██╔════╝██╔══██╗██╔═══██╗██╔════╝██║  ██║                              
█████╗  ██████╔╝██║   ██║██║     ███████║                              
██╔══╝  ██╔═══╝ ██║   ██║██║     ██╔══██║                              
███████╗██║     ╚██████╔╝╚██████╗██║  ██║                              
╚══════╝╚═╝      ╚═════╝  ╚═════╝╚═╝  ╚═╝   


2018_Epoch_distFnc_from_txt.py creates the dist fnc from the text files 
if the scarf plotting failed.
"""



import numpy as np
import os
import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib.colors as colors
import sys
sys.path.insert(0, '/Users/chrisunderwood/Documents/Python/')
import CUnderwood_Functions as func

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

   
def createPlot_dist_evo(allPx_integrated, all_xaxis, yaxis, Time = True):
    
#    plt.imshow(allPx_integrated.T)
#    plt.show()
#    
    plt.close()
    cmap = plt.cm.jet
    cmap.set_under(color='white')
    minDensityToPlot = 1
    
    if Time:
        all_xaxis = all_xaxis * 1e12 #Turn into picoseconds
        plt.pcolormesh(all_xaxis, yaxis,  allPx_integrated.T, norm=colors.LogNorm(), cmap = cmap, vmin = minDensityToPlot)
        xmin = all_xaxis[0]; xmax=all_xaxis[-1];
        plt.xlabel("Time (ps)")
    else:
        all_xaxis = all_xaxis * 1e6 #Turn into mu m
        plt.pcolormesh(all_xaxis, yaxis,  allPx_integrated.T, norm=colors.LogNorm(), cmap = cmap, vmin = minDensityToPlot)
        xmin = all_xaxis[0]; xmax=all_xaxis[-1];
        plt.xlabel(r"Distance $(\mu m)$")
    ymin= yaxis[0] ; 
    ymax=yaxis[- 1]
    xmin = all_xaxis[0]; 
    xmax = all_xaxis[-1]
    cbar = plt.colorbar()
    cbar.set_label("Density (nparticles/cell)", rotation=270)
#    plt.axis([xmin,xmax,None, ymax])
    plt.ylabel(r"Momentum ($kg.ms^{-1}$)")
    plt.show()




def loadFiles_From_Root_Folder(rootDir):
    fp = rootDir + 'Dist_Evo/'
    allPx_integrated = np.loadtxt(fp +'px_vs_t.txt')
    posAxis = np.loadtxt(fp + 'xaxis_x.txt')
    timeAxis = np.loadtxt(fp + 'xaxis_t.txt')
    yaxis = np.loadtxt(fp + 'yaxis_p_eV.txt') 
    return allPx_integrated, posAxis, timeAxis, yaxis

#def outputElectronSpectrum(yaxis, allPx_integrated, ylimMax = 1e10):
#    zeroIndex = func.nearposn(yaxis, 0.)
#    finalSpecPx = allPx_integrated[-1,:]
#    xEJ = yaxis[zeroIndex:]
#    ySpectrum = finalSpecPx[zeroIndex:]
#    xMeV = func.E_Joules_to_MeV(xEJ)
#    plt.plot(xEJ, ySpectrum)
#    plt.ylim([1, ylimMax])
#    plt.yscale('log')
#    


def createElectronSpectrum(rootDir):
    fig = plt.figure(figsize=(4,4))
    func.setup_figure_fontAttributes(size = 16)
    spec = np.loadtxt(rootDir + 'Dist_evo/Electron_Spectrum.txt')
    xPx = spec[:,0]
    zeroIndex = func.nearposn(xPx, 0.)
    eMev = func.momentumToMeV(xPx)[zeroIndex:]
    intensity = spec[:,1][zeroIndex:]
    plt.plot(eMev, intensity)
    plt.yscale('log')
    plt.ylim([1, None])
    plt.ylabel('Electrons')
    plt.xlabel('Energy (MeV)')
    func.saveFigure(rootDir + 'Dist_evo/Electron Spectrum.png')
    
    OneMeVandAboveIndex = func.nearposn(eMev, 1)
    nosE_above_1MeV = np.sum(intensity[OneMeVandAboveIndex:])
    print "Number of Electrons above 1 MeV {:.2e}".format(nosE_above_1MeV)
    print "Charge {:.2e} C".format(nosE_above_1MeV * 1.6e-19)
    print "Charge {:.2e} pC".format(nosE_above_1MeV * 1.6e-19 * 1e12)
    plt.vlines(eMev[OneMeVandAboveIndex], 0, 1e10)
    
    
gdrive = '/Volumes/GoogleDrive/My Drive/'
#gdrive += '2018_Epoch_1.5mmGasJet/1127_initial_attempt/fs6_d1e25_f80_15mJ/'
gdrive += '2018_Epoch_1.5mmGasJet/1127_initial_attempt/SG_fs6_d1e25_15mJ/'



logPlot = False
plot_MeV = False
##sns.set_palette(sns.color_palette("Set1", len(folderNames)))
##sns.set_context("talk")
##sns.set_style('darkgrid')

allPx_integrated, posAxis, timeAxis, yaxis = loadFiles_From_Root_Folder(gdrive)

#createPlot_dist_evo(allPx_integrated, posAxis, yaxis)
createElectronSpectrum(gdrive)