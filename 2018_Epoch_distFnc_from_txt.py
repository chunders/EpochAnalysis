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
    ymin= yaxis[0] ; ymax=yaxis[- 1]
    xmin = all_xaxis[0]; xmax = all_xaxis[-1]
    cbar = plt.colorbar()
    cbar.set_label("Density (nparticles/cell)", rotation=270)
    plt.axis([xmin,xmax,None, ymax])
    plt.ylabel(r"Momentum ($kg.ms^{-1}$)")
    plt.show()

hdrive = '/Volumes/CIDU_passport/2018_Epoch_vega_1/'
#hdrive += 'DensScan/'
hdrive += '1114/'
#hdrive += '20muFS_testing_selfInjection/dens/'


folderPaths, folderNames = listFolders(hdrive)
logPlot = True
plot_MeV = True
#==============================================================================
# Search for the set of folders to look at!
#==============================================================================
#Index_to_save = [i for i in xrange(len(folderNames)) if folderNames[i].endswith('FS')]
#Index_to_save = [i for i in xrange(len(folderNames)) if folderNames[i].startswith('rho')]

#folderPaths = folderPaths[Index_to_save]
#folderNames = folderNames[Index_to_save]
print folderNames

#sns.set_palette(sns.color_palette("Set1", len(folderNames)))
#sns.set_context("talk")
#sns.set_style('darkgrid')

#plt.figure(figsize = (10,7))
for fp, names in zip(folderPaths, folderNames):
    try:
        allPx_integrated = np.loadtxt(fp +'px_vs_t.txt')
        posAxis = np.loadtxt(fp + 'xaxis_x.txt')
        timeAxis = np.loadtxt(fp + 'xaxis_t.txt')
        yaxis = np.loadtxt(fp + 'yaxis_p_eV.txt')
        
        print fp.split('/')[-3]
        createPlot_dist_evo(allPx_integrated, posAxis, yaxis)
    except:
        print 'Error loading data from'
        print fp.split('/')[-3] + '\n'