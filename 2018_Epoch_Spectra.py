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

To compare the outputted Electron spectrums,
as part of a parameter scan
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



def subplotPerSpectra(data, Crop):
    sns.set_palette(sns.color_palette("Set1", len(folderNames)))
    sns.set_context("talk")
    sns.set_style('darkgrid')

    fig, axes = plt.subplots(nrows = len(data), sharex = True, figsize = (7,8))

    for d, names, ax in zip(data, folderNames, axes):
            yLims = [1e50, 0]

            px = d[:,0]
            Energy_J = (px ** 2) / (2 * 9.11e-31)
            
            Energy_eV = Energy_J / 1.6e-19
            Energy_MeV = Energy_eV * 1e-6
            xlow = nearposn(Energy_MeV, Crop[0])
            xhigh = nearposn(Energy_MeV, Crop[1])
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
            ax.plot(xAxis, intensity)
            ax.set_title('Blade Translation '  + names[1:] + 'mm')
            ax.set_ylim(yLims)
#            ax.set_ylabel('Intensity (# of electrons)')
            ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useOffset=False)

    if plot_MeV:
        plt.xlabel('Electron Energy (MeV)')
    else:
        plt.xlabel('Electron Energy (J)')
#    plt.ylabel('Intensity (# of electrons)')
    fig.text(0.02, 0.5, 'Intensity (# of electrons)', ha='center', va='center', rotation='vertical')
    #==============================================================================
    # Apply the plotting limits
    #==============================================================================
    #plt.xlim([-1e-14, 1e-13])
    #plt.yscale('log')
    #
#    if logPlot:
#        plt.ylim([yLims[1]/1e5, yLims[1]])
#        plt.yscale('log')
#    else:    
#        plt.ylim(yLims)
#    
    plt.xlim([xAxis[xlow],xAxis[xhigh]])
    
    plt.legend()
    
    

def createPlotOfAll_e_spectra(folderPaths, folderNames, Crop_X, Crop_Y = False):
    sns.set_palette(sns.color_palette("Set1", len(folderNames)))
    sns.set_context("talk")
    sns.set_style('darkgrid')
    yLims = [1e50, 0]
    data = []
    plt.figure(figsize = (10,7))
    for fp, names in zip(folderPaths, folderNames):
        fp += 'Electron_Spectrum.txt'
        
        try:
            #Assuming that the first row is currently px
            d = np.loadtxt(fp)
            data.append(d)
            px = d[:,0]
            Energy_J = (px ** 2) / (2 * 9.11e-31)
            
            Energy_eV = Energy_J / 1.6e-19
            Energy_MeV = Energy_eV * 1e-6
            xlow = nearposn(Energy_MeV, Crop_X[0])
            xhigh = nearposn(Energy_MeV, Crop_X[1])
        #    print xlow, xhigh
            #    xlow = 50; xhigh = 400
            intensity = d[:,1]
            if not Crop_Y:
                cropI = intensity[xlow:xhigh]
                if cropI.min() < yLims[0]:
                    yLims[0] = cropI.min()
                if cropI.max() > yLims[1]:
                    yLims[1] = cropI.max()
            else:
                yLims = Crop_Y
                
        #    print fp
            if plot_MeV:
                xAxis = Energy_MeV
            else:
                xAxis = Energy_J
            plt.plot(xAxis, intensity, label = names)
        except:
            print 'Error Reading File'
            print '    ' + fp
    
    if plot_MeV:
        plt.xlabel('Electron Energy (MeV)')
    else:
        plt.xlabel('Electron Energy (J)')
    plt.ylabel('Intensity (# of electrons)')
    
    #==============================================================================
    # Apply the plotting limits
    #==============================================================================
    #plt.xlim([-1e-14, 1e-13])
    #plt.yscale('log')
    #
    if logPlot:
        plt.ylim([yLims[1]/1e5, yLims[1]])
        plt.yscale('log')
    else:    
        plt.ylim(yLims)
    
    plt.xlim([xAxis[xlow],xAxis[xhigh]])
    
    plt.legend()
    print 'Crop corresponds to: ', [xAxis[xlow],xAxis[xhigh]], ' MeV'
    print 'Range of inputed data is: ', Energy_MeV[0], Energy_MeV[-1]
    return data

hdrive = '/Volumes/CIDU_passport/2018_Epoch_vega_1/'
gdrive = '/Volumes/GoogleDrive/My Drive/2018_Epoch_vega_1/'
#hdrive += '0601_Gaus_for_wavebreak/'
#fileSplice = [8,None]

#hdrive += '0607_Intensity_Scan/'
#fileSplice = [1,-11]

#hdrive += '0612_profileScan/'
#fileSplice = [2,None]


#hdrive = gdrive + '0711_highRes_selfInjection/'
#fileSplice = [-4,None]


#hdrive = gdrive + '0721_HR_Jump/'
#fileSplice = [-4,None]


hdrive = hdrive + '1010_SlurmJob/'
fileSplice = [10,12]


folderPaths, folderNames = listFolders(hdrive)
logPlot = False
plot_MeV = True
#==============================================================================
# Search for the set of folders to look at!
#==============================================================================
starts = 'SG_t8e23'
#starts = ''

fins = 'FS'
#Index_to_save = [i for i in xrange(len(folderNames)) if folderNames[i].endswith(fins)]
Index_to_save = [i for i in xrange(len(folderNames)) if folderNames[i].startswith(starts)]
#Index_to_save = [i for i in xrange(len(folderNames)) if folderNames[i].startswith(starts) and folderNames[i].endswith('23')]

#Modify the both arrays to just be the ones of interest
folderPaths = folderPaths[Index_to_save]
folderNames = folderNames[Index_to_save]
print folderNames

#==============================================================================
# Crop the axis to the interesting data
#==============================================================================
Energy_Crop = [1, 5]    # In MeV
sIntensityCrop = [0, 0.5e8]

#==============================================================================
# Slice name for number to sort by
#==============================================================================
Num = []
for f in folderNames:
    Num.append(float(f[fileSplice[0]:fileSplice[1]]))
print Num
sort =  sorted(zip(Num, folderNames, folderPaths))
folderNames = [x[1] for x in sort]
folderPaths = [x[2] for x in sort]

print 'Sorted'
print folderNames

#folderNames = folderNames[:-1] 

data = createPlotOfAll_e_spectra(folderPaths, folderNames, Energy_Crop, IntensityCrop)
plt.savefig(hdrive + 'Electron_spectrum.png')
plt.show()

#data = data[:4]
subplotPerSpectra(data, Energy_Crop)
plt.tight_layout()
plt.savefig(hdrive + 'Electron_spectrums_in_subplot.png', dpi = 300)





