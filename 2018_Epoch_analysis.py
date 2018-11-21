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

Created on Thu May 10 09:58:18 2018

@author: chrisunderwood

A file to be ran on scarf

A new file to do analysis on the epoch files
    reads in the folder path by parsing the option in.
    creates distfncevolution for whole simulation
    creates final electron spectrum
    creates number density ploy with the laser pulse ontop
    
    
There are several cmd line options when running the file:
"--folder", "-f", type=str, required=True
    Folder path of simulation, and then duplicate dir in home folder
"--vmin", "-l", type=str, required=False
    min density on num dens plot
"--vmax", "-u", type=str, required=False
    max dens on num dens plot
        Both vmin and vmax are required or not at all currently
"--append", "-a", type=str, required=False
    append onto previous files
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


def savedFiles(DirectoryPath):
    files = os.listdir(DirectoryPath + 'NumberDensity_with_L/')
    shots = []
    for i in files:
        if not i.startswith('.') and i.endswith('.png'):
            shots.append(i)
    # Sort
    timeStep = []
    for i in range(len(shots)):
        print shots[i], '\t',shots[i][14:-4]
        timeStep.append(int(shots[i][14:-4]))
    
    timeStep = np.asarray(timeStep)
    sorted_points = sorted(zip(timeStep, shots))
    timeStep = [point[0] for point in sorted_points]
    png_list = [point[1] for point in sorted_points]
        
    return png_list, timeStep

def createTicks(x, nos, exp = False):
#==============================================================================
#     function creates tick labels at positions, so imshow can be used
#     this massively increases the run speed over pcolor
#==============================================================================
    xlen = len(x)
    x_sep = int(xlen / nos)
    xpos = []
    xval = []
    start = 0
    while(start < xlen):
        xpos.append(start)
        if exp:
            xval.append("{0:.2e}".format(x[start]))
        else:
           if x[start] > 100:
               xval.append("{0:.2e}".format(x[start]))
           else:
               xval.append("{0:.2f}".format(x[start]))
        start += x_sep
    
    return xpos, xval   
 
def normaliseArr(arr):
    arr = np.array(arr)
    arr = arr - arr.min()
    return arr / arr.max()
    
def indivual_numDens(inData,time, inpath, savepath, vMin, vMax):
    plt.close()
    plt.figure(figsize=(8,7)) 
    numDens = inData.Derived_Number_Density.data
    grid = inData.Grid_Grid_mid.data
    x = grid[0] / 1e-3
    y = grid[1] / 1e-3
    Actual_time = inData.Header.values()[9]
    
    xpos, xval = createTicks(x,6)
    ypos, yval = createTicks(y,8)        
        
#    if vMin is not None and vMax is not None:
#        plt.imshow(numDens.T, aspect = 'auto', vmin = vMin, vmax = vMax)
#    else:
#        plt.imshow(numDens.T, aspect = 'auto')
#    plt.colorbar()
    
    #Sum in each direction for lineouts
    sumX = []
    for im in numDens:
        sumX.append(sum(im))
    sumY = []
    for im in numDens.T:
        sumY.append(sum(im))
#    print 'Len of x, y sum: ', len(sumX), ' ', len(sumY)

#==============================================================================
# Create fig with subplots in 
#==============================================================================
    fig = plt.figure(figsize=(8,6))
    # 3 Plots with one major one and two extra ones for each axes.
    gs = gridspec.GridSpec(4, 4, height_ratios=(1,1,1,1), width_ratios=(0.5,1,1,1))
    gs.update(wspace=0.025, hspace=0.025)
    
#    Create all axis, including an additional one for the cbar
    ax1 = plt.subplot(gs[0:3, 1:-1])             # Image
    ax1.axis('off')
    ax2 = plt.subplot(gs[0:3, 0] ) # right hand side plot
    ax3 = plt.subplot(gs[-1, 1:-1] ) # below plot
    
    cax4 = fig.add_axes([0.7, 0.35, 0.05, 0.5])

#    Make the axis look as I want them too
#    Modify the ticks so they are easily visible
    xticks = ticker.MaxNLocator(5)
    ax2.yaxis.set_major_locator(xticks)

    xticks = ticker.MaxNLocator(3)
    ax2.xaxis.set_major_locator(xticks)
    ax3.yaxis.tick_right()

    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0),useOffset=False)
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0,0),useOffset=False)

    
    im = ax1.imshow(numDens.T, aspect = 'auto')
    ax2.plot(sumY, y )
    ax3.plot(x, sumX)
    

#    Set the axis limits for the plots
    image_shape = np.shape(numDens.T)
    ax1.set_xlim(0,image_shape[1])
    ax1.set_ylim(0, image_shape[0])
    ax2.set_ylim(y[0], y[-1])
    ax3.set_xlim(x[0], x[-1])
    

    
    ax3.set_xlabel(r'$(mm)$')
    ax2.set_ylabel(r'$(mm)$')
    plt.colorbar(im, cax = cax4)
    
    plt.suptitle('Simulation Time: {0:.4f} (ps)'.format(Actual_time * 1e12))
        
    #Create folder to save into
    if not os.path.exists(savePath + 'NumberDensity/'):
        os.makedirs(savePath + 'NumberDensity/')
    
    plt.savefig(savepath + 'NumberDensity/' +'nd_' + str(time) + '.png', dpi = 150)
#    plt.show()
     
def indivual_numDens_with_laser(inData, intTime, inpath, savepath, cmap1, vMin, vMax):
    plt.close()
    plt.figure(figsize=(10,7)) 
    numDens = inData.Derived_Number_Density.data
    ez = inData.Electric_Field_Ez.data
    grid = inData.Grid_Grid_mid.data
    x = grid[0] / 1e-6
    y = grid[1] / 1e-6
    Actual_time = inData.Header.values()[9]
    
    thres2 = np.median(abs(ez)) * 10
    mask = abs(ez) > thres2
    xpos, xval = createTicks(x,6)
    ypos, yval = createTicks(y,8)        
        
    if vMin is not None and vMax is not None:
        plt.imshow(numDens.T, aspect = 'auto', vmin = vMin, vmax = vMax)
    else:
        minVal_cm = numDens.T.max() * 0.001
        if minVal_cm < numDens.T.min():
            minVal_cm = numDens.T.min()
        plt.imshow(numDens.T, aspect = 'auto', vmin = minVal_cm)
    cbar = plt.colorbar()
    cbar.set_label(inData.Derived_Number_Density.units)
    
    eField_masked = abs(ez) * mask
    plt.imshow(eField_masked.T, cmap=cmap1, aspect = 'auto')
    cbar = plt.colorbar()
    cbar.set_label(inData.Electric_Field_Ez.units)
    plt.xticks(xpos,xval, rotation=-90)
    plt.yticks(ypos,yval)
    plt.xlabel(r'$(\mu m)$')
    plt.ylabel(r'$(\mu m)$')
    plt.title('Simulation Time: {0:.4f} (ps)'.format(Actual_time * 1e12))
    plt.tight_layout()

        #Create folder to save into
    if not os.path.exists(savePath + 'NumberDensity_with_L/'):
        os.makedirs(savePath + 'NumberDensity_with_L/')

    plt.savefig(savepath + 'NumberDensity_with_L/' +'nd_with_laser_' + str(intTime) + '.png', dpi = 150)

def indivual_distF(inData, intTime, inpath, savepath):
    plt.close()
    x = inData.dist_fn_x_px_electron.grid.data[0] * 1e6
    y = inData.dist_fn_x_px_electron.grid.data[1]
    cmap = plt.cm.gist_rainbow
#    cmap.set_under(color = 'white')
    print inData.dist_fn_x_px_electron.data.min(), inData.dist_fn_x_px_electron.data.max()
    plt.imshow(inData.dist_fn_x_px_electron.data.T, aspect = 'auto', cmap = cmap) # , vmin = 1e-27 )
    plt.colorbar()
    xmin = x[0]; xmax=x[- 1]; ymin= y[0] ; ymax=y[- 1]
    xpos, xval = createTicks(x,5)
    ypos, yval = createTicks(y,6, True)
    plt.xticks(xpos,xval)
    plt.yticks(ypos,yval)
    
    plt.xlabel(r'X($\mu m$)')
    plt.ylabel(r'$P_x$ ($kg m s^{-1}$)')
    Actual_time = inData.Header.values()[9] * 1e12


    plt.title (r'$x - P_x$ @ t = {0:.5g} (ps)'.format(Actual_time))
    plt.axis([xmin,xmax,ymin, ymax])
    
    #Create folder to save into
    if not os.path.exists(savePath + 'DistFnc/'):
        os.makedirs(savePath + 'DistFnc/')

    plt.savefig(savepath + 'DistFnc/' + 'distFnc' + str(intTime) + '.png', dpi = 150)

def createPlot_dist_evo(allPx_integrated, all_xaxis, yaxis, savepath, xAxis = 0):
    plt.close()
    cmap = plt.cm.jet
    cmap.set_under(color='white')
    minDensityToPlot = 1e4
    maxDensityToPlot = 5e11

    if xAxis == 1:
        all_xaxis = all_xaxis * 1e12 #Turn into picoseconds
        plt.pcolormesh(all_xaxis, yaxis,  allPx_integrated.T, 
                       norm=colors.LogNorm(), cmap = cmap, 
                       vmin = minDensityToPlot, vmax = maxDensityToPlot)
        xmin = all_xaxis[0]; xmax=all_xaxis[-1];
        plt.xlabel("Time (ps)")
    elif xAxis == 2:
        all_xaxis = all_xaxis * 1e6 #Turn into mu m
        plt.pcolormesh(all_xaxis, yaxis,  allPx_integrated.T, 
                       norm=colors.LogNorm(), cmap = cmap, vmin = minDensityToPlot)
        xmin = all_xaxis[0]; xmax=all_xaxis[-1];
        plt.xlabel("Distance (um)")
    else:
        plt.pcolormesh(all_xaxis, yaxis,  allPx_integrated.T, 
                       norm=colors.LogNorm(), cmap = cmap, vmin = minDensityToPlot)
        xmin = all_xaxis[0]; xmax=all_xaxis[-1];
        plt.xlabel("SDF Number")
    ymin= yaxis[0] ; ymax=yaxis[- 1]
    xmin = all_xaxis[0]; xmax = all_xaxis[-1]
    cbar = plt.colorbar()
    cbar.set_label("Density (nparticles/cell)", rotation=270)
    plt.axis([xmin,xmax,ymin, ymax])
    plt.ylabel("Momentum (kg.ms^-1)")
    
    if xAxis == 1:
        plt.savefig(savepath + 'Dist_evo/' + folderPath.split('/')[-2] + '_DistPx_Vs_Time.png', dpi = 300)
    elif xAxis == 2:
        plt.savefig(savepath + 'Dist_evo/' + folderPath.split('/')[-2] + '_DistPx_Vs_Distance.png', dpi = 300)
    elif xAxis == 0:
        plt.savefig(savepath + 'Dist_evo/' + folderPath.split('/')[-2] + '_DistPx_Vs_SDF_ts.png', dpi = 300)
    
    print 'Distribution plot min, max and ave: '
    print allPx_integrated.min(), allPx_integrated.max(), np.average(allPx_integrated)

    #plt.show()
    plt.close()
    intensity = allPx_integrated[-1,:]
#    intensity = (intensity - min(intensity)) / max(intensity)
    plt.plot(yaxis, intensity)
    np.savetxt(savepath + 'Dist_evo/' + 'Electron_Spectrum.txt', np.c_[yaxis, intensity])
    plt.yscale('log')
#    xmin = px_GeV[0]; xmax = 1.0; ymin = 0;  ymax = 1.;
#    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel(r"Energy ()")
#    plt.ylabel("Intensity (Normalised)")
    plt.ylabel("Intensity ()")
    plt.title("Electron Bunch Spectrum")
    plt.savefig(savepath + 'Dist_evo/' + 'Electron Spectrum.png', dpi = 300)
    




def distFunc_pxX_plot(filelist, timesteps, inpath, savepath):
#==============================================================================
#     Create  a dist function for all timesteps
#==============================================================================
    for f, time in zip(filelist, timesteps):
        inData = sh.getdata(inpath + f)
        indivual_distF(inData, time, inpath, savepath)
        
def numberDens(filelist, timesteps, inpath, savepath, vMin, vMax):
#==============================================================================
#     Creates the plot of the number density, 
#==============================================================================
    #For each time step create plot and then save that with its simstep position
    for f, time in zip(filelist, timesteps):
        inData = sh.getdata(inpath + f)
        indivual_numDens(inData,time, inpath, savepath, vMin, vMax)
  
def indivual_eField(inData,time, inpath, savepath):
    plt.close()
    plt.figure(figsize=(8,7)) 
    ez = inData.Electric_Field_Ez.data
    grid = inData.Grid_Grid_mid.data
    x = grid[0] * 1e6
    y = grid[1] * 1e6
    Actual_time = inData.Header.values()[9]    
    
    
    plt.pcolormesh(x, y, ez.T)
    cbar = plt.colorbar()
    cbar.set_label(inData.Electric_Field_Ez.units)
#    plt.xticks(xpos,xval, rotation=-90)
#    plt.yticks(ypos,yval)
    plt.xlabel('Distance (um)')
    plt.ylabel('Distance(um)')
    plt.title('Simulation Time: {0:.4f} (ps)'.format(Actual_time * 1e12))
    plt.tight_layout()

        #Create folder to save into
    if not os.path.exists(savePath + 'efieldZ/'):
        os.makedirs(savePath + 'efieldZ/')

    plt.savefig(savepath + 'efieldZ/' +'laser_' + str(time) + '.png', dpi = 150)

    
    
def electricField(filelist, timesteps, inpath, savepath):
    for f, time in zip(filelist, timesteps):
        inData = sh.getdata(inpath + f)
        indivual_eField(inData,time, inpath, savepath)
        
def gridToEgrid(grid):
    Egrid = np.zeros((2,1000))
    Egrid[0]= ((grid[0] **2) / 9.11e-31) * np.sign(grid[0]) / 1.6e-19
    Egrid[1]= ((grid[1] **2) / 9.11e-31) * np.sign(grid[1]) / 1.6e-19
    Egrid = Egrid * 1e-6
    return Egrid    

def create_pxpy_plot(inData, time, inpath, savepath):
    plt.clf()
    
    #Create folder to save into
    if not os.path.exists(savePath + 'pxpy/'):
        os.makedirs(savePath + 'pxpy/')
    Actual_time = inData.Header.values()[9]
    
    grid = inData.dist_fn_px_py_electron.grid.data
    Egrid = gridToEgrid(grid)
    
#    # Create the divergence vs counts plot
#    divergence = []
#    energy = []
#    counts = [] 
#    for i in range(len(grid[0])):
#        for j in range(len(grid[1])):
#            divergence.append(abs(grid[0][i]/grid[1][j]))
#            energy.append(grid[0][i]) # (grid[0][i]**2 + grid[1][j]**2)**0.5)
#            counts.append(inData.dist_fn_px_py_electron.data.T[i][j])
#    divergence = np.array(divergence)
#    energy = ((np.array(energy) ** 2) / (2*m_e))/q_e * 1e-6 # Energy is in MeV
#    counts = np.array(counts)
#    print len(divergence), len(energy), len(counts)
#    print energy.min(), energy.max()
#    print divergence.min(), divergence.max()
#    
#    EnergyHist = np.linspace(energy.min(), energy.max(), 1000)
#    DivHist = np.linspace(divergence.min(), 5, 10000)
#    CountHist = np.zeros((len(EnergyHist), len(DivHist)))
#    
#    for i in range(len(counts)):
#        #See which energy bin   
#        e = 0
#        while( energy[i] > EnergyHist[e] and energy[i] < EnergyHist[e+1] and e < len(EnergyHist)):
#            e += 1
#        d = 0
#        while( divergence[i] > DivHist[d] and divergence[i] < DivHist[d+1] and d < len(DivHist)):
#            d += 1
#        CountHist[e][d] +=counts[i]
#            
#    print 'Max count in histogram', CountHist.max()        
##    plt.imshow(CountHist,aspect = 'auto')                
#    plt.imshow(CountHist, norm=colors.LogNorm(), vmin = 1e4, aspect = 'auto')            
##    plt.pcolormesh(EnergyHist, DivHist, CountHist, norm=colors.LogNorm(), vmin = 1e1)
#    plt.colorbar()
#    plt.title('Time {:.3f}(ps)'.format(Actual_time*1e12))
#    plt.xlabel('Energy (MeV)')
#    plt.ylabel('Divergence (Theta)')
##    plt.ylim([0, 1])
#    plt.savefig(savepath + 'pxpy/' + str(time) + 'divVsE.png', dpi = 150)

    #New Idea - get divergence angle per energy slice
    
    Energy_vs_div = []
    pxVals = grid[0]
    pyVals = grid[1]
    rowError = 0
    for px, row in zip(pxVals, inData.dist_fn_px_py_electron.data):
        divergence = abs(np.arctan(pyVals / px))
        
#        print  'div',divergence, 'row', row                 
#        plt.plot(divergence, row)
#        print row.min(), row.max(), row[len(row)/2], np.average(row)
#        print np.average(inData.dist_fn_px_py_electron.data)
        if np.average(row) > 0.:
            Energy_vs_div.append([px, np.average(divergence, weights = abs(row))])
        else:
            rowError +=1
#            print 'Row Error', rowError
            Energy_vs_div.append([px, 0])
#    plt.yscale('log')
#    plt.savefig(savepath + 'pxpy/' + str(time) + 'divVsE.png', dpi = 150)     
#    plt.clf()
    
    Energy_vs_div = np.array(Energy_vs_div)
#    print np.shape(Energy_vs_div)
    fig, ax = plt.subplots(nrows = 2, sharex  = True )
    ax[0].plot(Energy_vs_div[:,0], Energy_vs_div[:,1], '.-')
    ax[0].set_ylabel('Average Divergence (rads)')
    ax[1].plot(pxVals, inData.dist_fn_px_py_electron.data.sum(axis=1), '.-')
    ax[1].set_ylabel('Number of Electrons')    
    ax[1].set_xlabel('Momentum (kgms-1)')
    ax[1].set_ylim([0, 2e14])
    plt.tight_layout()
    
    
    np.savetxt(savepath + 'pxpy/' + str(time) + savepath.split('/')[-2] +'momentum_div.txt', np.c_[pxVals, inData.dist_fn_px_py_electron.data.sum(axis=1),Energy_vs_div[:,1]])
    plt.savefig(savepath + 'pxpy/' + str(time) + savepath.split('/')[-2] + 'Energy_vs_div.png', dpi = 150)       

    
    
    
    plt.clf()
    plt.pcolormesh(Egrid[0], Egrid[1], inData.dist_fn_px_py_electron.data.T,
                   norm=colors.LogNorm(), vmin = 1e4)
    plt.colorbar()
    plt.xlabel('Energy Beam Axis(MeV)')
    plt.ylabel('Energy Divergence (MeV)')   
    plt.title('Time {:.3f}(ps)'.format(Actual_time*1e12))
    plt.savefig(savepath + 'pxpy/' + str(time) + 'pxpy.png', dpi = 150)

        
def pxpySpectrum(sdf_list, simTimeSteps, scratchPath, savePath):
    for f, time in zip(sdf_list, simTimeSteps):
        inData = sh.getdata(scratchPath + f)
        create_pxpy_plot(inData,time, scratchPath, savePath)
    
        
    
def numberDens_with_laser_pulse(filelist, timesteps, inpath, savepath, vMin, vMax):
#==============================================================================
#     Creates the plot of the number density, with the laser pulse
#     thresholded to be greater than 10x the median value laid over the top
#==============================================================================
    #Creating the color map with transparency
    cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['orange','red'],256)
    cmap1._init() # create the _lut array, with rgba values
    alphas = np.linspace(0, 1.0, cmap1.N+3)    # create your alpha array and fill the colormap with them. here it is progressive, but you can create whathever you want
    cmap1._lut[:,-1] = alphas    
    
    #For each time step create plot and then save that with its simstep position
    for f, time in zip(filelist, timesteps):
        inData = sh.getdata(inpath + f)
        indivual_numDens_with_laser(inData,time, inpath, savepath, cmap1, vMin, vMax)
  

def numD_and_Dist_for_all_time(filelist, timesteps, inpath, savepath, vMin, vMax):
    #Creating the color map with transparency
    cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['orange','red'],256)
    cmap1._init() # create the _lut array, with rgba values
    alphas = np.linspace(0, 1.0, cmap1.N+3)    # create your alpha array and fill the colormap with them. here it is progressive, but you can create whathever you want
    cmap1._lut[:,-1] = alphas    
    
    #For each time step create plot and then save that with its simstep position
    for f, time in zip(filelist, timesteps):
        inData = sh.getdata(inpath + f)
        indivual_numDens_with_laser(inData,time, inpath, savepath, cmap1, vMin, vMax) 
        indivual_distF(inData, time, inpath, savepath)

def densityVsTime(filelist, inpath, savepath):
#==============================================================================
#     Convert the 2D arrays of the distfunc into 1D for each file by histogramming
#     the different rows into the px values.
#     This then forms a 2D array for the whole simulation showing the evolution
#==============================================================================   
    plt.close()
    all_N_integrated = []
    all_Pos = []
    for f in filelist:
        inData = sh.getdata(inpath + f)
        try:
            all_N_integrated.append(np.average(inData.Derived_Number_Density.data))
            all_Pos.append(np.average([inData.dist_fn_x_px_electron.grid.data[0][0], inData.dist_fn_x_px_electron.grid.data[0][-1]]))
        except:
            print 'Reading error for: ' + f
    #Convert into np arrays for ease of use

    all_Pos = np.array(all_Pos)
    all_N_integrated = np.array(all_N_integrated)
    #Create folder to save into
    if not os.path.exists(savePath + 'Dist_evo/'):
        os.makedirs(savePath+ 'Dist_evo/')
    #Save the files incase the plotting fails
    np.savetxt(savepath + 'Dist_evo/' + 'densityEvolution.txt', np.c_[all_Pos, all_N_integrated])
    all_Pos = all_Pos * 1e6 # Convert into mu m
    plt.plot(all_Pos, all_N_integrated)
    plt.xlabel(r"Distance $(\mu m)$")
    plt.ylabel(r'Number Density $m^{-3}$')
    plt.savefig(savepath + 'Dist_evo/' + 'densityEvolution.png')

        
def momentumVsTime(filelist, inpath, savepath):
#==============================================================================
#     Convert the 2D arrays of the distfunc into 1D for each file by histogramming
#     the different rows into the px values.
#     This then forms a 2D array for the whole simulation showing the evolution
#==============================================================================   
    plt.close()
    allPx_integrated = []
    all_Times = []
    all_Pos = []
    laserField_mag = []
    getAxis = 0
    for f in filelist:
        inData = sh.getdata(inpath + f)
        if f == filelist[getAxis]:
            try:
                # 181026:: This step can fail causing the saving of
                # px_eV to not occur.
                yaxis = inData.dist_fn_x_px_electron.grid.data[1]
                px_eV = (((yaxis**2) / (2 * m_e))) * q_e
#                px_MeV = px_eV / 1e6
#                px_GeV = px_eV / 1e9
            except:
                print 'failed to get axis data from sdf: ', getAxis
                getAxis += 1
        try:
            px = inData.dist_fn_x_px_electron.data.T
            intPx = []
            for i in range(len(px[:,0])):
                intPx.append(np.average(px[i]))
            allPx_integrated.append(intPx)
            all_Times.append(inData.Header.values()[9] * 1e12)
            all_Pos.append(np.average([inData.dist_fn_x_px_electron.grid.data[0][0], inData.dist_fn_x_px_electron.grid.data[0][-1]]))
            laserField_mag.append(np.sum(abs(inData.Electric_Field_Ez.data)))
        except:
            print 'Reading error for: ' + f
    #Convert into np arrays for ease of use
    allPx_integrated = np.array(allPx_integrated)
    all_Times = np.array(all_Times)
    all_Pos = np.array(all_Pos)
    laserField_mag = np.array(laserField_mag)
    #Create folder to save into
    if not os.path.exists(savePath + 'Dist_evo/'):
        os.makedirs(savePath+ 'Dist_evo/')
    #Save the files incase the plotting fails
    np.savetxt(savepath + 'Dist_evo/' + 'px_vs_t.txt', allPx_integrated, fmt = '%.5e')
    np.savetxt(savepath + 'Dist_evo/' + 'xaxis_t.txt', all_Times, fmt = '%.5e')
    np.savetxt(savepath + 'Dist_evo/' + 'yaxis_p_eV.txt', px_eV, fmt = '%.5e')
    np.savetxt(savepath + 'Dist_evo/' + 'xaxis_x.txt', all_Pos, fmt = '%.5e')
    np.savetxt(savepath + 'Dist_evo/' + 'laserField_mag.txt', laserField_mag, fmt = '%.5e')
    
    createPlot_dist_evo(allPx_integrated, all_Times, yaxis, savepath, xAxis=1)
    createPlot_dist_evo(allPx_integrated, all_Pos, yaxis, savepath, xAxis=2)
    createPlot_dist_evo(allPx_integrated, np.array(range(len(all_Pos))), yaxis, savepath, xAxis=0)


    plotLaserFieldStrength_evo(laserField_mag, all_Times, all_Pos, savepath)
    
    
def momentumEvo_and_numD_with_laser(filelist, timesteps, inpath, savepath, vMin, vMax):
    #==============================================================================
#     Convert the 2D arrays of the distfunc into 1D for each file by histogramming
#     the different rows into the px values.
#     This then forms a 2D array for the whole simulation showing the evolution
#==============================================================================   
    plt.close()
    allPx_integrated = []
    all_Times = []
    all_Pos = []
    laserField_mag = []

    #Creating the color map with transparency
    cmap1 = mpl.colors.LinearSegmentedColormap.from_list('my_cmap2',['orange','red'],256)
    cmap1._init() # create the _lut array, with rgba values
    alphas = np.linspace(0, 1.0, cmap1.N+3)    # create your alpha array and fill the colormap with them. here it is progressive, but you can create whathever you want
    cmap1._lut[:,-1] = alphas    

    for f, time in zip(filelist, timesteps):
        inData = sh.getdata(inpath + f)
        if f == filelist[0]:
            yaxis = inData.dist_fn_x_px_electron.grid.data[1]
            px_eV = (((yaxis**2) / (2 * m_e))) * q_e
#            px_MeV = px_eV / 1e6
#            px_GeV = px_eV / 1e9
        try:
            px = inData.dist_fn_x_px_electron.data.T
            intPx = []
            for i in range(len(px[:,0])):
                intPx.append(np.average(px[i]))
            allPx_integrated.append(intPx)
            all_Times.append(inData.Header.values()[9] * 1e12)
            all_Pos.append(np.average([inData.dist_fn_x_px_electron.grid.data[0][0], inData.dist_fn_x_px_electron.grid.data[0][-1]]))
            laserField_mag.append(np.sum(abs(inData.Electric_Field_Ez.data)))
            indivual_numDens_with_laser(inData,time, inpath, savepath, cmap1, vMin, vMax)
            
        except:
            print 'Reading error for: ' + f
    #Convert into np arrays for ease of use
    allPx_integrated = np.array(allPx_integrated)
    all_Times = np.array(all_Times)
    all_Pos = np.array(all_Pos)
    laserField_mag = np.array(laserField_mag)
    #Create folder to save into
    if not os.path.exists(savePath + 'Dist_evo/'):
        os.makedirs(savePath+ 'Dist_evo/')
    #Save the files incase the plotting fails
    np.savetxt(savepath + 'Dist_evo/' + 'px_vs_t.txt', allPx_integrated, fmt = '%.5e')
    np.savetxt(savepath + 'Dist_evo/' + 'xaxis_t.txt', all_Times, fmt = '%.5e')
    np.savetxt(savepath + 'Dist_evo/' + 'yaxis_p_eV.txt', px_eV, fmt = '%.5e')
    np.savetxt(savepath + 'Dist_evo/' + 'xaxis_x.txt', all_Pos, fmt = '%.5e')
    np.savetxt(savepath + 'Dist_evo/' + 'laserField_mag.txt', laserField_mag, fmt = '%.5e')
    
    createPlot_dist_evo(allPx_integrated, all_Times, yaxis, savepath, xAxis = 1)
    createPlot_dist_evo(allPx_integrated, all_Pos, yaxis, savepath, xAxis = 2)
    createPlot_dist_evo(allPx_integrated, np.array(range(len(all_Pos))), yaxis, savepath, xAxis = 0 )
    plotLaserFieldStrength_evo(laserField_mag, all_Times, all_Pos, savepath)
    
def plotLaserFieldStrength_evo(Strength, Time, Distance, savepath):
    plt.close()
    plt.plot(Time, Strength)
    plt.ylabel('Total Mag of Laser Field, Ez')
    plt.xlabel('Time (ps)')
    plt.savefig(savepath + 'Dist_evo/' + 'laserField_evo_time.png', dpi = 250)
    plt.close()
    plt.plot(Distance, Strength)
    plt.ylabel('Total Mag of Laser Field, Ez')
    plt.xlabel(r'Distance $(\mu m)$')
    plt.savefig(savepath + 'Dist_evo/' + 'laserField_evo_dist.png', dpi = 250)
    
    
    


    
            
#==============================================================================
# Get the file input from the cmd line
#==============================================================================
parser = argparse.ArgumentParser()                                               
parser.add_argument("--folder", "-f", type=str, required=True)
parser.add_argument("--options", "-o", type=str, required=True)
parser.add_argument("--vmin", "-l", type=str, required=False)
parser.add_argument("--vmax", "-u", type=str, required=False)
parser.add_argument("--append", "-a", type=str, required=False)

plotOptions = parser.parse_args().options.split('_')


if parser.parse_args().vmin is not None:
    print parser.parse_args().vmin
    vMin = float(parser.parse_args().vmin)

if parser.parse_args().vmax is not None:
    print parser.parse_args().vmax
    vMax = float(parser.parse_args().vmax)
    
if parser.parse_args().append is not None:
    sdf_crop = True
else:
    sdf_crop = False


try:
    vMin
except NameError:
    vMin = None
    
try:
    vMax
except NameError:
    vMax = None
    

folderPath = str(parser.parse_args().folder)

#==============================================================================
# Search for the files in that folder in scratch 
#==============================================================================
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

if sdf_crop:
    print; print 'Appending files created'
    png_list, pngTimeSteps = savedFiles(savePath) 
     
    remainingTimesteps = [x for x in simTimeSteps if x not in pngTimeSteps]
    
    remianingSDF = [sdf_list[i] for i in remainingTimesteps]
    print 'Remaining times: ', remainingTimesteps
#    print 'files: ', remianingSDF
    sdf_list = remianingSDF
    simTimeSteps = remainingTimesteps  
    print 'Cropped List: ', simTimeSteps


#==============================================================================
# copy the input deck so there is a record of it
#==============================================================================
if len(inputDeck) == 1:
    import shutil
    shutil.copy2(scratchPath+inputDeck[0], savePath)
    
#==============================================================================
#   Can chose
#     a -- only do sdf files in numbder dens with laser if true
#
# Give the user choice of what plots are occuring at run time
# with out having to modify the script each run
# Each option is selected using a char
#     m -- momentumVsTime
#     d -- distFunc_pxX_plot
#     l -- numberDens_with_laser_pulse
#     n -- numberDens
#     b -- numD_and_Dist_for_all_time
#     a -- momentumEvo_and_numD_with_laser
#     p -- densityVsTime
#     e -- electric field
#     s -- spectrum from px py
#==============================================================================

##Last file can be corrupt
#sdf_list = sdf_list[:-1]
#simTimeSteps = simTimeSteps[:-1]
  
for option in plotOptions:
    if option is 'm':
        momentumVsTime(sdf_list, scratchPath, savePath)
    if option is 'd':
        distFunc_pxX_plot(sdf_list, simTimeSteps, scratchPath, savePath)
    if option is 'l':
        numberDens_with_laser_pulse(sdf_list, simTimeSteps, scratchPath, savePath, vMin, vMax)
    if option is 'b':
        numD_and_Dist_for_all_time(sdf_list, simTimeSteps, scratchPath, savePath, vMin, vMax)
    if option is 'n':
        numberDens(sdf_list, simTimeSteps, scratchPath, savePath, vMin, vMax)
    if option is 'a':
        momentumEvo_and_numD_with_laser(sdf_list, simTimeSteps, scratchPath, savePath, vMin, vMax)
    if option is 'p':
        densityVsTime(sdf_list, scratchPath, savePath)
    if option is 'e':
        electricField(sdf_list, simTimeSteps, scratchPath, savePath)
    if option is 's':
        pxpySpectrum(sdf_list, simTimeSteps, scratchPath, savePath)

     
