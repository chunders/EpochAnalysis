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

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


d = sh.getdata(113)
NumberDensity = d.Derived_Number_Density.data
NDSmooth = smooth(NumberDensity, 20)

grid = (d.Electric_Field_Ex.grid.data[0][1:] + d.Electric_Field_Ex.grid.data[0][:-1]) * 0.5

ax = plt.subplot()
ax.plot(grid, d.Electric_Field_Ex.data, label = 'X'); ax.plot(grid, d.Electric_Field_Ey.data , label = 'Y'); ax.plot(grid, d.Electric_Field_Ez.data, label = 'Z'); ax.legend();
ax1 = ax.twinx()
ax1.plot(grid, NDSmooth)