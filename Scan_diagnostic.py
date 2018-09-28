#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 14:43:42 2018

@author: esposito_v
"""

import sys
import os
import subprocess
import time
import matplotlib.pyplot as plt
import numpy as np
import datastorage
from importlib import reload
from pathlib import Path
from dask import compute, delayed


""" custom libraries """
import sf_ana_tools as ana

os.chdir('/das/work/units/bernina/p17536/dev/parallel')
import config
reload(config)

#from escape.parse.swissfel import parseScanEco_v01, readScanEcoJson_v01
from swissfel_parser import parseScanEco_v01, readScanEcoJson_v01


useI0 = 1

#plt.close('all')
plt.ion()

step = config.step
if not step:
    step = [0]
bins = config.bins

hist_range = np.arange(-2,50,0.05)
histo = dict()

""" ------------------------------------------------ """
run_ana = config.run_ana

time_tot = time.time()
for fname in run_ana.fnames:
    print(fname+'\n')
    
    """ Parse data """
    fpath = Path(run_ana.config_folder) / Path(fname)
    data, dstores = parseScanEco_v01(fpath, createEscArrays=True)
    scan_info, p = readScanEcoJson_v01(fpath)
    motor_pos = np.squeeze(scan_info['scan_values'])
    print('Data parsed. \n')
    
    
    """ Get bin index """
    key = config.detI0.Id
#    eventIds = data[key].eventIds #this step takes a surprise amount of time... smth is probably wrong with the JF files
    eventIds = ana.get_JFeventIds_dirty(dstores, key, step_size = 4)
    last_idx_step = np.cumsum(data[key].stepLengths)
    motor_pos_tot = np.zeros(eventIds.shape)
    for ii in range(len(last_idx_step)):
        if ii==0:
            motor_pos_tot[:last_idx_step[ii]] = motor_pos[ii]
        else:
            motor_pos_tot[last_idx_step[ii-1]:last_idx_step[ii]] = motor_pos[ii]
        
    if bins is None:
        bin_center = motor_pos
    else:
        bin_center = bins
    bin_size = min(np.diff(bin_center))
    bin_edges = np.append(bin_center[0]-0.5*bin_size, bin_center+0.5*bin_size)
    bin_target = np.digitize(motor_pos_tot, bin_edges)


    """ Loop through all bins """
    print('Start bin loop. \n')
    for bin_count, step_pos in enumerate(bin_center):
        if (not np.in1d(bin_count,step)) & (not (step==None)):
            continue
    
        start = time.time()
        results = datastorage.DataStorage()
        print('\n\nAnalyze bin {} of {}'.format(bin_count, len(bin_center)-1) )
        
        """ Find index for laser on/off shots in the bin """
        idx_bin = bin_target == (bin_count+1)
        eventIds_bin = eventIds[idx_bin]
        if config.laserReprate:
            laser_off = np.mod(eventIds_bin,int(100/config.laserReprate))>0
            laser_on = np.logical_not(laser_off)
        else:
            laser_off = np.ones(eventIds_bin.shape)
            laser_on = np.zeros(eventIds_bin.shape)
        
        JFdata = dict()

        """ Analyze I0 detector """
        if useI0:
            det = config.detI0
            I0dat = ana.JFdata(data[det.Id].data[idx_bin], det)
            I0dat.filtered = I0dat.apply_filter(hist_range=hist_range)
            I0 = I0dat.run_ana()
            I0 = np.r_[compute(*compute(*I0))]
            
            """ Plot detector """
            imgI0 = ana.sum_delayedImgs(I0dat.filtered)
            plt.figure(det.name, figsize=(12,8))
            plt.subplot2grid((2,2),(0,0), rowspan=2)
            plt.title(det.name)
            plt.imshow(imgI0, origin='lower', clim=(0,5000))
            
            plt.subplot2grid((2,2),(0,1))
            plt.title('Histograms')
            hist, bins = ana.getHist(I0dat.CMcorr[0].compute(), bins=hist_range)
            plt.semilogy(bins[:-1],hist)
            hist, bins = ana.getHist(I0dat.filtered[0].compute(), bins=hist_range)
            plt.semilogy(bins[:-1],hist)
            plt.xlabel('Pixel intensity')
            plt.ylabel('Pixel count')
            
            plt.subplot2grid((2,2),(1,1))
            plt.title('I0 histogram')
            n, I0bins, patch = plt.hist(I0, bins=100)
            plt.xlabel('I0 intensity')
            plt.ylabel('Shot count')
            plt.tight_layout()
            
            
            """ I0 from PBPS """
            channelsPBPS = ['SLAAR21-LSCP1-FNS:CH4:VAL_GET', 'SLAAR21-LSCP1-FNS:CH5:VAL_GET', 
                        'SLAAR21-LSCP1-FNS:CH6:VAL_GET', 'SLAAR21-LSCP1-FNS:CH7:VAL_GET']
            
            
            """ Loop through all detectors """
            for JF, det_name in enumerate(config.detectors.keys()):
                det = config.detectors[det_name]
                if not config.useJF[JF]:
                    continue
                
                print('\nNow analyzing detector '+det_name)
                JFdata[det.name] = ana.JFdata(data[det.Id].data[idx_bin], det)
                JFdata[det.name].filtered = JFdata[det.name].apply_filter(hist_range=hist_range)
                if hasattr(det, 'roi'):
                    if not det.roi is None:
                        JFdata[det.name].roi(coord=det.roi)
                        
                
                """ Analyze detector """                
                ana_out = JFdata[det.name].run_ana()
                res = np.r_[compute(*compute(*ana_out))]
                results[det.name] = {'laser_on': res[laser_on], 'laser_off': res[laser_off]}
                
                """ Plot detector """
                imgDet = ana.sum_delayedImgs(JFdata[det.name].filtered)
                plt.figure(det.name, figsize=(12,8))
                plt.subplot2grid((2,2),(0,0), rowspan=2)
                plt.title(det.name)
                plt.imshow(imgDet, origin='lower', clim=(0,5000))
                
                plt.subplot2grid((2,2),(0,1))
                plt.title('Pixel histograms')
                hist, bins = ana.getHist(JFdata[det.name].CMcorr[0].compute(), bins=hist_range)
                plt.semilogy(bins[:-1],hist)
                hist, bins = ana.getHist(JFdata[det.name].filtered[0].compute(), bins=hist_range)
                plt.semilogy(bins[:-1],hist)
                plt.xlabel('Pixel intensity')
                plt.ylabel('Pixel count')
                
                plt.subplot2grid((2,2),(1,1))
                plt.title('ROI')
                plt.tight_layout()

time_tot = time.time() - time_tot
print('\n\nIt took {0:.1f} minutes to analyze run {1}'.format(time_tot/60, str(config.run)))

plt.figure('Correlations')
plt.scatter(I0, res, s=8)


























