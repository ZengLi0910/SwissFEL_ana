#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:28:06 2018

@author: esposito_v

This script processes the data from the scan taken with ECO.
The results of the analysis are written fro each step in a HDF5 file.

You need to configure the config.py file for the scan you want to analyze. In particular:
    - Configure workspace by providing the different folders (data folder, save folder, etc)
    - Give the scan number
    - Configure the detectors: provide gain and pedestals, 
        filters, analysis function to be preformed, etc
    - Provide laser rep rate and other information required
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
from dask import compute


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
    key = config.detectors['I0'].Id
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
        
        """ Prepare file to save step """
        saveDir_step = Path(run_ana.save_dir + fname[:-15] + '_steps/')
        if not os.path.exists(saveDir_step):
            os.mkdir(saveDir_step)
#            subprocess.call(['chmod', '-R', 'g+rw', saveDir_step])
        
        save_nb = str(bin_count)            
        if len(save_nb)==1:
            save_nb = '00'+save_nb
        elif len(save_nb)==2:
            save_nb = '0'+save_nb
        save_name = Path(saveDir_step / (run_ana.save_prefix + 'step' + save_nb + ".h5"))
        
        if not config.use_stepfile or not os.path.exists(save_name):
            start = time.time()
            results = datastorage.DataStorage()
            results.save(save_name)
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
                
            print( '\nNumber of laser on shots '+str(len(eventIds_bin[laser_on])) )
            print( 'Number of laser off shots '+str(len(eventIds_bin[laser_off]))+'\n' )
            
            JFdata = datastorage.DataStorage()

            """ Analyze I0 detector """
            if useI0:
                det = config.detectors['I0']
                I0dat = ana.JFdata(data[det.Id].data[idx_bin], det)
                I0dat.filtered = I0dat.apply_filter(hist_range=hist_range)

                I0 = I0dat.run_ana()
                I0 = np.r_[compute(*I0)]

                    
            """ I0 from PBPS (to do if needed) """
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
                res = np.r_[compute(*ana_out)]
                results[det.name] = {'laser_on': res[laser_on], 'laser_off': res[laser_off]}
            
            
            """ Save result in step file """
            results.step_pos = step_pos
            results.scan_parameter = scan_info['scan_parameters']
            results.I0 = results['I0'] = {'laser_on': I0[laser_on], 'laser_off': I0[laser_off]}
            results.save(save_name)
            os.chmod(save_name,0o664)
            time_step = time.time()-start
            print('\nAnalysis of bin {0} took {1:.1f} minutes'.format(str(bin_count), time_step/60))
            
        else:
            print('\nStep file '+ str(save_name)+' exist already and was not overwritten.')

time_tot = time.time() - time_tot
print('\n\nIt took {0:.1f} minutes to analyze run {1}'.format(time_tot/60, str(config.run)))

















