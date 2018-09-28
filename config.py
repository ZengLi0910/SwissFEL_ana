#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 11:55:20 2018

@author: esposito_v

Configuration file for the data processing and diagnostic:
    - Process_data.py
    - Scan_diagnostic.py
"""
import os
import subprocess
import numpy as np
import sf_ana_tools as ana


""" --------------------------- SETUP EXPERIMENT -------------------------- """
exp = ana.experiment(inputs=False)
exp.experiment_type = 'other'
exp.config_folder = '/sf/bernina/data/p17536/res/scan_info/' # where the scan_info are
exp.pgroup_dir = '/sf/bernina/data/p17536/'
exp.save_dir = '/das/work/units/bernina/p17536/ana/' # where you wanna save the step files
exp.save_prefix = 'test_' # prefix for the saved step files


""" ------------------------- Analysis parameters ------------------------- """
step = 1 # number of bins (steps) to be analyzed
bins = None # bins for the scan motor. If none the motor position are used
use_stepfile = 0 # Set to 0 to reanalyze the images and overwrite the step files


useJF = [1,0,0] # perform the analysis of the relevant JF: [scatt, fluo, stripsel]

run = [140]

laserReprate = 12.5 # [Hz]
#laserReprate = 0





""" ------------------------------ JF SETUP ------------------------------ """
exp.gainJF = {
        'scatt':'/sf/bernina/config/jungfrau/gainMaps/JF01T03V01/gains.h5',
        'I0':'/sf/bernina/config/jungfrau/gainMaps/JF03T01V01/gains.h5',
        'fluo':'/sf/bernina/config/jungfrau/gainMaps/JF04T01V01/gains.h5',
        'stripsel':'/sf/bernina/config/jungfrau/gainMaps/JF05T01V01/gains.h5'
        }

#base = '/sf/bernina/data/p17536/res/JF_pedestal/pedestal_20180717_1632'
base = '/sf/bernina/data/p17536/res/JF_pedestal/pedestal_20180725_1038'
fpedestal = {
        'scatt':base+'.h5.JF01T03V01.res.h5',
        'I0':base+'.h5.JF03T01V01.res.h5',
        'fluo':base+'.h5.JF04T01V01.res.h5',
        'stripsel':base+'.h5.JF05T01V01.res.h5'
        }

detectors = dict()

""" ----- JF I0  (3) ----- """
det = 'I0'
roi_I0 = [[50,200],[100,900]]
ana_fun = ana.I0_JF
ana_args = dict(roi=roi_I0)
filters = [[4.5,30]] # set to None if you dont want any filters
detI0 = ana.detector(name=det, gainFile=exp.gainJF[det], pedestalFile=fpedestal[det], 
                     filters=filters, ana_fun=ana_fun, ana_args=ana_args)
detI0.Id = 'JF03T01V01'



""" ----- JF fluo (4) ----- """
det = 'fluo'
roi_fluo = [[10,500],[10,800]]
#roi_fluo = [[110,390],[260,1000]] # diffraction peak on lfuo det...
#ana_fun = ana.fluo_ana
#ana_args = dict(roi=roi_fluo)
ana_fun = ana.roi_bkgRoi
ana_args = dict(roi=roi_fluo, bkg_roi=roi_fluo)
filters = [[4.5,1e6]]
detectors[det] = ana.detector(name=det, gainFile=exp.gainJF[det], pedestalFile=fpedestal[det], 
         filters=filters, ana_fun=ana_fun, ana_args=ana_args)
detectors[det].Id = 'JF04T01V01'



""" ----------------------------------------------------------------------- """
run_ana = ana.sf_scan(exp)
if not os.path.exists(run_ana.save_dir):
    os.mkdir(run_ana.save_dir)
    subprocess.call(['chmod', '-R', 'g+rw', run_ana.save_dir])
run_ana.add_file(run)




#channels = {
#        'CH1_buffer':'SAROP21-PALMK134:CH1_BUFFER',
#        'CH2_buffer':'SAROP21-PALMK134:CH2_BUFFER',
#        'BAM070_EOM1_arrt_B1':'S10BC01-DBAM070:GPAC_EOM1_arrt_B1',
#        'BAM070_EOM1_arrt_B2':'S10BC01-DBAM070:GPAC_EOM1_arrt_B2',
#        'BAM070_EOM2_arrt_B1':'S10BC01-DBAM070:GPAC_EOM2_arrt_B1',
#        'BAM070_EOM2_arrt_B2':'S10BC01-DBAM070:GPAC_EOM2_arrt_B2',
#        'BAM010_EOM1_arrt_B1':'SINLH01-DBAM010:GPAC_EOM1_arrt_B1',
#        'BAM010_EOM1_arrt_B2':'SINLH01-DBAM010:GPAC_EOM1_arrt_B2',
#        'BAM010_EOM2_arrt_B1':'SINLH01-DBAM010:GPAC_EOM2_arrt_B1',
#        'BAM010_EOM2_arrt_B2':'SINLH01-DBAM010:GPAC_EOM2_arrt_B2'
#        }







