#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:58:41 2018

@author: esposito_v
"""

import numpy as np
import os
import sys
import pandas as pd
import h5py
import jungfrau_utils as ju
import matplotlib.pyplot as plt
from dask import delayed


  
    
    
def roi_bkgRoi(image_in, roi, bkg_roi):
    """
    Returns the total intensity (sum of all pixels) in a roi and corresponding background based on a background
    roi from an input image. The function checks for overlap between the roi and bkg_roi, and takes it into account.
    """

    # check for intersection
    temp_roi = roi + np.array([[-8, 8], [-7, 7]])  # extended roi for safe background intensity
    fintersect = (temp_roi[0][0] < bkg_roi[0][1] and bkg_roi[0][0] < temp_roi[0][1] and
                  temp_roi[1][0] < bkg_roi[1][1] and bkg_roi[1][0] < temp_roi[1][1])

    if fintersect:
        intersect = [[max(temp_roi[0][0], bkg_roi[0][0]), min(temp_roi[0][1], bkg_roi[0][1])],
                     [max(temp_roi[1][0], bkg_roi[1][0]), min(temp_roi[1][1], bkg_roi[1][1])]]
    else:
        intersect = []

    temp_roi = intersect
#    return temp_roi

    if len(image_in.shape) == 2:
        img_roi = image_in[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
        img_bkg_roi = image_in[bkg_roi[0][0]:bkg_roi[0][1], bkg_roi[1][0]:bkg_roi[1][1]]
        if fintersect:
            img_temp_roi = image_in[temp_roi[0][0]:temp_roi[0][1], temp_roi[1][0]:temp_roi[1][1]]
        
    #    return img_roi
    
        size_roi = img_roi.shape[0] * img_roi.shape[1]
        size_bkg_roi = img_bkg_roi.shape[0] * img_bkg_roi.shape[1]
        if fintersect:
            size_temp_roi = img_temp_roi.shape[0] * img_temp_roi.shape[1]
    
        intensity_roi = np.nansum(np.nansum(img_roi))
        intensity_bkg_roi = np.nansum(np.nansum(img_bkg_roi))
        if fintersect:
            intensity_temp_roi = np.nansum(np.nansum(img_temp_roi))
    
            intensity_bkg_roi = (intensity_bkg_roi-intensity_temp_roi) / (size_bkg_roi-size_temp_roi) * size_roi
        else:
            intensity_bkg_roi = intensity_bkg_roi / size_bkg_roi * size_roi
            
        intensity = np.array(intensity_roi)
        bkg = np.array(intensity_bkg_roi)
        
    elif len(image_in.shape) == 3:
        img_roi = image_in[:, roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
        img_bkg_roi = image_in[:, bkg_roi[0][0]:bkg_roi[0][1], bkg_roi[1][0]:bkg_roi[1][1]]
        if fintersect:
            img_temp_roi = image_in[:, temp_roi[0][0]:temp_roi[0][1], temp_roi[1][0]:temp_roi[1][1]]
    
        size_roi = img_roi.shape[1] * img_roi.shape[2]
        size_bkg_roi = img_bkg_roi.shape[1] * img_bkg_roi.shape[2]
        if fintersect:
            size_temp_roi = img_temp_roi.shape[1] * img_temp_roi.shape[2]
    
        intensity_roi = np.nansum(np.nansum(img_roi,axis=1),axis=1)
        intensity_bkg_roi = np.nansum(np.nansum(img_bkg_roi,axis=1),axis=1)
        if fintersect:
            intensity_temp_roi = np.nansum(np.nansum(img_temp_roi,axis=1),axis=1)
    
            intensity_bkg_roi = (intensity_bkg_roi-intensity_temp_roi) / (size_bkg_roi-size_temp_roi) * size_roi
        else:
            intensity_bkg_roi = intensity_bkg_roi / size_bkg_roi * size_roi
        
        intensity = np.asarray(intensity_roi)
        bkg = np.asarray(intensity_bkg_roi)
        
    else: print('Image format no valid')

    return intensity, bkg


def roi_integrate(image_in, roi):
    """ integrate a given roi in the images """
    if len(image_in.shape) == 2:
        img_roi = image_in[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
        intensity_roi = np.nansum(np.nansum(img_roi))
        return intensity_roi
        
    elif len(image_in.shape) == 3:
        img_roi = image_in[:, roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
        intensity_roi = np.nansum(np.nansum(img_roi,axis=1),axis=1)
        return np.asarray(intensity_roi)
    
    else: print('Image format no valid')
    return 0



def I0_JF(JFcorr, roi=[[1530,2047],[0,1023]]):
    """ takes corrected images as input for the moment. If the JF IO modules will 
    be moved to a separate detector, then is may be a good idea to implement in this function
    the JFImages class and corrections for the I0 
    REMARK: nan pizel are being ignored """
    
    return roi_integrate(JFcorr, roi)



def get_DAQ_data(filenames, daq_labels, motor_pos = None):
    """
    daq_labels syntax:
        daq_labels = {}
        daq_labels["gonio_z"] = "event_info/goniometer/g_z"
        ...
    """
    df_orig = pd.DataFrame(columns=daq_labels.keys(), )
    failed_filenames = []
    
    if isinstance(filenames, str):
        filenames = [filenames, ]
    filenames = sorted(filenames)
    
    print('Load DAQ_labels for files:')
    for ii, fname in enumerate(filenames):
        print('%s' %fname)
        try:
            f = h5py.File(fname, "r")
#            main_dset = f[f.keys()[1]]
        except:
            print("Error loading file %s: %s" % (fname, sys.exc_info()[1]) )
            failed_filenames.append(fname)
            continue
        mydict = {}
        for k,v in enumerate(daq_labels.keys()):                
            if ('lam' in v) or ('PIPS' in v) or ('tcorr' in v):
                if f[daq_labels[v]].shape[0] == 1:
                    print('Step %d is crap' % ii )
                    continue
                mydict[v] = np.nansum(f[daq_labels[v]], axis=1) # temporary solution to integrate diode signal
            else:
                mydict[v] = f[daq_labels[v]]
        
        try:
            tmp_df = pd.DataFrame(data=mydict)
        except ValueError:
            print('Step %d is crap' % ii )
            continue
                
        
        if motor_pos is not None: tmp_df['scan_motor'] = float(motor_pos[ii])
        # Append the data to the dataframe
        df_orig = df_orig.append(tmp_df)
        
    if 'pulseID' in df_orig.keys():
        df_orig = df_orig.set_index('pulseID')
    
    print('\n\n-----------------------------------------------------------------------------------------\n\n')
    return df_orig, mydict
#    return mydict



def readJson(fname):
    import json
    with open(fname) as f:
        a = json.load(f)
    return a



def bin_df(df, bin_key=None, bin_edges=None, bin_size=None, 
        statkeys=None,
        statfuns=[np.nanmean,np.nansum,np.nanstd,len], statlbls=['_mean','_sum','_std','_count'],
        weightkey=None,
        wstatfuns=[np.average], wstatlbls=['_avg']):
    """
    Binned statistics of pandas datafield using arbitrary statistics functions.
    Can calculate weighted average and can uses any statistics function that uses 'weights' as argument
    
    Default statistics on each column
        np.nanmean,np.nansum,np.nanstd,len

    Default weighted statistics on each column
        default:np.average
    
    Usage
    -----
    
    To use weighted statistics:
        set weightkey to the key holding the weight\n
        provide list of statistics function which can use 'weights' as argument
 
    Parameter description
    ---------------------
    df:
        Pandas dataframe
    bin_key: 
        key in df to bin (e.g. jitter corrected delay). If None or not found: uses first key (with warning)
    bin_edges,bin_size: 
        definition of bins.
        If binning scan motor with always same value for some pulses, leave both empty. It bins to the unique values then
    statkeys:
        keys for which to apply statistics; None means all (default). To speed up, select only needed keys
    statfuns/statlbls:
        function list/string label list of non weighted statistics functions to be applied. Result will be stored in new columns named by original key + the label 
    weightkey:
        key holding the weight of each row in df when doing weighted statistics. If None or not found, ones are used (with warning)
    wstatfuns/wstatlbls: 
        Same as statfuns/statlbls. A wstatfun must accept 'weights' as keyword argument
        
        
    Room to improve: Add bin count manually and remove len from statfuns to avoid identical bin counts for all keys
    """
    
    
    if len(df)==0:
            raise Exception('Binning: Empty dataframe, cannot bin')


    if statkeys is None:
        statkeys=df.keys()

    if bin_key is None:
    	bin_key=df.keys()[0]
    	print('Binning: Warning, no bin_key provided, using '+bin_key)

    # Do statistics also on bin field to get an idea of the x error due to binning
    if bin_key not in statkeys:
        statkeys=np.concatenate([bin_key],statkeys)
    
    if (bin_size is None) and (bin_edges is None):
        bin_center = sorted(df[bin_key].unique())
        if len(bin_center)<2:
            print('Binning: Warning, only one unique value inbin_key, using only one bin!')
            bin_size=1
            bin_center=[bin_center]
        else:
            bin_size = min(np.diff(bin_center))
        bin_edges = np.append(bin_center[0]-0.5*bin_size, bin_center+0.5*bin_size)
    elif bin_edges is None:
        bin_edges = np.arange(df[bin_key].min()-0.5*bin_size,df[bin_key].max()+0.5*bin_size,bin_size)
        bin_center = bin_edges[:-1]+0.5*bin_size
    else:
        bin_center = (bin_edges[0:-2]+bin_edges[1:-1])/2


    if not len(statfuns)==len(statlbls):
            raise Exception('alls statistics functions need a column label')
        
    if not set(statkeys) <= set(df.keys()):
            raise Exception('at least one key for statistics not in dataframe')
            
    if (not weightkey in df.keys()) and (len(wstatfuns)>0):
        print('weightkey not in df, using ones as weights; set wstatfuns=[] and wstatlbs=[] to remove weighted average')
          

    df_out = pd.DataFrame(bin_center, index=range(0,len(bin_center)),columns=[bin_key+'_binned'])    
    
    #clip unused data before binning (remove if data outside bin range should go to first and last bin)
    df=df[(df[bin_key]>=bin_edges[0])&(df[bin_key]<=bin_edges[-1])]    
 
 
    d_tbin=np.digitize(df[bin_key],bin_edges)-1  
    d_all=df[statkeys].values
    d_w=df[weightkey] if weightkey in df.keys() else np.ones(np.shape(d_all)[0])
    l_bins = np.unique(d_tbin)    
    for binidx in l_bins:
           d_all_bin=d_all[binidx==d_tbin,:]
           d_w_bin=d_w[binidx==d_tbin]
           for nkey,key in enumerate(statkeys):
                for fun,lbl in zip(statfuns,statlbls):
                      df_out.loc[binidx,key+lbl]=fun(d_all_bin[:,nkey])
                for fun,lbl in zip(wstatfuns,wstatlbls):
                      kwargs={'weights':d_w_bin}
                      df_out.loc[binidx,key+lbl]=fun(d_all_bin[:,nkey],**kwargs)
                   
		      #Fdf_out.loc[binidx,key+lbl] may be optimized for speed
            #(avoid locating in df_out, use array instead)
    return df_out




























