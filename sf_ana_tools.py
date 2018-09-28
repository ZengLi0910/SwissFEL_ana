#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:17:37 2018

@author: esposito_v
"""

import os
import sys
import glob
import h5py
import numpy as np
import dask
from dask import array as da
from dask import delayed


""" ------------------------- SCAN AND EXP CLASS ------------------------- """
class experiment(object):    
    def __init__(self, inputs = True):
        if inputs == True:
            print('Welcome to your new experiment. \nPlease enter a mandatory settings beore proceeding.')
            self.experiment_type = input('Enter the type of experiment:')
            self.save_prefix = input('Choose a prefix for the files that will be saved:')
            self.config_folder = input('Scan info folder:')
            self.save_dir = input('Save directory:')
            
        else:
            self.experiment_type = []
            self.pgroup_dir = []
            self.save_prefix = []
            self.config_folder = []
            self.save_dir = []
            self.gainJF = []
        
#        if not os.path.exists(self.save_dir):
#            os.mkdir(self.save_dir)
#            subprocess.call(['chmod', '-R', 'g+rw', self.save_dir])
            
    def __call__(self):
        print('Experiment settings: \n')
        members = [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith("__")]
        for member in members:
            if member is 'daq_labels':
                continue
            else:
                print(member + ' : ' +  self.__dict__[member])
        print('\n\n')



class sf_scan(object): 
    def __init__(self, experiment):
        
        """ Config given from experiment class """
        try:
            self.config_folder = experiment.config_folder
            self.experiment = experiment.experiment_type
            self.save_prefix = experiment.save_prefix
            self.save_dir = experiment.save_dir
            self.gainJF = experiment.gainJF
        except:
            print('\n\nPlease setup experiment.')
        
        self.fnames = []
        self.scan_info = []
        
#        if pedestals is None:
#            pfolder = Path(experiment.pgroup_dir)
#            pfolder = pfolder / Path('res/JF_pedestal')
#            self.pedestals = find_last_pedestal(folder=pfolder)
#            print('\nNo pedestals given, last pedestal will be used: %s\n' %self.pedestal)
#        else:
#            self.pedestals = pedestals
#        
#        self.data = pd.DataFrame()
#        
#        """ Images diagnostics """
#        self.JFsum = dict()
#        self.histo = dict()
#        self.histo_filt = dict()
#        
#        """ Other """
        self.isana = []
#        self.warning = []
#        self.force_ana = 0
        
        print('Have a good (beam)time!\n\n')
        
        
        
    def add_file(self, newfiles):
        """
        Add files to the file list
        Assumes that the number are between 000 and 999.
        """
            
        if isinstance(newfiles, int):
            newfiles = '%03d'%newfiles
            file = glob.glob(self.config_folder + '*'+'run'+newfiles+'*')
            print(file)
            newfiles = os.path.split(file[0])[-1]
        
        elif isinstance(newfiles, list):
            if isinstance(newfiles[0], int):
                files = []
                for newfile in newfiles:
                    newfile = '%03d'%newfile
                    file = glob.glob(self.config_folder + '*'+ 'run'+newfile+'*')
                    if not file:
                        file = glob.glob(self.config_folder + '*'+ 'scan'+newfile+'*')
                        if not file: print('No file found')
                    print(file)
                    file = os.path.split(file[0])[-1]
                    files.append(file)
                newfiles = files
                
        if isinstance(newfiles, str):
            newfiles = [newfiles, ]
            newfiles = sorted(newfiles)
                
                
        for fname in newfiles:
            isempty = not self.scan_info
                
            if fname in self.fnames:
                print('File %s already exists' %fname)
                continue
            
            fname_full = self.config_folder+fname
            scan_info = readJson(fname_full)
            
            if isempty:
                self.scan_motor = scan_info['scan_parameters']
            else:
                if not (scan_info['scan_parameters']['Id'] == self.scan_motor['Id']):
                    print('File %s has a different scan_motor than expected.' %fname)
                    continue
                               
            self.scan_info.append(scan_info)
            self.fnames.append(fname)
            self.isana.append(False)
            
            print('Scan info for file %s uploaded.' %fname)
        print('\n\nHave fun analyzing!\n\n')

    
    
    def remove_file(self, files):
        if isinstance(files, str):
            files = [files, ]
            files = sorted(files)
            
        for file in files:
            self.fnames.remove(file)



class detector:
    def __init__(self, name=None, gainFile=None, pedestalFile=None, filters=None, ana_fun=None, ana_args=None):
        self.name = name
        if (not gainFile is None) and (not pedestalFile is None):
            self.gains, self.pedestals, self.pixel_mask = self.load_gain_pedestal(gainFile, pedestalFile)
        else:
            raise NameError('gains or pedestals not defined')
        self.filters = filters
        self.ana_fun = ana_fun
        self.ana_args = ana_args
        
    def __call__(self):
        print(self.name)
        
    def load_gain_pedestal(self, gainFile, pedestalFile):
        assert isinstance(gainFile, str), 'gain file invalid'
        assert isinstance(pedestalFile, str), 'pedestal file invalid'
        gains = readF(gainFile,"gains")
        pedestals, pixel_mask = readF(pedestalFile,["gains","pixel_mask"])
        return gains, pedestals, pixel_mask





""" ------------------------- JungFrau CLASS ------------------------- """
class JFdata:
    def __init__(self, raw, detector=None, roi=None):
        if roi != None:
            self.raw = self.apply_roi(raw, roi=roi)
        else:
            self.raw = raw
        
        if (not detector is None):
            self.detector = detector
            self.corrected = self.apply_gain_pede_np(detector.gains, detector.pedestals, pixel_mask = detector.pixel_mask)
            self.CMcorr = self.apply_CMcorr(int_max=3, COM_range=8)
            self.filtered = self.apply_filter(detector.filters, to_filter='CM_corr')
        else: print("Warning: no detector given, no pedestal or gain files found.")
    
    def apply_gain_pede_np(self, G, P, pixel_mask=None, mask_value = np.nan):
        image = self.raw
        
        mask = int('0b' + 14 * '1', 2)
        mask2 = int('0b' + 2 * '1', 2)
    
        gain_mask = np.bitwise_and(np.right_shift(image, 14), mask2)
        data = np.bitwise_and(image, mask)
    
        m1 = gain_mask == 0
        m2 = gain_mask == 1
        m3 = gain_mask >= 2
        if G is not None:
            g = m1*G[0] + m2*G[1] + m3*G[2]
        else:
            g = np.ones(data.shape, dtype=np.float32)
        if P is not None:
            p = m1*P[0] + m2*P[1] + m3*P[2]
        else:
            p = np.zeros(data.shape, dtype=np.float32)
        res = np.divide(data - p, g)
        
        if pixel_mask is not None:
#            dv,mv = np.broadcast_arrays(res,pixel_mask) # seems not necessary
            mv = pixel_mask
            if isinstance(image,da.Array):
                mv = da.from_array(mv, chunks=image.chunks[-2:])
#            if len(image.shape)==3:
#                res[mv!=0] = mask_value
#            else:
            res[mv!=0] = mask_value
        return res
    
    
    def apply_CMcorr(self, int_max=3, COM_range=8):
        imgs_CMcorr = np.squeeze(self.corrected.to_delayed())
        imgs_CMcorr = [CM_corr(imgs, int_max=int_max, COM_range=COM_range) for 
                       imgs in imgs_CMcorr]
        return imgs_CMcorr
    
    
    def apply_filter(self, hist_range=None, to_filter = 'CMcorr'):
        if to_filter == 'corrected':
            return filterImage(self.corrected, self.detector.filters, hist_range=hist_range)
        elif to_filter == 'CMcorr':
            return [filterImage(delay, self.detector.filters, hist_range=hist_range) for 
                    delay in self.CMcorr]
    

    def apply_roi(imgs, roi=None):
        roi=np.array(roi)
        return imgs[:,roi[0,0]:roi[0,1],roi[1,0]:roi[1,1]]
#        self.gains = self.gains[:][coord[0,0]:coord[0,1],coord[1,0]:coord[1,1]]
#        self.pedestals = self.pedestals[:][coord[0,0]:coord[0,1],coord[1,0]:coord[1,1]]


    def run_ana(self):
        if self.detector.ana_fun.compute():
            ana_fun = self.detector.ana_fun
            if self.detector.ana_args:
                ana_args = self.detector.ana_args
            else:
                print('Problem with the detector: an analysis function was provided, but no arguments')
                return
        else:
            print('Problem with the detector: please provide an analysis function and its arguments')
            return
            
        ana_out = [ana_fun(imgs, **ana_args) for imgs in self.filtered]
        return ana_out





""" ------------------------- Image analysis FUNCTIONS (delayed) ------------------------- """
@delayed
def filterImage(imgs, filters, hist_range = None):
    """
    Apply filters on pixels.
    INPUT:
        img: input images: single or stack of images
        filters: intensity interval to be retained: [[int1_min, int1_max], [int1_min, int1_max], ...]
        intensity histogram before and after the filtering can be obtained by inputing an histo_range
        ATTENTION: the histogram part does not work with dask.array.array inputs
        UPDATE: For the delayed framework, we want only one output, so the histograms were removed
    """
    
    imgs = np.asarray(imgs)
    
#    if not hist_range is None:
#        hist_before,E = getHist(imgs, bins=hist_range)

    if filters is None:
        print("No filters selected, original images returned")
#        if not hist_range is None:
#            return imgs, [hist_before]
#        else:
#            return imgs, 0
        return imgs
    
    else:
        idx = np.zeros(imgs.shape)
        idx = idx.astype(bool)
        for filt in filters:
            temp = (imgs>filt[0]) & (imgs<filt[1])
            temp = np.logical_or(temp, np.isnan(imgs))
            temp = temp.astype(bool)
            idx = np.logical_or(idx, temp)
        imgs[~idx] = 0
#        imgs[~idx] = np.nan
        
#        if not hist_range is None:
#            hist_after,E = getHist(imgs, bins=hist_range)
#            return imgs, [hist_before, hist_after]
#        else:
#            return imgs
        return imgs


@delayed
def CM_corr(imgs_in, int_max=3, COM_range=5):
    """
    analyze many images at once. Does not work until the end because dask does
    not suport indexation with dask array. Either the COM has to be computed or
    we have to use the delayed function below    
    """
    
    bins = np.arange(-5,int_max,0.05)
    if len(imgs_in.shape) == 2:
        hist, bins = getHist(imgs_in, bins=bins)
        max_pos = np.argmax(hist)
        if max_pos+COM_range+1 > bins.shape[0]:
            return np.nan
    
        idx_min = max_pos - COM_range
        idx_max = max_pos + COM_range
        COM = np.nansum(hist[idx_min:idx_max]*bins[idx_min:idx_max]) / np.nansum(hist[idx_min:idx_max])
        return imgs_in - COM
        
    elif len(imgs_in.shape) == 3:
        kwargs = {'bins':bins}
        hist = np.apply_along_axis(getHist2, 1, imgs_in.reshape(len(imgs_in),-1), **kwargs)
        max_pos = np.apply_along_axis(np.argmax, 1, hist)
            
        idx_min = max_pos - COM_range
        idx_max = max_pos + COM_range
        COM = []
        for hh, imin, imax in zip(hist, idx_min, idx_max):
            if imax > len(hh):
                COM.append(np.nan)
            else:
                COM.append( np.nansum(hh[imin:imax]*bins[imin:imax]) / np.nansum(hh[imin:imax]) )

        imgs_CMcorr = [img - c for img, c in zip(imgs_in, COM)]
    return imgs_CMcorr
#    return img_CMcorr


@delayed
def CM_corr_single(img_in, int_max=3, COM_range=8):
    bins = np.arange(-5,int_max,0.05)
    hist = getHist2(img_in, bins=bins)
    max_pos = np.argmax(hist)
    
#    max_pos = max_pos.compute()
    
    if max_pos+COM_range+1 > bins.shape[0]:
        return np.nan
    
    idx_min = max_pos - COM_range
    idx_max = max_pos + COM_range
    COM = np.nansum(hist[idx_min:idx_max]*bins[idx_min:idx_max]) / np.nansum(hist[idx_min:idx_max])
    img_CMcorr = img_in - COM
    return img_CMcorr
#    return COM


@delayed
def I0_JF(I0imgs, roi=[[1530,2047],[0,1023]]):
    """ takes corrected images as input for the moment. If the JF IO modules will 
    be moved to a separate detector, then is may be a good idea to implement in this function
    the JFImages class and corrections for the I0 
    REMARK: nan pixel are being ignored """
    
    return roi_integrate(I0imgs, roi)


@delayed
def stripsel_ana(img_in):
    """ Analyze data of the stripsel JF detector (incomplete: still needs to integrate etc) """
    if len(img_in.shape)== 3:
        if isinstance(img_in, da.Array):
            img_in = da.nanmean(img_in, axis=0)
            img_in = img_in.compute()
        else:
            img_in = np.nanmean(img_in, axis=0)
    img_corr = correct_stripeJF(img_in)
    return {'img_corr':img_corr, 'img_init':img_in}


@delayed
def correct_stripeJF(imgin):
    """ Correct the pixel positions for the stripsel detector """
#    imgout=np.zeros((86,(1024*3+18)),dtype=int)
    imgout=np.zeros((86,(1024*3+18)),dtype=float)
#     256 not divisible by 3, so we round up
#     18 since we have 6 more pixels in H per gap

#     first we fill the normal pixels, the gap ones will be overwritten later
    for yin in range(256) :
          for xin in range(1024) :
              ichip=int(xin/256)
              xout=(ichip*774)+(xin%256)*3+yin%3
              ## 774 is the chip period, 256*3+6
              yout=int(yin/3)
              imgout[yout,xout]=imgin[yin,xin]
    
    
    # now the gap pixels...
    for igap in range(3) :
          for yin in range(256):
              yout=int(yin/6)*2
              #first the left side of gap
              xin=igap*64+63
              xout=igap*774+765+yin%6
              imgout[yout,xout]=imgin[yin,xin]
              imgout[yout+1,xout]=imgin[yin,xin]
              #then the right side is mirrored
              xin=igap*64+63+1
              xout=igap*774+765+11-yin%6
              imgout[yout,xout]=imgin[yin,xin]
              imgout[yout+1,xout]=imgin[yin,xin]

#              if we want a proper normalization (the area of those pixels is double, so they see 2x the signal)
#              imgout[yout,xout]=imgout[yout,xout]/2 
    return imgout


@delayed
def fluo_ana(imgs_in, roi=None):
    """ Analyze fluo JF detector (redundant with roi_integrate somehow) """
    if roi is None:
        img_size = imgs_in.shape
        if len(img_size)==2:
            roi = [[0,img_size[0]],[0,img_size[1]]]
        elif len(img_size)==2:
            roi = [[0,img_size[1]],[0,img_size[2]]]
    fluo = roi_integrate(imgs_in, roi)
    return fluo


@delayed
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


@delayed
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
    
    if len(image_in.shape) == 2:
        img_roi = image_in[roi[0][0]:roi[0][1], roi[1][0]:roi[1][1]]
        img_bkg_roi = image_in[bkg_roi[0][0]:bkg_roi[0][1], bkg_roi[1][0]:bkg_roi[1][1]]
        if fintersect:
            img_temp_roi = image_in[temp_roi[0][0]:temp_roi[0][1], temp_roi[1][0]:temp_roi[1][1]]
    
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

    return np.transpose(np.asarray([intensity, bkg]))





""" ------------------------- OTHER TOOLS ------------------------- """
def find_last_pedestal(folder):
    files=list(glob.glob( os.path.join(folder,"pedestal_*")) )
    files.sort()
    return files[-1]


def getHist(imgs, bins=np.arange(-2,20,0.05)):
    """ get intensity histogram from a stack of imgs """
    
    if isinstance(imgs, dask.array.Array):
            H = da.histogram(imgs[da.isfinite(imgs)],bins=bins)[0]
            return H,bins
    else:
        imgs = np.asarray(imgs)
        H = np.histogram(imgs[np.isfinite(imgs)],bins)[0]
        return np.asarray(H),bins
    
    
def getHist2(imgs, bins=np.arange(-2,20,0.05)):
    """ get intensity histogram from a stack of imgs """
    
    if isinstance(imgs, dask.array.Array):
            H = da.histogram(imgs[da.isfinite(imgs)],bins=bins)[0]
            return H
    else:
        H = np.histogram(imgs[np.isfinite(imgs)],bins)[0]
        return np.asarray(H)


def load_gain_pedestal(sf_ana):
    if isinstance(sf_ana.gainJF, dict):
        gains = dict()
        for key in sf_ana.gainJF.keys():
            gains[key] = readF(sf_ana.gainJF[key],"gains")
    else:
        gains = readF(sf_ana.gainJF,"gains")
        
    if isinstance(sf_ana.pedestals, dict):
        pedestals = dict()
        for key in sf_ana.pedestals.keys():
            pedestals[key], pixel_mask = readF(sf_ana.pedestals[key],["gains","pixel_mask"])
    else:
        pedestals, pixel_mask = readF(sf_ana.pedestals,["gains","pixel_mask"])
    return gains, pedestals
    
    
def readF(fname,what):
    """ read given field in a h5 file """
    if not isinstance(what,(list,tuple)): what = (what,)
    f = h5py.File(fname,"r")
    ret = [f[w][:] for w in what]
    if len(ret) == 1: ret=ret[0]
    f.close()
    return ret

def readJson(fname):
    import json
    with open(fname) as f:
        a = json.load(f)
    return a


def get_JFeventIds_dirty(dstores, JFid, step_size = 4):
    step_size = 4 # take every step_size Ids
    eventIds = []
    for ii in range(len(dstores[JFid]['eventIds'])):
        event_min = dstores[JFid]['eventIds'][ii][0]
        event_max = dstores[JFid]['eventIds'][ii][-1]
        eventIds_step = np.arange(event_min, event_max+1, step_size, dtype='uint64')
        eventIds.append(eventIds_step)
    return np.hstack(eventIds)


def sum_delayedImgs(delayed_imgList):
    temp = sum(delayed_imgList).compute()
    return np.nansum(temp, axis=0)




















