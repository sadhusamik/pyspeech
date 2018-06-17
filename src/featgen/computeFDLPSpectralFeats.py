#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:09:08 2018

@author: samiksadhu
"""

'Computing FDLP Spectral Features'

import sys
import argparse
import io
import numpy as np
import os
from scipy.io.wavfile import read
import subprocess
import scipy.fftpack as freqAnalysis 
from scipy.signal import freqz

from features import getFrames,createFbank,computeLpcFast


def extractFDLPSpectrum(wavs, outdir, phone_map, phn_file_dir, get_phone_labels=False, only_center=False, ignore_edge=False, order=50, fduration=0.5, frate=100,
                            nfft=512, nfilters=30, srate=16000,
                            window=np.hanning):
    
    if not only_center:
        fbank = createFbank(nfilters, int(2*fduration*srate), srate)
        
        # Get list of phonemes
        phn_list=[]
        
        with open(phone_map,'r') as fid2:
            for line2 in fid2:
                line2=line2.strip().split()
                if len(line2)==2:
                    if 'sil' not in line2 and 'SIL' not in line2:
                        phn_list.append(line2[1])
                        
        phn_list=list(set(phn_list))
        phn_list.sort()              
                
        with open(wavs, 'r') as fid:
            
            # Initialize matrix for all features
            if get_phone_labels:
                all_feats=np.empty(nfilters+1)
            else:
                all_feats=np.empty(nfilters)
                    
            for line in fid:
                tokens = line.strip().split()
                uttid, inwav = tokens[0], ' '.join(tokens[1:])
                    
                fname_phn=uttid+'.PHN'
                
                if get_phone_labels:
                    if os.path.isfile(os.path.join(phn_file_dir,fname_phn)):
                        # Get first line of phone file in the beginning 
                        phn_file=open(os.path.join(phn_file_dir,fname_phn))
                        phn_line=phn_file.readline()
                        phn_locs=phn_line.strip().split()
                        # Get phoneme information 
                        phone_now=phn_locs[2] 
                        phone_end=int(int(phn_locs[1]))
                        beg_frame=int(int(phn_locs[0]))
                    else:
                        break
                else:
                    beg_frame=0
                    
                if inwav[-1] == '|':
                    proc = subprocess.run(inwav[:-1], shell=True,
                                          stdout=subprocess.PIPE)
                    sr, signal = read(io.BytesIO(proc.stdout))

                else:
                    sr, signal = read(inwav)
                    
                assert sr == srate, 'Input file has different sampling rate.'
                # I want to work with numbers from 0 to 1 so.... 
                signal=signal/np.power(2,15)
                
                time_frames = np.array([frame for frame in
                    getFrames(signal, srate, frate, fduration, window)])
    
                cos_trans=freqAnalysis.dct(time_frames)/np.sqrt(2*int(srate * fduration))
                [frame_num, ndct]=np.shape(cos_trans)
                
                if get_phone_labels:
                    feats=np.zeros([frame_num,nfilters+1])
                else:
                    feats=np.zeros([frame_num,nfilters])
                    
                print('Computing Features for file: %s' % uttid)
                
                for i in range(beg_frame,frame_num):
                    
                    each_feat=np.zeros(nfilters)
                    for j in range(nfilters):
                        filt=fbank[j,0:-1]
                        band_dct=filt*cos_trans[i,:]
                        xlpc, gg=computeLpcFast(band_dct,order) # Compute LPC coefficients 
                        w, h=freqz(np.sqrt(gg),xlpc,ndct)
                        h_mid=np.log10(np.mean(np.abs(h[int(ndct/2-160):int(ndct/2+160)])))
                        each_feat[j]=h_mid
                     
                    if get_phone_labels:
                    # Updates to current phoneme
                        if i>phone_end:                
                            # Get new phone label
                            phn_line=phn_file.readline()
                            if phn_line:
                                phn_locs=phn_line.strip().split()
                                phone_now=phn_locs[2] 
                                phone_end=int(phn_locs[1])
                                phone_end=int(int(phn_locs[1])/160)
                            else:
                                break # Break if no more phones are remaining
                            
                        ind=phn_list.index(phone_now)
                        each_feat=np.append(each_feat,ind)
                    feats[i,:]=each_feat
                        
                
                all_feats=np.vstack([all_feats, feats])
            all_feats=all_feats[1:,:]
            
            # Save the final BIG feature file 
            np.save(os.path.join(outdir), all_feats)
            np.save(os.path.join(os.path.dirname(outdir), 'phone_list'), phn_list)
    else:
        fbank = createFbank(nfilters, int(2*fduration*srate), srate)
        
        # Get list of phonemes
        phn_list=[]
        
        with open(phone_map,'r') as fid2:
            for line2 in fid2:
                line2=line2.strip().split()
                if len(line2)==2:
                    if 'sil' not in line2 and 'SIL' not in line2:
                        phn_list.append(line2[1])
                        
        phn_list=list(set(phn_list))
        phn_list.sort()
        
        with open(wavs, 'r') as fid:
            # Initialize matrix for all features
            if get_phone_labels:
                all_feats=np.empty(nfilters+1)
            else:
                all_feats=np.empty(nfilters)
                    
            for line in fid:
                tokens = line.strip().split()
                uttid, inwav = tokens[0], ' '.join(tokens[1:])
                
                if inwav[-1] == '|':
                    proc = subprocess.run(inwav[:-1], shell=True,
                                          stdout=subprocess.PIPE)
                    sr, signal = read(io.BytesIO(proc.stdout))
                else:
                    sr, signal = read(inwav)
                assert sr == srate, 'Input file has different sampling rate.'
                # I want to work with numbers from 0 to 1 so.... 
                signal=signal/np.power(2,15)
                
                fname_phn=uttid+'.PHN'
                

                # Get all phones and their center
                
                if os.path.isfile(os.path.join(phn_file_dir,fname_phn)):
                    phn_file=open(os.path.join(phn_file_dir,fname_phn))
                    phone_mid=np.empty(0)
                    phone_now=np.empty(0)
                    for line2 in phn_file:
    
                        phn_locs=line2.strip().split() 
                        if phn_locs[2] in phn_list:
                            ind=phn_list.index(phn_locs[2])
                            phone_now=np.append(phone_now, ind) 
                            phone_mid=np.append(phone_mid
                            ,int(int(phn_locs[0])+int(phn_locs[1]))/2)
                    
                    time_frames = np.array([frame for frame in
                        getFrames(signal, srate, frate, fduration, window)])
    
                    cos_trans=freqAnalysis.dct(time_frames)/np.sqrt(2*int(srate * fduration))
                    
                    [frame_num, ndct]=np.shape(cos_trans)
                    
                    if ignore_edge:
                        phone_mid=phone_mid[1:-1]
                        phone_now=phone_now[1:-1]
                        
                    only_compute=len(phone_mid)
                     
                    feats=np.zeros([only_compute,nfilters])     
                    
                    print('Computing Features for file: %s' % uttid)
                    
                    for kk in range(only_compute):
                        i=int(np.floor((phone_mid[kk])))
                    
                        each_feat=np.zeros(nfilters)
                        for j in range(nfilters):
                            filt=fbank[j,0:-1]
                            band_dct=filt*cos_trans[i,:]
                            xlpc, gg=computeLpcFast(band_dct,order) # Compute LPC coefficients 
                            w, h=freqz(np.sqrt(gg),xlpc,ndct)
                            h_mid=np.log10(np.mean(np.abs(h[int(ndct/2-160):int(ndct/2+160)])))
                            each_feat[j]=h_mid
                        
                        feats[kk,:]=each_feat
                    if get_phone_labels:
                        feats=np.append(feats,np.reshape(phone_now,(len(phone_now),1)),axis=1)
                    all_feats=np.vstack([all_feats, feats])
                
                
            all_feats=all_feats[1:,:]
            
            # Save the final BIG feature file 
            np.save(os.path.join(outdir), all_feats)
            np.save(os.path.join(os.path.dirname(outdir), 'phone_list'), phn_list)
                
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Extract FDLP Spectral Features.')
    parser.add_argument('scp', help='"scp" list NOTE: Code assumes that the .PHN files have the same locations as in the scp file')
    parser.add_argument('outdir', help='output directory')
    parser.add_argument('phn_file', help='Phone map file')
    parser.add_argument('phn_file_dir', help='Location of all phone files')
    parser.add_argument('nfilters', type=int, default=15,help='number of filters (15)')
    parser.add_argument('order', type=int, default=50,help='LPC filter order (50)')
    parser.add_argument('fduration', type=float, default=0.5,help='Window length (0.5 sec)')
    parser.add_argument('frate', type=int, default=100,help='Frame rate (100 Hz)')
    parser.add_argument('--get_phone_labels', action='store_true',
                        help='get phone labels for each feature attached as another column to feature matrix(True)')
    parser.add_argument('--only_center', action='store_true',
                        help='get features from only the center of each phonemes for each utterance (True)') 
    parser.add_argument('--ignore_edge', action='store_true',
                        help='Ignore the phonemes at the edges of a phone file (False)')
    args = parser.parse_args()
    
    print('%s: Extracting features....' % sys.argv[0])
    extractFDLPSpectrum(args.scp, args.outdir, args.phn_file, args.phn_file_dir, args.get_phone_labels,args.only_center,args.ignore_edge,args.order,args.fduration,args.frate)
