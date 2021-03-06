#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 14:49:37 2018

@author: samiksadhu
"""

'Some Functions for Feature Computation' 

import numpy as np
import scipy.linalg as lpc_solve
import subprocess
import os
import sys

def dict2Ark(feat_dict,outfile,kaldi_cmd):
    with open(outfile+'.txt','w+') as file:
        for key, feat in feat_dict.items():
            np.savetxt(file,feat,fmt='%.3f',header=key+' [',footer=' ]',comments='')
    cmd=kaldi_cmd+' ark,t:'+outfile+'.txt'+' ark,scp:'+outfile+'.ark,'+outfile+'.scp'
    subprocess.run(cmd, shell=True)
    os.remove(outfile+'.txt')   
    
def ark2Dict(ark,dim,kaldi_cmd):
    
    cmd=kaldi_cmd+ ' ark:'+ark+' ark,t:-'
    proc=subprocess.run(cmd,shell=True,stdout=subprocess.PIPE)
    x=proc.stdout.decode('utf-8')
    feat_count=0
    start=0
    feats=np.empty((0,dim))
    all_feats={}
    fcount=0;
    for line in x.splitlines(): 

        line=line.strip().split()
        if len(line)>=1:
            if line[-1]=='[':
                start=1
                feat_count+=1 #Starting of a feature       
                uttname=line[0]
                feats=np.empty((0,dim))
                fcount+=1;
            if start==1 and line[-1]!='[':
                if line[-1]==']':
                    line=line[0:-1]
                    x=np.array(line).astype(np.float)
                    x=np.reshape(x,(1,len(x)))

                    feats=np.concatenate((feats,x),axis=0)
                    all_feats[uttname]=feats
                    # Refresh everything 
                    start=0
                    feats=np.empty((0,dim))
                else:
                    x=np.array(line).astype(np.float)
                    x=np.reshape(x,(1,len(x)))
                    feats=np.concatenate((feats,x),axis=0)
    print('%s: Tranfered %d utterances from ark to dict' % (sys.argv[0],fcount))  
    return all_feats

def addReverb(sig,reverb):
    out=np.convolve(sig,reverb)
    xxc=np.correlate(sig,out,'valid')
    indM=len(xxc)-np.argmax(xxc)
    out=out[indM:indM+len(sig)]
    return out
    

def getFrames(signal, srate, frate, flength, window):
    '''Generator of overlapping frames

    Args:
        signal (numpy.ndarray): Audio signal.
        srate (float): Sampling rate of the signal.
        frate (float): Frame rate in Hz.
        flength (float): Frame length in second.
        window (function): Window function (see numpy.hamming for instance).

    Yields:
        frame (numpy.ndarray): frame of length ``flength`` every ``frate``
            second.

    '''
    
    
    flength_samples = int(srate * flength)
    frate_samples = int(srate/frate)
    
    if flength_samples % 2 ==0:
        sp_b=int(flength_samples/2)-1
        sp_f=int(flength_samples/2)
        extend=int(flength_samples/2)
    else:
        sp_b=int((flength_samples-1)/2)
        sp_f=int((flength_samples-1)/2)
        extend=int((flength_samples-1)/2)
        
    sig_padded=np.pad(signal,extend,'reflect')
    win = window(flength_samples)
    idx=sp_b;
    
    while (idx+sp_f) < len(sig_padded):
        frame = sig_padded[idx-sp_b:idx + sp_f+1]
        yield frame * win
        idx += frate_samples
        

def spliceFeats(feats,context):
    context=int(context)
    frame_num=feats.shape[0]
    feat_dim=feats.shape[1]
    
    spliced_feats=np.zeros((frame_num,int(feat_dim*(2*context+1))))
    
    feats=np.append(np.zeros((context,feat_dim)),feats,axis=0)
    feats=np.append(feats,np.zeros((context,feat_dim)),axis=0)
    
    for i in range(0,frame_num-context):
        spliced_feats[i,:]=feats[i:i+2*context+1].reshape(-1)
    return spliced_feats
    
def createFbank(nfilters, nfft, srate):
    mel_max = 2595 * np.log10(1 + srate / 1400)
    fwarped = np.linspace(0, mel_max, nfilters + 2)

    mel_filts = np.zeros((nfilters, int(np.floor(nfft / 2 + 1))))
    hz_points = (700 * (10 ** (fwarped / 2595) - 1))
    bin = np.floor((nfft + 1) * hz_points / srate)

    for m in range(1, nfilters + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            mel_filts[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            mel_filts[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])  

    return mel_filts

def computeLpcFast(signal,order):
    #y=np.correlate(signal,signal,'full')
    #y=y[(len(signal)-1):]
    y=np.fft.ifft(np.fft.fft(signal,len(signal))*np.conj(np.fft.fft(signal,len(signal))))
    y=np.real(y)  
    xlpc=lpc_solve.solve_toeplitz(y[0:order],-y[1:order+1])
    xlpc=np.append(1,xlpc)
    gg=y[0]+np.sum(xlpc*y[1:order+2])
    
    #xlpc=np.random.rand(order)
    #gg=1
    return xlpc, gg

def computeModSpecFromLpc(gg,xlpc,lim):
    xlpc[1:]=-xlpc[1:]
    lpc_cep=np.zeros(lim)
    lpc_cep[0]=np.log(np.sqrt(gg))
    lpc_cep[1]=xlpc[1]
    
    for n in range(2,lim):
        aa=np.arange(1,n)/n
        bb=np.flipud(xlpc[1:n])
        cc=lpc_cep[1:n]
        acc=np.sum(aa*bb*cc)
        lpc_cep[n]=acc+xlpc[n]
    return lpc_cep

