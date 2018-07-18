#!/export/b18/ssadhu/tools/python/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:52:48 2018

@author: samiksadhu
"""

'Generate lip features from video data'

import numpy as np 
import cv2
import dlib
import sys

from features import dict2Ark

import argparse

class TooManyFaces(Exception):
    pass
class NoFaces(Exception):
    pass

def get_args():
    parser = argparse.ArgumentParser('Extract lip features from video files')
    parser.add_argument('scp', help='wav.scp with all video files')
    parser.add_argument('outfile', help='output file')
    parser.add_argument('predictor_path', help='path to trained predictor file')
    parser.add_argument('--dim', type=int, default=50, help='High and width of the final image')
    parser.add_argument('--kaldi_cmd', help='Kaldi command to use to get ark files')
    args = parser.parse_args()
    
    return args

def get_landmarks(image,detector,predictor):
    if image is not None:
        rects=detector(image,1)
#        if len(rects) > 1:
#            raise TooManyFaces
        
#        if len(rects)==0:
#            raise NoFaces
        
        return np.matrix([[p.x, p.y] for p in predictor(image, rects[0]).parts()])

def annotate_landmarks(image,landmarks):
    # Basically puts the landmark locations on an image
    if image is not None:
        image=image.copy()
        for idx, point in enumerate(landmarks):
            pos=(point[0,0], point[0,1])
            cv2.putText(image,str(idx),pos,fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,fontScale=0.4,color=(0,0,255))
            cv2.circle(image,pos,3,color=(0,255,255))
        return image

def get_lip_feature(video,dim,detector,predictor):
    
    cap=cv2.VideoCapture(video)
    feats=np.empty([0,dim*dim])
    ret=True
    while(ret):
        ret, image=cap.read()
        
        if image is not None: 
            landmarks=get_landmarks(image,detector,predictor)
            landmarks = np.squeeze(np.asarray(landmarks)) 
            lips=landmarks[48:67,:]
            x_mean=np.mean(lips[:,0]); 
            y_mean=np.mean(lips[:,1]); 

            margin=2 #pixels
            x_max=np.max(abs(lips[:,0]-x_mean))+margin; y_max=np.max(abs(lips[:,1]-y_mean))+margin
            # only lip part of image

            image_crop = image[int(y_mean-y_max):int(y_mean+y_max), int(x_mean-x_max):int(x_mean+x_max)]
            image_crop=cv2.resize(image_crop,(dim,dim))
            image_gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
            img=image_gray.flatten().astype(float)/np.power(2,8)
            feats=np.vstack((feats,img))
    return feats

def get_features(scp,dim,detector,predictor):
    with open(scp, 'r') as fid:   
        all_feats={}   
        for line in fid:
            tokens = line.strip().split()
           
            uttid, inwav = tokens[0], ' '.join(tokens[1:])  
            print('Computing lip features for utterance: %s' % (uttid)) 
            sys.stdout.flush()
            all_feats[uttid]=get_lip_feature(inwav,dim,detector,predictor)
        return all_feats
       
if __name__=='__main__':
    
    print('%s: Starting..' % sys.argv[0])
    args=get_args()
    
    PREDICTOR_PATH= args.predictor_path 
    predictor=dlib.shape_predictor(PREDICTOR_PATH)
    detector=dlib.get_frontal_face_detector()
    
    print('%s: Extracting lip features' % sys.argv[0])
    sys.stdout.flush()
    all_feats=get_features(args.scp,args.dim,detector,predictor)
    print('%s: Finished extracting lip features, saving them as ark file' % sys.argv[0])
    sys.stdout.flush()
    dict2Ark(all_feats,args.outfile,args.kaldi_cmd)
    print('%s: Finished saving lip features' % sys.argv[0])
    sys.stdout.flush()