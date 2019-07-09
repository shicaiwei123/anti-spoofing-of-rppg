import numpy as np
import argparse
import cv2
import time
import os
import matplotlib.pyplot as plt
import scipy.io as sio
from util.opencv_util import *
from rPPG_preprocessing import *
import math


class rPPG_Lukas_Extracter():
    def __init__(self):
        self.prev_face = [0,0,0,0]
        self.skin_prev = []
        self.rPPG = []
        self.cropped_gray_frames = []
        self.frame_cropped = []
        self.flow_frames = 8
        self.points = []
        self.lk_params = dict( winSize  = (15,15),maxLevel = 2,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    def crop_to_face_and_safe(self,frame): 
        self.cropped_gray_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        self.frame_cropped = frame    
    
    def calc_ppg(self,frame):
        try:
            x_region = np.zeros(0,dtype = int)
            y_region = np.zeros(0,dtype = int)

            for point_id in range(self.points.shape[0]):
                x= int(self.points[point_id,0,1])
                y = int(self.points[point_id,0,0])
                margin = 7
                x_region= np.concatenate((x_region,np.arange(x-margin,x+margin,dtype = int)))
                y_region = np.concatenate((y_region,np.arange(y-margin,y+margin,dtype = int)))
           
            X,Y = np.meshgrid(x_region,y_region)
            tracked_section = frame[X,Y,:] 
            print(tracked_section.shape)
            npix = tracked_section.shape[0] * tracked_section.shape[1] 
            r_avg = np.sum(tracked_section[:,:,0])/npix
            g_avg = np.sum(tracked_section[:,:,1])/npix
            b_avg = np.sum(tracked_section[:,:,2])/npix
            ppg = [r_avg,g_avg,b_avg]
            for i,col in enumerate(ppg):
                if math.isnan(col):
                    ppg[i] = 0 
            self.rPPG.append(ppg)
        except Exception:
            self.rPPG.append(rPPG[-1])
        
        
    def track_Local_motion_lukas(self):        
        h, w = self.cropped_gray_frames[-1].shape[:2]
        if len(self.points) == 0:
            self.points = np.zeros((1,1,2),dtype = np.float32)
            self.points[0,0,0] = 380
            self.points[0,0,1] = 280
        num_frames = len(self.cropped_gray_frames)

        if num_frames > 1:
             self.points, st, err = cv2.calcOpticalFlowPyrLK(self.cropped_gray_frames[-2], self.cropped_gray_frames[-1], self.points, None, **self.lk_params)

             #else:
        #    feature_params = dict( maxCorners = 100,
        #               qualityLevel = 0.3,
        #               minDistance = 7,
        #               blockSize = 7 )
        #    self.points = cv2.goodFeaturesToTrack(self.cropped_gray_frames[-1], mask = None, **feature_params)


   

                
    

# cleanup the camera and close any open windows




