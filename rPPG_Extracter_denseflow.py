import numpy as np
import argparse
import cv2
import time
import os
import matplotlib.pyplot as plt
import scipy.io as sio
from util.opencv_util import *
from rPPG_preprocessing import *
from csk_facedetection import CSKFaceDetector
import math
class rPPG_Extracter():
    def __init__(self):
        self.prev_face = [0,0,0,0]
        self.skin_prev = []
        self.rPPG = []
        self.cropped_gray_frames = []
        self.frame_cropped = []
        self.flow_frames = 8
        self.x_track = []
        self.y_track = []
        self.dx = []
        self.dy = []
        self.error = []
        self.sub_roi_rect = []
        self.csk_tracker = CSKFaceDetector()
        
        
    def calc_ppg(self,num_pixels,frame):
        
        r_avg = np.sum(frame[:,:,0])/num_pixels
        g_avg = np.sum(frame[:,:,1])/num_pixels
        b_avg = np.sum(frame[:,:,2])/num_pixels
        ppg = [r_avg,g_avg,b_avg]
        for i,col in enumerate(ppg):
            if math.isnan(col):
                ppg[i] = 0
        return ppg

    def process_frame(self, frame, sub_roi, use_classifier,use_csk=False):
        gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
       
        if use_csk:
            frame_cropped,gray_frame,self.prev_face = self.csk_tracker.track_face(frame,gray_frame)
        else:
            frame_cropped,gray_frame,self.prev_face = crop_to_face(frame,gray_frame,self.prev_face)

        if len(sub_roi) > 0:
            sub_roi_rect = get_subroi_rect(frame_cropped,sub_roi)
            frame_cropped = crop_frame(frame_cropped,sub_roi_rect)
            gray_frame = crop_frame(gray_frame,sub_roi_rect)
            self.sub_roi_rect = sub_roi_rect

        num_pixels = frame.shape[0] * frame.shape[1]
        if use_classifier:
            frame_cropped,num_pixels = apply_skin_classifier(frame_cropped)
        return frame_cropped, gray_frame,num_pixels

    def measure_rPPG(self,frame,use_classifier = False,sub_roi = []): 
        frame_cropped, gray_frame,num_pixels = self.process_frame(frame, sub_roi, use_classifier)
        self.rPPG.append(self.calc_ppg(num_pixels,frame_cropped))
        self.frame_cropped = frame_cropped
    
    def measure_rPPG_delta_saturated(self,frame): 
        #frame_cropped, gray_frame,num_pixels = self.process_frame(frame, [], True)
        #ppg_val = self.calc_ppg(num_pixels,frame_cropped)
        
        #if np.sum(len(self.rPPG) > 1 and np.array(ppg_val) - np.array(self.rPPG[-1])) > 30:
        #     print("Delta TOO LARGE!")
        #     print("Suggested" + str(self.prev_face) + " Overwritted : " + str(self.back_up_face))
        #     frame_cropped = crop_frame(frame,self.back_up_face)
        #     self.prev_face = self.back_up_face
        #     frame_cropped,num_pixels = apply_skin_classifier(frame_cropped)
        #     ppg_val = self.calc_ppg(num_pixels,frame_cropped)
        #else:
        #   self.rPPG.append(ppg_val)

        #self.back_up_face = self.prev_face
        frame,num_pixels = apply_skin_classifier(frame)
        ppg_val = self.calc_ppg(num_pixels,frame)
        self.rPPG.append(ppg_val)
        self.frame_cropped = frame        


    def find_flow_pixels(self):
        base_flow = np.zeros()
        for frame_id in range(1,self.flow_frames):
            flow = cv2.calcOpticalFlowFarneback(flow_frames[frame_id-1],flow_frames[frame_id], None, 0.5, 3, 15, 3, 5, 1.2, 0)
          
 
    def crop_to_face_and_safe(self,frame,use_classifier = False,sub_roi = []): 
        frame_cropped, gray_frame,_ = self.process_frame(frame, sub_roi, use_classifier)
        self.cropped_gray_frames.append(gray_frame)
        self.frame_cropped = frame_cropped

    def track_Local_motion(self):
        try : 
            h, w = self.cropped_gray_frames[-1].shape[:2]
            step = 16
            if len(self.x_track) == 0:
                self.y_track, self.x_track = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)

            num_frames = len(self.cropped_gray_frames)

            if num_frames > 1:
                flow = cv2.calcOpticalFlowFarneback(self.cropped_gray_frames[-2],self.cropped_gray_frames[-1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                self.x_track = self.x_track.clip(0,w-1)
                self.y_track = self.y_track.clip(0,h-1)
                fx, fy = flow[self.y_track.astype(int),self.x_track.astype(int)].T#.astype(int)
                self.dx.append(fx)
                self.dy.append(fy)
                self.x_track+=fx
                self.y_track+=fy
                #print(self.x_track.shape)

                orig_pos_y,orig_pos_x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
                estimated_pos_x = np.copy(self.x_track)   
            
                estimated_pos_y = np.copy(self.y_track)
                for fr in range(np.min([15,len(self.dx)])):
                    estimated_pos_x += self.dx[-fr] 
                    estimated_pos_y += self.dy[-fr]

                self.error = np.sqrt((estimated_pos_x - orig_pos_x)**2 + (estimated_pos_y - orig_pos_y)**2 )
        except Exception :
            print("Flow Failed")

     #def track_Local_motion_lukas(self):
     #   try : 
     #       h, w = self.cropped_gray_frames[-1].shape[:2]
     #       step = 16
     #       p = [(h/2,w/2)]

     #       num_frames = len(self.cropped_gray_frames)

     #       if num_frames > 1:
     #           flow = cv2.calcOpticalFlowFarneback(self.cropped_gray_frames[-2],self.cropped_gray_frames[-1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
     #           self.x_track = self.x_track.clip(0,w-1)
     #           self.y_track = self.y_track.clip(0,h-1)
     #           fx, fy = flow[self.y_track.astype(int),self.x_track.astype(int)].T#.astype(int)
     #           self.dx.append(fx)
     #           self.dy.append(fy)
     #           self.x_track+=fx
     #           self.y_track+=fy
     #           #print(self.x_track.shape)

     #           orig_pos_y,orig_pos_x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
     #           estimated_pos_x = np.copy(self.x_track)   
            
     #           estimated_pos_y = np.copy(self.y_track)
     #           for fr in range(np.min([15,len(self.dx)])):
     #               estimated_pos_x += self.dx[-fr] 
     #               estimated_pos_y += self.dy[-fr]

     #           self.error = np.sqrt((estimated_pos_x - orig_pos_x)**2 + (estimated_pos_y - orig_pos_y)**2 )
     #   except Exception :
     #       print("Flow Failed")


                
    

# cleanup the camera and close any open windows




