import numpy as np
import argparse
import cv2
import time
import os
import matplotlib.pyplot as plt
import scipy.io as sio
from util.opencv_util import *
from rPPG_Extracter import *
import csv
from rPPG_lukas_Extracter import *

matname = "mixed_motion"
data_path = "C:\\Users\\marti\\Downloads\\Data\\mixed_motion"
vl = VideoLoader(data_path + "\\bmp\\" )
fs = 20

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


with open(data_path + '\\reference.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))
pulse_ref = np.array([float(row[1]) for row in data if is_number(row[1])])

rPPG_extracter = rPPG_Extracter()
i=0
timestamps = []
while True:
    frame,should_stop,timestamp = vl.load_frame()
    timestamps.append(timestamp)
    if should_stop:
        break
    # Forehead Only:   
    #rPPG_extracter.measure_rPPG(frame,False,[.35,.65,.05,.15])
    # Skin Classifier:
    rPPG_extracter.measure_rPPG(frame,True,[])
    

    # Use this if you want lukas Kanade
    #rPPG_extracter.crop_to_face_and_safe(frame)
    #rPPG_extracter.track_Local_motion_lukas()
    #rPPG_extracter.calc_ppg(frame)
    i=i+1 # Progress
    print(i)

rPPG = np.transpose(rPPG_extracter.rPPG)



mat_dict = {"rPPG" : rPPG,"ref_pulse_rate" : pulse_ref}
sio.savemat(matname,mat_dict)

t = np.arange(rPPG.shape[1])/fs

plt.figure()
plt.plot(t,1/np.array(timestamps),'-r')
plt.xlabel("Time (sec)")
plt.ylabel("FPS")


plt.figure()
plt.plot(t,rPPG[0],'-r')
plt.plot(t,rPPG[1],'-g')
plt.plot(t,rPPG[2],'-b')
plt.xlabel("Time (sec)")
plt.ylabel("Amplitude ")

plt.figure()
plt.plot(t,pulse_ref[0:t.shape[0]],'-r')
plt.xlabel("Time (sec)")
plt.ylabel("Pulse_rate (BPM) ")
plt.grid(1)
plt.show()
