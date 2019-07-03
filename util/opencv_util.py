import cv2
import os
import time
class VideoLoader():    
    def __init__(self,folder):
        self.frame = 0
        self.video_folder = folder

    def load_frame(self):
        self.frame+=1
        frame_path = self.video_folder + str(self.frame) + ".bmp"
        exists = os.path.isfile(frame_path)
        if exists:
            return cv2.imread(frame_path),False,self.frame/20
        else:
            print(frame_path + " Does not exist")
            return None,True,0


cascPath = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascPath)

def write_text(img,text,location):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    cv2.putText(img,text,location,font,fontScale,fontColor,lineType)

def write_fps(skin, start_time):
    fps = 1/(time.time() - start_time)
import numpy as np

def draw_tracking_dots(img,x_track,y_track,offset,error, step=16):
    
    
    lines = np.vstack([x_track + offset[0], y_track + offset[1],error]).T.reshape(-1, 3)
    lines = np.int32(lines + 0.5)
    for (x1, y1,e) in lines:
        col = (0,255,0)
        if e > 10:
            col = (0,0,255)

        cv2.circle(img, (x1, y1), 1, col, -1)


def draw_flow(img, flow,offset, step=16):
    h, w = flow.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
    fx, fy = flow[y,x].T
    y += offset[1]
    x += offset[0]
    
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    cv2.polylines(img, lines, 0, (0, 255, 0))
    for (x1, y1), (_x2, _y2) in lines:
        cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
    

def draw_rect(frame,rect):
    rects = [rect]
    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

def make_border(frame,rect):
    border = cv2.imread('util/Border.png',cv2.IMREAD_UNCHANGED)
       
    rows,cols,channels = border.shape

    x =  0 #rect[0]
    y =  0#rect[1]
    w =  cols #rect[2]
    h =  rows#rect[3]
    bg_part = frame[y:y+h,x:x+w]
    border_rgb = border[:,:,0:3]
    
    alpha= border[:,:,3] 
    alpha = alpha[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha = np.concatenate((alpha,alpha,alpha), axis=2)
    border_rgb = border_rgb.astype(np.float32) * alpha
    bg_part=bg_part.astype(np.float32) * (1-alpha)
    merged = bg_part + border_rgb
    frame[y:y+h,x:x+w] = merged.astype(np.uint8)
            
