import numpy as np
import argparse
import cv2
import time
import os
import matplotlib.pyplot as plt
import scipy.io as sio
from VideoHealthMonitoring.util.opencv_util import *
from VideoHealthMonitoring.rPPG_preprocessing import *
from VideoHealthMonitoring.FaceDetection import c_face_detection
import math
import dlib
import datetime


# 拍照，对于每一帧，检测人脸，统计人脸照片的rgb空间的分量，求均值，然后缓存得到rppg

class rPPG_Extracter():
    def __init__(self):
        self.prev_face = [0, 0, 0, 0]  # 脸部脚点,随机赋值
        self.skin_prev = []
        self.rPPG = [[],[],[],[],[],[],[],[],[],[]]
        self.rPPG_right = []
        self.sub_roi_rect = []
        PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        self.a_detector = dlib.get_frontal_face_detector()
        self.a_predictor = dlib.shape_predictor(PREDICTOR_PATH)

    def calc_ppg(self, num_pixels, frame):
        '''
        # 求不同颜色分量的均值。并保存，返回
        :param num_pixels:  读取的图片的像素点，而不是感兴趣的区域的像素的。
        :param frame:  人脸图像
        :return:
        '''

        if num_pixels == 0:
            ppg = [0, 0, 0]
        else:
            r_avg = np.sum(frame[:, :, 0]) / num_pixels
            g_avg = np.sum(frame[:, :, 1]) / num_pixels
            b_avg = np.sum(frame[:, :, 2]) / num_pixels
            if num_pixels == 0:
                print(num_pixels)
            ppg = [r_avg, g_avg, b_avg]
            for i, col in enumerate(ppg):
                if math.isnan(col):
                    ppg[i] = 0
        # print("one ppg",ppg)
        return ppg

    def get_landmarks(self, gray, face_rect):

        '''
        求关键点
        :param gray:图像灰度图
        :param face_rect:opencv 人脸识别结果
        :return:
        '''
        begin = datetime.datetime.now()

        # 坐标
        coordinate = face_rect
        x1 = coordinate[0]
        y1 = coordinate[1]
        x2 = x1 + coordinate[2]
        y2 = y1 + coordinate[3]

        # 类型转变，opencv_to_dlib
        rect = dlib.rectangle(x1, y1, x2, y2)

        points_keys = []
        # 特征点检测,只取第一个,也就是最大的一个
        landmarks = np.mat([[p.x, p.y] for p in self.a_predictor(gray, rect).parts()])

        # 特征点提取,标注
        for idx, point in enumerate(landmarks):
            # pos = (point[0,0],point[0,1])
            points_keys.append([point[0, 0], point[0, 1]])
            # cv2.circle(img_key,pos,2,(255,0,0),-1)

        return points_keys

    def get_global_face(self, frame, key):
        '''
        利用关键点，获取整张人脸图片
        :param frame:
        :param gray:
        :param key:
        :return:
        '''
        # 求新的面部图片
        left = key[3][0]  # 第一个特征点的纵坐标,也就是宽的方向
        right = key[13][0]  # 第17个特征点的纵坐标,也就是宽方形
        # distance = key[27][1] - key[21][1]  # 使用相对间距而不是绝对间距来删选
        distance = 0
        top = key[19][1] - distance  # 第21个点的横坐标,也即是高的起始
        bottom = key[9][1]  # 第8个点的横坐标

        frame_cropped = frame[top:bottom, left:right]

        cv2.imshow("picture", frame)
        if frame_cropped is not None:
            cv2.imshow("face", frame_cropped)
        # print(self.prev_face)
        return frame_cropped

    def get_local_face(self, frame, key):
        '''
        获取脸局部信息，鼻子以下，两个部位，或者多个部位
        :param frame:
        :param key:
        :return:
        '''
        frame_local = []
        # 坐标
        distance = 0
        x1 = key[29][1] - distance
        x2 = key[9][1]
        y1 = key[3][0]
        y2 = key[30][0]
        y3 = key[13][0]

        frame_cropped = frame[x1:x2, y1:y2]
        frame_local.append(frame_cropped)

        frame_right = frame[x1:x2, y2:y3]
        frame_local.append(frame_right)

        # 绘图
        cv2.imshow("picture", frame)
        if frame_local is not None:
            cv2.imshow("face", frame_local[0])

        return frame_local

    def process_frame_global(self, frame, sub_roi):

        '''
        处理视频帧，提取整个人脸部分并返回
        :param frame: 视频帧，但是还可能是空为什么？？？
        :param sub_roi:
        :return:
        '''

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测,返回人脸，灰色人脸，上次人脸

        frame_cropped, gray_frame, self.prev_face = crop_to_face(frame, gray, self.prev_face)

        # 求关键点
        key = self.get_landmarks(gray, self.prev_face)

        # 利用关键点分割，返回精准人脸
        frame_cropped = self.get_global_face(frame, key)
        frame_cropped=[frame_cropped]           # 转成三维列表和local face统一

        # 用自己规定的区域提取人脸，不过没有用的到
        if len(sub_roi) > 0:
            print("sub_roi")
            sub_roi_rect = get_subroi_rect(frame_cropped, sub_roi)
            frame_cropped = crop_frame(frame_cropped, sub_roi_rect)
            gray_frame = crop_frame(gray_frame, sub_roi_rect)
            self.sub_roi_rect = sub_roi_rect

        num_pixels = frame.shape[0] * frame.shape[1]

        return frame_cropped, num_pixels

    def process_frame_local(self, frame, sub_roi):

        '''
        处理视频帧，提取人脸的多个局部部分
        :param frame: 视频帧，但是还可能是空为什么？？？
        :param sub_roi:
        :return:
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测,返回人脸，灰色人脸，上次人脸

        frame_cropped, gray_frame, self.prev_face = crop_to_face(frame, gray, self.prev_face)

        # 求关键点
        key = self.get_landmarks(gray, self.prev_face)

        # 利用关键点分割，返回多个局部人脸
        frame_cropped = self.get_local_face(frame, key)

        # 用自己规定的区域提取人脸，不过没有用的到
        if len(sub_roi) > 0:
            print("sub_roi")
            sub_roi_rect = get_subroi_rect(frame_cropped, sub_roi)
            frame_cropped = crop_frame(frame_cropped, sub_roi_rect)
            gray_frame = crop_frame(gray_frame, sub_roi_rect)
            self.sub_roi_rect = sub_roi_rect

        num_pixels = frame.shape[0] * frame.shape[1]

        return frame_cropped, num_pixels

    def measure_rPPG(self, frame, use_classifier=False, sub_roi=[]):
        '''
        传入图片，测量rppg，返回的是一个二维list，不断累计
        :param frame: 视频帧
        :param use_classifier:
        :param sub_roi: 感兴趣区域的脚点，也就是人脸区域，不过一般都是自动确定而不是人为指定
        :return:
        '''
        # frame_cropped, num_pixels = self.process_frame_global(frame, sub_roi)
        frame_cropped, num_pixels = self.process_frame_local(frame, sub_roi)

        face_num = len(frame_cropped)
        for i in range(face_num):
            face_data=frame_cropped[i]
            self.rPPG[i].append(self.calc_ppg(num_pixels, face_data))

        # fb=open('./rppg.txt','a+')
        # fb.write(str(self.rPPG))
        # fb.write('\n')
        # fb.close()
