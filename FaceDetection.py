import cv2
import pwd
import os
from skimage.feature import local_binary_pattern
from sklearn.externals import joblib

os.seteuid(pwd.getpwnam(os.getlogin()).pw_uid)

import numpy as np
from scipy.fftpack import fft
import math

import scipy.io as sio
import datetime
import dlib
import copy
from matplotlib import pyplot as plt


class c_face_detection():
    """
    人脸检测类，检测人脸，人脸图像预处理
    """

    def __init__(self):
        """
        构造函数，初始化参数
        """
        self.picture = None
        self.a_face_picture = None
        self.a_face_profile = None
        PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        self.a_detector = dlib.get_frontal_face_detector()
        self.a_predictor = dlib.shape_predictor(PREDICTOR_PATH)
        self.a_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def face_detection(self):
        """
        检测人脸，返回拍摄有人脸的图片
        :return:
        """
        cap = cv2.VideoCapture(0)

        def face_detect():
            while 1:
                face_detection_number = 0
                # get a frame
                ret, frame = cap.read()
                img = frame
                begin = datetime.datetime.now()
                img_key = img.copy()
                detector = self.a_detector
                predictor = self.a_predictor

                # 预处理
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # 人脸检测
                rects = detector(gray, 1)

                # 显示
                cv2.imshow("face", gray)
                if len(rects) == 0:
                    face_detection_number = 0
                else:
                    face_detection_number += 1

        while (1):

            # get a frame
            ret, frame = cap.read()
            img = frame
            begin = datetime.datetime.now()
            img_key = img.copy()
            detector = self.a_detector
            predictor = self.a_predictor

            # 预处理
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # 人脸检测
            rects = detector(gray, 1)

            # 显示
            cv2.imshow("face", gray)
            if len(rects) == 0:
                face_detection_number = 0
            else:
                face_detection_number += 1

            # 连续10帧人脸就认为是检测到了
            if face_detection_number > 10:
                break
            cv2.waitKey(0)
        points_keys = []
        # 特征点检测,只取第一个,也就是最大的一个
        landmarks = np.matrix([[p.x, p.y] for p in predictor(gray, rects[0]).parts()])

        # 特征点提取,标注
        for idx, point in enumerate(landmarks):
            # pos = (point[0,0],point[0,1])
            points_keys.append([point[0, 0], point[0, 1]])
            # cv2.circle(img_key,pos,2,(255,0,0),-1)

        # 求端点,分割处人脸
        key = points_keys
        left = key[3][0]  # 第一个特征点的纵坐标,也就是宽的方向
        right = key[13][0]  # 第17个特征点的纵坐标,也就是宽方形
        top = key[20][1] - 10  # 第21个点的横坐标,也即是高的起始
        bottom = key[8][1] - 20  # 第9个点的横坐标
        img_roi = img[top:bottom, left:right]

        end = datetime.datetime.now()
        time_sub = end - begin
        # print("人脸截取时间:",time_sub.total_seconds())
        return img_roi
        # release camera
        cap.release()
        cv2.destroyAllWindows()

    def face_detetion_picture(self, p_img, model='process'):

        # 全景图片人脸检测
        # 初始化
        begin = datetime.datetime.now()
        img = p_img
        points_keys = []
        img_key = img.copy()
        detector = self.a_detector
        predictor = self.a_predictor

        # 预处理和实际使用的时候,这个边界条件设置的不一样,是为了更好的限制输入
        # 同时也为了预处理的时候不至于丢掉太多图片
        if model == 'process':
            threshold = 130
        elif model == 'test':
            threshold = 120
        else:
            threshold = 70

        # 预处理
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 人脸检测
        rects = detector(gray, 1)
        if len(rects) == 0:
            # cv2.imshow("none", gray)
            # cv2.waitKey(0)
            return None
        # 特征点检测,只取第一个,也就是最大的一个
        landmarks = np.mat([[p.x, p.y] for p in predictor(gray, rects[0]).parts()])

        # 特征点提取,标注
        for idx, point in enumerate(landmarks):
            # pos = (point[0,0],point[0,1])
            points_keys.append([point[0, 0], point[0, 1]])
            # cv2.circle(img_key,pos,2,(255,0,0),-1)

        # 求端点,分割处人脸
        key = points_keys
        left = key[3][0]  # 第一个特征点的纵坐标,也就是宽的方向
        right = key[13][0]  # 第17个特征点的纵坐标,也就是宽方形
        distance = key[27][1] - key[21][1]  # 使用相对间距而不是绝对间距来删选
        top = key[19][1] - distance  # 第21个点的横坐标,也即是高的起始
        bottom = int((key[7][1] + key[9][1]) / 2)  # 第8个点的横坐标
        img_roi = img[top:bottom, left:right]

        # print(bottom - top, right - left)

        # 删除过于小尺寸的,有一个边小于70,就删除
        if bottom - top < threshold or right - left < threshold:
            return None

        # 删除过大的
        if bottom - top > 330 or right - left > 280:
            return None

        end = datetime.datetime.now()
        time_sub = end - begin
        # print("人脸截取时间:",time_sub.total_seconds())

        # 显示截取之后的人脸
        # cv2.imshow("face",img_roi)
        # cv2.waitKey(0)

        return img_roi

    def face_profile(self, p_img=None):
        if p_img is None:
            img = self.a_face_picture
        else:
            img = p_img
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 把图像转换到HSV色域
        (_h, _s, _v) = cv2.split(hsv)  # 图像分割, 分别获取h, s, v 通道分量图像
        skin3 = np.zeros(_h.shape, dtype=np.uint8)  # 根据源图像的大小创建一个全0的矩阵,用于保存图像数据
        (x, y) = _h.shape  # 获取源图像数据的长和宽

        # 遍历图像, 判断HSV通道的数值, 如果在指定范围中, 则置把新图像的点设为255,否则设为0
        begin = datetime.datetime.now()
        # for i in range(0, x):
        #     for j in range(0, y):
        #         if (_h[i][j] > 7) and (_h[i][j] < 20) and (_s[i][j] > 28) and (_s[i][j] < 255) and (_v[i][j] > 50) and (
        #                 _v[i][j] < 255):
        #             skin3[i][j] = 255
        #         else:
        #             skin3[i][j] = 0

        # _h[_h>7]

        end = datetime.datetime.now()
        sub = end - begin
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))

        # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
        erosion = cv2.erode(skin3, element1, iterations=1)
        c = erosion[100, :]
        begin = datetime.datetime.now()
        for i in range(x):
            if 255 in erosion[i, :]:
                high = i
            if 255 in erosion[x - i - 1, :]:
                low = 254 - i

            if 255 in erosion[:, i]:
                left = i
            if 255 in erosion[:, x - i - 1]:
                right = 254 - i
        end = datetime.datetime.now()
        sub = end - begin
        face_img = img[low:high, right:left]
        self.a_face_profile = face_img
        return face_img
        # rege_img=cv2.rectangle(img, (right,low), (left,high), (0, 255, 0), 2)  # 用矩形圈出人脸
        # cv2.imshow("imname", img)
        # cv2.imshow("bbb", face_img)
        # cv2.imshow("aa", erosion)
        # cv2.imshow(" Skin3 HSV", skin3)
        # cv2.waitKey(0)

    def get_face_picture(self):
        return self.a_face_picture

    def face_alignment(self, picture):

        # 初始化
        behin = datetime.datetime.now()
        predictor = self.a_predictor
        face_cascade = self.a_face_cascade
        faces_aligned = []

        # 转灰
        gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(5, 5))

        # 如果找不到人脸
        if len(faces) == 0:
            cv2.imshow("none", gray)
            cv2.waitKey(0)

        face = None
        for (x, y, w, h) in faces:
            # 边界判断
            if x < 80:
                left = 0
            else:
                left = x - 80

            if 640 - (x + w) < 80:
                right = 640
            else:
                right = x + w + 80

            if y < 80:
                top = 0
            else:
                top = y - 80

            if 480 - (y + h) < 80:
                botoom = 480
            else:
                botoom = y + h + 80
            face = picture[top:botoom, left:right]  # 扩大范围截取图片

        # 如果没有圈到人脸,则返回
        if face is None:
            return None

        # 求角度
        rec = dlib.rectangle(0, 0, face.shape[0], face.shape[1])
        shape = predictor(np.uint8(face), rec)  # 注意输入的必须是uint8类型
        order = [36, 45, 30, 48, 54]  # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找

        eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2,  # 计算两眼的中心坐标
                      (shape.part(36).y + shape.part(45).y) * 1. / 2)
        dx = (shape.part(45).x - shape.part(36).x)  # note: right - right
        dy = (shape.part(45).y - shape.part(36).y)

        angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度

        # 矫正
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
        RotImg = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))  # 进行放射变换，即旋转
        faces_aligned.append(RotImg)
        # 计算时间
        end = datetime.datetime.now()
        time_sub = end - behin
        # print("人脸对齐时间:", time_sub.total_seconds())

        # 返回
        return faces_aligned


if __name__ == '__main__':
    # 人脸检测
    while 1:
        o_face_detection = c_face_detection()
        o_face_detection.face_detection()
        face_picture = o_face_detection.get_face_picture()
        size = (64, 64)
        # picture_resize = cv2.resize(face_picture, size)
        picture_resize = face_picture
        # cv2.imshow("resize", picture_resize)
        # cv2.waitKey(0)
