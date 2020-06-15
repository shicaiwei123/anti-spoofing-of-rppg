
from rPPG_preprocessing import *
from FaceDetection import c_face_detection
import math
import dlib
import datetime


# 拍照，对于每一帧，检测人脸，统计人脸照片的rgb空间的分量，求均值，然后缓存得到rppg

class rPPG_Extracter():
    def __init__(self):
        self.prev_face = [0, 0, 0, 0]
        self.skin_prev = []
        self.rPPG = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]  #
        self.rPPG_right = []
        self.sub_roi_rect = []
        PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        self.a_detector = dlib.get_frontal_face_detector()
        self.a_predictor = dlib.shape_predictor(PREDICTOR_PATH)
        self.a_face_detection = c_face_detection()

    def calc_ppg(self, num_pixels, frame):
        '''
        # 求不同颜色分量的均值。并保存，返回   /Find the average of different color components and return
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
        求关键点 / get key points
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
        利用关键点，获取整张人脸图片   / Use key points to get the entire face picture
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

    def coor_to_face(self, x_list, y_list, frame):
        x_len = len(x_list)
        y_len = len(y_list)
        local_face_list = []
        for i in range(x_len - 1):
            if i < x_len - 2:
                for j in range(y_len - 1):
                    local_face = frame[x_list[i]:x_list[i + 1], y_list[j]:y_list[j + 1]]
                    local_face_list.append(local_face)

            # 对最后一行单独处理
            else:
                for j in range(1, y_len - 2):
                    local_face = frame[x_list[i]:x_list[i + 1], y_list[j]:y_list[j + 1]]
                    local_face_list.append(local_face)
        return local_face_list

    def get_local_face(self, frame, key):
        '''
        将人脸分成多个局部，鼻子以下，两个部位，或者多个部位  /Divide the face into multiple parts,
        :param frame:
        :param key:
        :return:
        '''
        frame_local = []
        x_list = []
        y_list = []
        # 坐标
        x1 = key[28][1]
        x2 = key[9][1]
        x3 = key[34][1]
        x_mid_up = int((x1 + x3) / 2)
        x_mid_down = int((x3 + x2) / 2)
        x_list.append(x1)
        # x_list.append(x_mid_up)
        x_list.append(x3)
        # x_list.append(x_mid_down)
        x_list.append(x2)

        y1 = key[3][0]
        y2 = key[30][0]
        y3 = key[13][0]
        y_mid_left = int((y1 + y2) / 2)
        y_mid_right = int((y2 + y3) / 2)
        y_list.append(y1)
        y_list.append(y_mid_left)
        y_list.append(y2)
        y_list.append(y_mid_right)
        y_list.append(y3)

        local_face_List = self.coor_to_face(x_list, y_list, frame)
        frame_local = local_face_List

        # 左边背景
        distance = y_mid_left - y1
        y0 = y1 - distance * 3
        y4 = y3 + distance * 3

        # 拉远距离，避免人脸的出现
        y1 = y1 - distance * 2
        y3 = y3 + distance * 2

        if y0 < 0:
            background_left = frame[x1:x3, 0:y1]
        else:
            background_left = frame[x1:x3, y0:y1]
        frame_local.append(background_left)

        # 右边背景
        if y4 > 640:
            background_right = frame[x1:x3, y3:640]
        else:
            background_right = frame[x1:x3, y3:y4]

        frame_local.append(background_right)

        # 绘图
        cv2.imshow("picture", frame)
        if list(frame_local[0]) != []:
            try:
                cv2.imshow("face1", frame_local[-1])
                cv2.imshow("face2", frame_local[-2])
            except Exception as e:
                print("debug")

        return frame_local

    def process_frame_global(self, frame):

        '''
        处理视频帧，提取整个人脸部分并返回   /Process  frames, extract the entire face part and return
        :param frame:
        :param sub_roi:
        :return:
        '''

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测,返回人脸，灰色人脸，上次人脸

        frame_cropped, gray_frame, self.prev_face, flag = crop_to_face(frame, gray, self.prev_face)

        # 求关键点
        key = self.a_face_detection.landmark_detection(frame_cropped, self.prev_face)

        # 利用关键点分割，返回精准人脸
        frame_cropped = self.get_global_face(frame, key)
        frame_cropped = [frame_cropped]  # 转成三维列表和local face统一


        num_pixels = frame.shape[0] * frame.shape[1]

        return frame_cropped, num_pixels, flag

    def process_frame_local(self, frame):

        '''
        处理视频帧，提取人脸的多个局部部分 /Process  frames to extract multiple local parts of the face
        :param frame:
        :param sub_roi:
        :return:
        '''
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测 / Face Detection

        frame_cropped, gray_frame, self.prev_face, flag = crop_to_face(frame, gray, self.prev_face)

        # 求关键点 / find key point of face
        key = self.get_landmarks(gray, self.prev_face)

        # 利用关键点分割，返回多个局部人脸  /segment image to different parts by key point
        frame_cropped = self.get_local_face(frame, key)

        num_pixels = frame.shape[0] * frame.shape[1]

        return frame_cropped, num_pixels, flag

    def measure_rPPG(self, frame):
        '''
        输入图片，测量图片的RGB分量均值并且保存 /
        :param frame:
        :return:
        '''
        # frame_cropped, num_pixels = self.process_frame_global(frame, sub_roi)
        frame_cropped, num_pixels, flag = self.process_frame_local(frame)

        face_num = len(frame_cropped)
        for i in range(face_num):
            face_data = frame_cropped[i]
            # 三维数组，RGB，每一个是一个一维数组，组合起来是二维数据，多个部位，多个二维数组组合起来是三维数组
            self.rPPG[i].append(self.calc_ppg(num_pixels, face_data))

        # fb=open('./rppg.txt','a+')
        # fb.write(str(self.rPPG))
        # fb.write('\n')
        # fb.close()
