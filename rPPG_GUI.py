# Author : Martin van Leeuwen
# Video pulse rate monitor using rPPG with the chrominance method.
# python动态绘图
############################## User Settings #############################################

class Settings():
    def __init__(self):
        self.use_classifier = True  # Toggles skin classifier
        self.use_flow = False  # (Mixed_motion only) Toggles PPG detection
        # with Lukas Kanade optical flow
        self.show_cropped = True  # Shows the processed frame on the aplication instead of the regular one.
        self.sub_roi = []  # [.35,.65,.05,.15] # If instead of skin classifier, forhead estimation should be used
        # set to [.35,.65,.05,.15]
        self.use_resampling = True  # Set to true with webcam 


# In the source either put the data path to the image sequence/video or "webcam"
# source = "C:\\Users\\marti\\Downloads\\Data\\13kmh.mp4" # stationary\\bmp\\"
# source = "C:\\Users\\marti\\Downloads\\Data\\stationary\\bmp\\"
source = "webcam"
fs = 20  # Please change to the capture rate of the footage.    镜头采样率

############################## APP #######################################################

from VideoHealthMonitoring.util.qt_util import *
from VideoHealthMonitoring.util.pyqtgraph_util import *
import numpy as np
from VideoHealthMonitoring.util.func_util import *
import matplotlib.cm as cm
from VideoHealthMonitoring.util.style import style
from VideoHealthMonitoring.util.opencv_util import *
from VideoHealthMonitoring.rPPG_Extracter import *
from VideoHealthMonitoring.rPPG_lukas_Extracter import *
from VideoHealthMonitoring.rPPG_processing_realtime import extract_pulse

## Creates The App 

fftlength = 300

f = np.linspace(0, fs / 2, int(fftlength / 2 + 1)) * 60
# f = np.linspace(0, fs / 2, 151) * 60
settings = Settings()


def create_video_player():
    frame = cv2.imread("placeholder.png")
    vb = pg.GraphicsView()
    frame, _ = pg.makeARGB(frame, None, None, None, False)
    img = pg.ImageItem(frame, axisOrder='row-major')
    img.show()
    vb.addItem(img)
    return img, vb


# 绘图设置
app, w = create_basic_app()
img, vb = create_video_player()

layout = QHBoxLayout()
control_layout = QVBoxLayout()
layout.addLayout(control_layout)
layout.addWidget(vb)

fig = create_fig()
fig.setTitle('rPPG')
addLabels(fig, 'time', 'intensity', '-', 'sec')
plt_r = plot(fig, np.arange(0, 5), np.arange(0, 5), [255, 0, 0])
plt_g = plot(fig, np.arange(0, 5), np.arange(0, 5), [0, 255, 0])
plt_b = plot(fig, np.arange(0, 5), np.arange(0, 5), [0, 0, 255])

fig_bpm = create_fig()
fig_bpm.setTitle('Frequency')
fig_bpm.setXRange(0, 300)
addLabels(fig_bpm, 'Frequency', 'intensity', '-', 'BPM')
plt_bpm = plot(fig_bpm, np.arange(0, 5), np.arange(0, 5), [255, 0, 0])
plt_bpm_right = plot(fig_bpm, np.arange(0, 5), np.arange(0, 5), [0, 255, 0])

layout.addWidget(fig)
layout.addWidget(fig_bpm)
timestamps = []
time_start = [0]


def resample_rppg(rppg, timestamps, fs):
    # 初始化
    rppg_resample = []

    rPPG_len = len(rppg)
    for i in range(rPPG_len):
        rppg_one = rppg[i]
        if rppg_one !=[]:
            rppg_one = np.transpose(rppg_one)

            t = np.arange(0, timestamps[-1], 1 / fs)  # 按照镜头的帧率重采样，间隔不变，上限变大，t会越来越长
            rPPG_resampled = np.zeros((3, t.shape[0]))  # 3xn

            # 按照每一个颜色空间来处理，一维线性差值
            for col in [0, 1, 2]:
                rPPG_resampled[col] = np.interp(t, timestamps, rppg_one[col])
            rppg_resample.append(rPPG_resampled)

    return rppg_resample


def extract_pulse_local(rppg, fs):
    '''
    输入所有的face的rppg信号,初始采样时间戳，摄像头采样率，然后输出重采样之后的rppg以及获取的pulse信号
    :param rppg:三维list
    :return:
    '''

    # 初始化
    pulse = []

    # 操作
    rPPG_len = len(rppg)
    for i in range(rPPG_len):
        rppg_one = rppg[i]
        pulse_one = extract_pulse(rppg_one, fftlength, fs)
        pulse.append(pulse_one)

    return pulse

def pulse_process(pulse_list):
    # 相乘
    face_multil_face=pulse_list[0]*pulse_list[1]
    face_mutil_ground=pulse_list[0]*pulse_list[2]
    ground_mutil_ground=pulse_list[1]*pulse_list[2]

    #相减
    face_sub_face=pulse_list[0] - pulse_list[1]
    face_sub_ground=pulse_list[0] - pulse_list[2]
    face_distance_face=face_sub_face**2
    face_distance_ground=face_sub_ground**2

    #相关
    face_cor_face_data=np.array([pulse_list[0],pulse_list[1]])
    face_cor_face=np.corrcoef(face_cor_face_data)
    ground_cor_ground_data=np.array([pulse_list[2],pulse_list[3]])
    ground_cor_ground=np.array(ground_cor_ground_data)

    return [face_multil_face,ground_mutil_ground,face_cor_face,ground_cor_ground]

def update(load_frame, rPPG_extracter, rPPG_extracter_lukas, settings: Settings):
    '''
    循环主体
    :param load_frame: 帧读取句柄，
    :param rPPG_extracter:
    :param rPPG_extracter_lukas:
    :param settings:
    :return:
    '''
    bpm = 0
    frame, should_stop, timestamp = load_frame()  # frame_from_camera()

    # 保存刷新时间，然后求帧率显示而已
    dt = time.time() - time_start[0]
    fps = 1 / (dt)

    time_start[0] = time.time()

    # 缓存每一次处理的时间，而且是不断叠加的。
    if len(timestamps) == 0:
        timestamps.append(0)
    else:
        timestamps.append(timestamps[-1] + dt)

    # print("Update")
    if should_stop:
        return

    # 求rppg,两类方法，目前用的是第二个
    rPPG_sample = []
    if settings.use_flow:
        rPPG_extracter = rPPG_extracter_lukas
        rPPG_extracter.crop_to_face_and_safe(frame)
        rPPG_extracter.track_Local_motion_lukas()
        rPPG_extracter.calc_ppg(frame)
        points = rPPG_extracter.points
        frame = cv2.circle(frame, (points[0, 0, 0], points[0, 0, 1]), 5, (0, 0, 255), -1)

        rPPG_sample = np.transpose(rPPG_extracter_lukas.rPPG_sample)
    else:
        # 获取到目前的所有图片的ppg值，并且转置，一开始是nx3，转置变成3xn
        rPPG_extracter.measure_rPPG(frame, settings.use_classifier, settings.sub_roi)

        #  提取分量，再做变换
        rPPG_sample = rPPG_extracter.rPPG[0]
        rPPG_sample = np.transpose(rPPG_sample)

    # Extract Pulse
    # 如果列数大于10，也就是录入了至少10张照片
    if rPPG_sample.shape[1] > 10:
        if settings.use_resampling:
            rppg = resample_rppg(rPPG_extracter.rPPG, timestamps, fs)
        else:
            rppg = rPPG_extracter.rPPG

        pulse = extract_pulse_local(rppg, fs)
        # print(len(f))
        # print(len(pulse))

        # 求帧数,确定作图截取数据的范围
        rppg_one = rppg[0]
        num_frames = rppg_one.shape[1]
        start = max([num_frames - 100, 0])

        # 从0开始，每一帧的真实时间
        t = np.arange(num_frames) / fs

        multil_process=pulse_process(pulse)
        plt_bpm.setData(f, multil_process[0])
        plt_bpm_right.setData(f, multil_process[1])
        print("face_to_face",multil_process[2])
        print("ground_to_ground",multil_process[3])


        plt_r.setData(t[start:num_frames], rppg_one[0, start:num_frames])
        plt_g.setData(t[start:num_frames], rppg_one[1, start:num_frames])
        plt_b.setData(t[start:num_frames], rppg_one[2, start:num_frames])

        # 求能量然后取最值
        bpm = f[np.argmax(multil_process[1])]
        fig_bpm.setTitle('Frequency : PR = ' + str(bpm) + ' BPM')

    # # print(fps)
    # face = rPPG_extracter.prev_face
    # if not settings.use_flow:
    #     if settings.show_cropped:
    #         frame = rPPG_extracter.frame_cropped
    #         if len(settings.sub_roi) > 0:
    #             sr = rPPG_extracter.sub_roi_rect
    #             draw_rect(frame, sr)
    #     else:
    #         try:
    #             draw_rect(frame, face)
    #             if len(settings.sub_roi) > 0:
    #                 sr = rPPG_extracter.sub_roi_rect
    #                 sr[0] += face[0]
    #                 sr[1] += face[1]
    #                 draw_rect(frame, sr)
    #         except Exception:
    #             print("no face")
    # #    write_text(frame,"bpm : " + '{0:.2f}'.format(bpm),(0,100))
    # write_text(frame, "fps : " + '{0:.2f}'.format(fps), (0, 50))
    # frame, _ = pg.makeARGB(frame, None, None, None, False)
    # # cv2.imshow("images", np.hstack([frame]))
    # img.setImage(frame)


# 初始化计时器，计时单位ms
# 单纯缩小计时时长没用，因为相机的帧率本身是有限制的，提高刷新时间，但是还是同一张图片的话没有意义。
timer = QtCore.QTimer()
timer.start(10)


def setup_update_loop(load_frame, timer, settings):
    # = "C:\\Users\\marti\\Downloads\\Data\\translation"
    '''
     timer就在这里作用了，这也是循环迭代的地方
    :param load_frame:
    :param timer:
    :param settings:
    :return:
    '''
    try:
        timer.timeout.disconnect()
    except Exception:
        pass
    rPPG_extracter = rPPG_Extracter()
    rPPG_extracter_lukas = rPPG_Lukas_Extracter()
    update_fun = lambda: update(load_frame, rPPG_extracter, rPPG_extracter_lukas, settings)

    # 计时结束时，就调用这个函数，计时器内部自由这个计时逻辑
    timer.timeout.connect(update_fun)


def setup_loaded_image_sequence(data_path, timer, settings):
    vl = VideoLoader(data_path)

    setup_update_loop(vl.load_frame, timer, settings)


def setup_webcam(timer, settings):
    '''
    从摄像头读取帧数据，并处理
    :param timer:
    :param settings:
    :return:
    '''
    camera = cv2.VideoCapture(0)
    # camera.set(3, 1280)
    # camera.set(4, 720)
    settings.use_resampling = True

    def frame_from_camera():
        _, frame = camera.read()
        return frame, False, camera.get(cv2.CAP_PROP_POS_MSEC)

    setup_update_loop(frame_from_camera, timer, settings)


def setup_video(data_path, timer, settings):
    '''
    从视频读取帧数据并处理
    :param data_path:
    :param timer:
    :param settings:
    :return:
    '''
    vi_cap = cv2.VideoCapture(data_path)

    # settings.use_resampling = True
    def frame_from_video():
        _, frame = vi_cap.read()
        rows, cols = frame.shape[:2]

        # M = cv2.getRotationMatrix2D((cols/2,rows/2),270,1)
        # frame = cv2.warpAffine(frame,M,(cols,rows))
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
        return frame, False, 0

    # 把函数当做参数
    setup_update_loop(frame_from_video, timer, settings)


# settings = Settings()

# 根据全局变量，控制不同的分支
# 真正的main
if source.endswith('.mp4'):
    setup_video(source, timer, settings)
elif source == 'webcam':
    setup_webcam(timer, settings)
else:
    setup_loaded_image_sequence(source, timer, settings)

w.setLayout(layout)
execute_app(app, w)
camera.release()
cv2.destroyAllWindows()
