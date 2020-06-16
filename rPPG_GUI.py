# -*- coding: utf-8
from config import Settings
from config import args

source = args.source
fs =args.fs

############################## APP #######################################################

from util.qt_util import *
from util.pyqtgraph_util import *
from rPPG_Extracter import *
from rPPG_processing_realtime import extract_pulse

## Creates The App 

fftlength = args.fftlength

f = np.linspace(0, fs / 2, int(fftlength / 2 + 1)) * 60
settings = Settings()
no_face_frme = 0
mean_counter = 1


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
        if rppg_one != []:
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
    :param rppg:rppg data from rPPG_Extracter.py
    : fs sampling rate
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
        if pulse[0][0] != 0:
            end = datetime.datetime.now()
            time_sub = end - begin
            # print("time", time_sub.total_seconds())

    return pulse


def cross_enhace(data_list):
    '''
    输入二维数组，输出二维数组，求每一行与其他行的互相关增强
    :param data_list:
    :return:
    '''
    data_len = len(data_list)
    data_enhance = []
    # 相关增强
    for i in range(data_len):
        j = i + 1
        while j < data_len:
            data_enhance.append(data_list[i] * data_list[j])
            j += 1
    return data_enhance


def data_cross_data(data_list_one, data_list_two):

    '''
    输入两个二维数组求互相关，增强  /mutual correlation
    :param data_list_one:
    :param data_list_two:
    :return:
    '''
    len_one = len(data_list_one)
    len_two = len(data_list_two)
    data_cross = []
    for i in range(len_one):
        for j in range(len_two):
            data = np.array([data_list_one[i], data_list_two[j]])
            data_corff = np.corrcoef(data)

            # 只保留相关系数
            data_cross.append(data_corff[0][1])

    return data_cross


def pulse_process(pulse_list):
    '''
    对获取的pluse信号做处理,判断是否是欺骗  /Process the acquired pluse signal to determine whether it is spoofing attacks
    :param pulse_list:
    :return:
    '''
    '''
    :param pulse_list:
    :return:
    '''
    pulse_len = len(pulse_list)
    face_pulse = pulse_list[0:pulse_len - 2]
    ground_pulse = pulse_list[pulse_len - 2:pulse_len]

    # 相关增强
    face_pulse_enhance = cross_enhace(face_pulse)
    ground_pulse_enhance = cross_enhace(ground_pulse)

    # 相减
    face_sub_face = pulse_list[0] - pulse_list[1]
    face_sub_ground = pulse_list[0] - pulse_list[2]
    face_distance_face = face_sub_face ** 2
    face_distance_ground = face_sub_ground ** 2

    # 相关
    face_cor_face = data_cross_data([pulse_list[0]], [pulse_list[1]])
    ground_cor_ground = data_cross_data([pulse_list[-1]], [pulse_list[-2]])
    face_cor_ground = data_cross_data(face_pulse_enhance, ground_pulse_enhance)
    print("face_cor_ground", face_cor_ground)
    for _ in range(3):
        face_cor_ground.remove(max(face_cor_ground))
    mean_cor = np.mean(np.array(face_cor_ground))
    # if mean_cor < 0.50:
    #     print("true")
    # else:
    #     print("false")
    print("mean_cor", np.mean(np.array(face_cor_ground)))
    print("face_cor_face", face_cor_face)
    print("ground_cor_ground", ground_cor_ground)

    return face_pulse_enhance, ground_pulse_enhance, mean_cor


def update(load_frame, rPPG_extracter, settings=Settings):
    '''
    循环主体
    :param load_frame: 帧读取句柄，
    :param rPPG_extracter:
    :param rPPG_extracter_lukas:
    :param settings:
    :return:
    '''
    bpm = 0
    global mean_counter
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

    # 求rppg
    # 获取到目前的所有图片的ppg值，并且转置，一开始是nx3，转置变成3xn   /Get the ppg values of all the  pictures, and transpose, at first it is nx3, the transpose becomes 3xn
    rPPG_extracter.measure_rPPG(frame)

    #  提取分量，再做变换   / Extract the components and transform
    rPPG_sample = rPPG_extracter.rPPG[0]
    rPPG_sample = np.transpose(rPPG_sample)

    # Extract Pulse

    if rPPG_sample.shape[1] > 10:
        rppg = []
        if settings.use_resampling:
            rppg = resample_rppg(rPPG_extracter.rPPG, timestamps, fs)

        else:
            # 没有重采样也要做转置
            rppg_len = len(rPPG_extracter.rPPG)
            for i in range(rppg_len):
                rppg_one = rPPG_extracter.rPPG[i]
                if rppg_one != []:
                    rppg_one = np.transpose(rppg_one)
                    rppg.append(rppg_one)

        pulse = extract_pulse_local(rppg, fs)
        # print(len(f))
        # print(len(pulse))

        # 求帧数,确定作图截取数据的范围
        rppg_one = rppg[0]
        num_frames = rppg_one.shape[1]
        start = max([num_frames - 100, 0])

        # 从0开始，每一帧的真实时间
        t = np.arange(num_frames) / fs
        # a,b=pulse_process(rppg)
        face_pulse, ground_pulse, mean_cor = pulse_process(pulse)
        if mean_cor < 0.5:
            print("true")
        else:
            mean_counter += 1
        if mean_counter >= 4:
            print("false")
            mean_counter = 1

        plt_bpm.setData(f, face_pulse[0])
        plt_bpm_right.setData(f, ground_pulse[0])

        plt_r.setData(t[start:num_frames], rppg_one[0, start:num_frames])
        plt_g.setData(t[start:num_frames], rppg_one[1, start:num_frames])
        plt_b.setData(t[start:num_frames], rppg_one[2, start:num_frames])

        # 求能量然后取最值
        bpm = f[np.argmax(face_pulse[0])]

        # max_num=[]
        # for i in range(len(face_pulse)):
        #     max_num.append(f[np.argmax(face_pulse[i])])
        # print("max_num",max_num)
        fig_bpm.setTitle('Frequency : PR = ' + str(bpm) + ' BPM')


# 初始化计时器，计时单位ms /Initialize timer, time unit ms
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
    update_fun = lambda: update(load_frame, rPPG_extracter, settings)

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
    begin = datetime.datetime.now()
    setup_webcam(timer, settings)
else:
    setup_loaded_image_sequence(source, timer, settings)

w.setLayout(layout)
execute_app(app, w)
cv2.destroyAllWindows()
