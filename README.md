# anti-spoofing-of-rppg
the implement of 3D Mask Face Anti-spoofing with Remote Photoplethysmography

# 代码参数介绍（Introduction of code parameter ）
- 滤波器截止频率（Cutoff frequency of filter）
    - 在rPPG_preprocessing.py，40,200，就是滤波器的频率范围，40Hz到200Hz（In rPPG_preprocessing.py, 40,200, is the frequency range of the filter, 40Hz to 200Hz）
- fft参数（fft parameter）
    - fft_length
    - 决定频谱分辨率（Determine spectral resolution）
- fs
    -摄像头采样率（Camera sampling rate）

# 代码运行逻辑（logic of code ）
- 主函数（rPPG_GUI.py）中依据source参数选择图片读入方式，视频或者摄像头（The main function (rPPG_GUI.py) selects the image reading mode, video or camera according to the source parameter.）
- 然后启动定时器，每隔制定时间刷新，获取一张图片进行分析。刷新频率一般高于相机采样率，因为后面会有一个对图片流的重采样（Then start the timer, refresh every time, and get a picture for analysis. The refresh rate is generally higher than the camera sample rate because there will be a resampling of the picture stream later.）
- 对采集到的图片数据流的处理（Processing of the collected image data stream）
    - 因为代码刷新频率低于摄像头采样率，实际的图像采样率率会低于摄像头的采样率。需要求实际的图像采样率（Because the code refresh rate is lower than the camera sample rate, the actual image sample rate will be lower than the camera's sample rate. Need to get the actual image sampling rate）
    - 求每一帧图像的ppg数据，并保存（Get ppg data for each frame of image and save）
    - 根据镜头帧率重采样（Resampling based on lens frame rate）
    - 求脉冲（Get the pulse）
        - fft求频谱（fft for spectrum）
        - 求模，取最大值对应的频率分量（Get the Amplitude, take the frequency component corresponding to the maximum value）
- 运行（run）:

    ```python3 rPPG_GUI.py```

