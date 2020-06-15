# anti-spoofing-of-rppg
the implement of 3D Mask Face Anti-spoofing with Remote Photoplethysmography

# Introduction of project(config.py)
- config.py 
    - Contains all the settings we need to configure 
-rPPG_GUI.py
    - Entry function
    - ompleted the design of GUI interface
    - Detect if it is spoofing(Just a simple judgment, there is no data set, so SVM training is not used.)
- rPPG_preprocessing.py
    - Image preprocessing, face detection
- rPPG_processing_realtime.py
    - Extract rppg signals by fftlength's preprocessed photos 
- rPPG_Extracter.py
    - Take a picture and detect the face, then count he RGB components  of the face and find the average, finally cache them in  self.rPPG
- FaceDetection.py
    - Face detection, we uses key point detection function of it
- Processing
    - Run the program (rPPG_GUI.py) to collect photos by camera or video collected  , perform face detection on the photos (rPPG_preprocessing.py), and extract the average of RGB components (rPPG_Extracter.py). After preprocessing specified number (fftlength) photos, perform fft analysis and extraction rppg signal (rPPG_processing_realtime.py). Sliding window continuously extracts the signal and displays
    - 
- Run
    - Check if your camera is working by cheese
    - Configure config.py
    - Check if requirements.txt is satisfied
    - run rPPG_GUI.py
    
- 项目介绍
    - config.py 
        - 所有的参数配置都在这
    - rPPG_GUI.py
        - 入口函数
        - GUI设计显示
    - rPPG_preprocessing.py
        - 照片预处理:人脸检测
    - rPPG_processing_realtime.py
        - 通过fft,实时处理fftlenght张照片数据,提取rPPG信号
    - rPPG_Extracter.py
        - 拍照，对于每一帧，检测人脸，统计人脸照片的rgb空间的分量，求均值，然后缓存得到rppg
    - FaceDetection.py
        - 人脸检测,我主要用了里面的特征提取 
- 算法过程
    - 运行程序(rPPG_GUI.py)利用摄像头采集照片,对照片进行人脸检测(rPPG_preprocessing.py),并提取RGB分量的均值(rPPG_Extracter.py),当保存到指定数目(fftlength)之后,进行fft分析提取rppg信号(rPPG_processing_realtime.py),之后滑动窗口不断提取信号并显示
- 运行
    - 利用chees命令,检查摄像头是否工作
    - 配置config.py ,已经配置好了
    - 检测是否满足requirements.txt
    - 运行 rPPG_GUI.py

- run rPPG_GUI.py
    - ```python3 rPPG_GUI.py```
    
- reference
    - Remote Photoplethysmography Correspondence Feature for 3D Mask Face Presentation Attack Detection
    - Generalized face anti-spoofing by detecting pulse from face videos
    


