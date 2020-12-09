
# anti-spoofing-of-rppg
the implement of 3D Mask Face Anti-spoofing with Remote Photoplethysmography


[English](https://github.com/shicaiwei123/anti-spoofing-of-rppg/blob/master/README.md)|中文

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
    - 利用cheese命令,检查摄像头是否工作
    - 配置config.py ,已经配置好了
    - 运行 rPPG_GUI.py
    
- run rPPG_GUI.py
    - ```python3 rPPG_GUI.py```
    
- reference
    - Remote Photoplethysmography Correspondence Feature for 3D Mask Face Presentation Attack Detection
    - Generalized face anti-spoofing by detecting pulse from face videos

-  Face Anti-Spoofing Using Patch and Depth-Based CNNs 的实现
    - https://github.com/shicaiwei123/patch_based_cnn
    