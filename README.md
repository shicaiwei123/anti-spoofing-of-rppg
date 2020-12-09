# anti-spoofing-of-rppg
the implement of 3D Mask Face Anti-spoofing with Remote Photoplethysmography

English|[中文](https://github.com/shicaiwei123/anti-spoofing-of-rppg/blob/master/ReadMe_CH.md)
)


# Introduction of project
- config.py 
    - Contains all the settings we need to configure 
- rPPG_GUI.py
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
    - Check if your camera is working by cheese command in linux bash
    - Configure config.py
    - run rPPG_GUI.py


- run rPPG_GUI.py
    - ```python3 rPPG_GUI.py```
    
- reference
    - Remote Photoplethysmography Correspondence Feature for 3D Mask Face Presentation Attack Detection
    - Generalized face anti-spoofing by detecting pulse from face videos
- the implement of Face Anti-Spoofing Using Patch and Depth-Based CNNs
    - https://github.com/shicaiwei123/patch_based_cnn
    


