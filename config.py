from argparse import ArgumentParser
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

parser = ArgumentParser()
parser.add_argument('--fs',type=int,default=30,help='sampling rate of camera')
parser.add_argument('--fftlength',type=int,default=300,help='data length of fft,the larger the value, the slower the processing speed and the higher the accuracy')
parser.add_argument('--freq',default=[40,200],help='Cutoff frequency of bandpass filter')
parser.add_argument('--source',default='webcam',help='source of data')

args = parser.parse_args()