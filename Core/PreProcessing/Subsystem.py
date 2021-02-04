# TODO(Luis): we need to source these dependencies for the entire project workspace
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2


class Subsystem:
    # input: frame as numpy.ndarray, the frame is just the face
    # output: frame, modified as numpy.ndarray
    def __init__(self, frame):
        # Attributes
        self.frame = frame
        self.modified = frame

    # Functions relevant to Pre_Processing module
    def initialize(self):
        print("Hello this is the preprocessor subsystem")

    def setFrame(self, frame):
        self.frame = frame
        self.modified = frame

    def cvtToRGB(self):
        self.modified = cv2.cvtColor(self.modified, cv2.COLOR_BGR2RGB)
        return self.modified

    def resize(self, newW, newH):
        self.modified = cv2.resize(self.modified, (newW, newH))
        return self.modified

    def imgToArr(self):
        self.modified = img_to_array(self.modified)
        return self.modified

    def encode(self):
        self.modified = preprocess_input(self.modified)
        return self.modified

    def prepareFace(self):
        self.cvtToRGB()
        self.resize(224, 224)
        self.imgToArr()
        self.encode()
        return self.modified