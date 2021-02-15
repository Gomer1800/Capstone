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

    # Functions relevant to Pre_Processing module
    def initialize(self):
        print("Hello this is the preprocessor subsystem")

    def setFrame(self, frame):
        self.frame = frame

    def cvtToRGB(self, frame):
        modified = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return modified

    def cvtToGRAY(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return gray

    def resize(self, frame, newW, newH):
        modified = cv2.resize(frame, (newW, newH))
        return modified

    def imgToArr(self, frame):
        modified = img_to_array(frame)
        return modified

    def encode(self, frame):
        modified = preprocess_input(frame)
        return modified

    def prepareFace(self):
        mod = self.cvtToRGB(self.frame)
        mod = self.resize(mod, 224, 224)
        mod = self.imgToArr(mod)
        mod = self.encode(mod)
        return mod