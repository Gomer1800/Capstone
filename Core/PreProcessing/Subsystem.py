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

    def cvtToRGB(self):
        self.modified = cv2.cvtColor(self.modified, cv2.COLOR_BGR2RGB)
        pass

    def resize(self, newW, newH):
        self.modified = cv2.resize(self.modified, (newW, newH))
        pass

    def imgToArr(self):
        self.modified = img_to_array(self.modified)
        pass

    def encode(self):
        self.modified = preprocess_input(self.modified)
        pass
