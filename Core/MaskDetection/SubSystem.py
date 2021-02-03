from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os

class SubSystem:
    # inputs:    preprocessed image data: image frame
    # outputs:   mask detection status: flag
    #            image data: xy coordinates of mask
    def __init__(self, face, maskStatus):
        self.face = face
        self.maskStatus = maskStatus

    def initialize(self):
        print("Hello this is the MaskDetection subsystem")

    def prediction(self, maskModel):
        self.maskStatus = maskModel.predict(self.face)[0]

    #       pass the face through the model to determine if the face
    # 		has a mask or not
    # 		(mask, withoutMask) = model.predict(face)[0]

    # i dont think need this
    def statusUpdate(self):
