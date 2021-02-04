from tensorflow.keras.models import load_model
import numpy as np


class SubSystem:
    # inputs:    preprocessed image data: image frame
    # outputs:   mask detection status: flag
    #            image data: xy coordinates of mask
    def __init__(self):
        pass

    def initialize(self):
        print("Hello this is the MaskDetection subsystem")

    # takes in an array of faces and converts them into an np array for prediction
    def convertFacestoNPArray(self):
        self.faces = np.array(self.faces, dtype="float32")

    # applies mask prediction method on an np array and returns a list of predictions
    # format of each prediction should be (mask, withoutMask)
    def prediction(self, maskModel):
        self.predictions = maskModel.predict(self.faces, batch_size=32)

    def printMaskPrediction(self):
        for pred in self.predictions:
            if pred[0] > pred[1]:
                print("Mask")
            else:
                print("No Mask")

    # performs all of maskDetection functions
    def runInference(self, faces, predictions, maskModel):
        self.faces = faces
        self.predictions = predictions
        self.convertFacestoNPArray()
        self.prediction(maskModel)
        self.printMaskPrediction()
