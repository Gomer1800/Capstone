from tensorflow.keras.models import load_model
import numpy as np
import cv2


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
        return self.predictions

    def faceCoordsToMaskROI(self, startX, startY, endX, endY):
        # couldn't find any example code that bounded facial features so just gonna resize the
        # face detection coordinates as a start

        return startX, startY + 55, endX, endY

    def detectionBox(self, frame, startX, startY, endX, endY):
        color = (0, 255, 0)  # GREEN outline if mask is detected
        for pred in self.predictions:
            if pred[0] > pred[1]:
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            else:
                continue
