import cv2
import numpy as np
import Core.PreProcessing.Subsystem as Preprocessor


class Subsystem:
    # input: frame, faceNet
    # output: array of faces and array of corresponding x,y box of faces
    def __init__(self, frame, height, width, blob, faceNet):
        # Attributes
        self.frame = frame
        self.height = height
        self.width = width
        self.blob = blob
        self.faceNet = faceNet

    # Functions relevant to face detection module
    def initialize(self, faceNet):
        print("Hello this is the FaceDetection subsystem")
        self.faceNet = faceNet

    def setFrame(self, frame):
        self.frame = frame
        self.height = frame.shape[0]
        self.width = frame.shape[1]
        self.blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    def obtainFaceDetects(self, blob):
        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()
        return detections

    def computeFaceBox(self, detections, index):
        # compute the (x, y)-coordinates of the bounding box for
        # the object
        box = detections[0, 0, index, 3:7] * np.array([self.width, self.height, self.width, self.height])
        (startX, startY, endX, endY) = box.astype("int")

        # ensure the bounding boxes fall within the dimensions of
        # the frame
        (startX, startY) = (max(0, startX), max(0, startY))
        (endX, endY) = (min(self.width - 1, endX), min(self.height - 1, endY))
        return (startX, startY, endX, endY)

    def runFaceDetect(self):
        detections = self.obtainFaceDetects(self.blob)

        faces = []
        locations = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence, Adrian Rosebrock default confidence = 0.5
            if confidence > 0.5:
                (startX, startY, endX, endY) = self.computeFaceBox(detections, i)
                cropped_face = self.frame[startY:endY, startX:endX]

                faces.append(cropped_face)
                locations.append((startX, startY, endX, endY))

        return faces, locations