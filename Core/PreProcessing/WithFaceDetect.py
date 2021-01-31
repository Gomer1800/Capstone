import cv2
import numpy as np


class FaceDetect:
    # input: frame, faceNet
    # output: array of faces and array of corresponding x,y box of faces
    def __init__(self, frame, faceNet):
        # Attributes
        self.frame = frame
        self.height = frame[0]
        self.width = frame[1]
        self.blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.faceNet = faceNet

        # output
        self.faces = []
        self.locations = []

    # Functions relevant to face detection module
    def obtainFaceDetects(self):
        self.faceNet.setInput(self.blob)
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


    def addFace(self, face):
        self.faces.append(face)

    def addLocations(self, startX, startY, endX, endY):
        self.locations.append((startX, startY, endX, endY))
