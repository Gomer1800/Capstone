from imutils import face_utils
import dlib
import cv2


class Subsystem:
    def __init__(self, detector, predictor, mouthCascade, noseCascade):
        self.detector = detector
        self.predictor = predictor

        # CASCADE STUFF
        self.mouthCascade = mouthCascade
        self.noseCascade = noseCascade

    def initialize(self, predictor, mouth_xml, nose_xml):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor)

        # CASCADE STUFF
        self.mouthCascade = cv2.CascadeClassifier(mouth_xml)
        if self.mouthCascade.empty():
            raise IOError('Unable to load the mouth cascade classifier xml file')

        self.noseCascade = cv2.CascadeClassifier(nose_xml)
        if self.noseCascade.empty():
            raise IOError('Unable to load the nose cascade classifier xml file')

        print("Hello this is the Facial Feature subsystem")

    # ============================= ADRIAN ROSEBROCK STUFF ===================================================

    def detect_faces(self, grayframe):
        rects = self.detector(grayframe, 0)
        return rects

    # returns an array of tuples [(x, y), (x, y), ...], index 28-35 are nose dots, 49+ are mouth dots
    # this is how to call it in post processing if want to display dots:
    # for (x, y) in shape:
    #     cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    def detect_facial_landmarks(self, grayframe):
        shape = []
        rects = self.detect_faces(grayframe)
        for rect in rects:
            shape = self.predictor(grayframe, rect)
            shape = face_utils.shape_to_np(shape)
        return shape

    # ====================================== CASCADE STUFF ====================================================

    def calculate_coords(self, rects, endYscale):
        new = []

        if len(rects) > 0:
            for (x, y, w, h) in rects:
                startX = x
                startY = y
                endX = x + w
                endY = int(y - endYscale * h) + h
                new.append((startX, startY, endX, endY))
        return new

    # Takes in the grayscaled frame and a string that specifies which type of detection
    # Returns an array of tuples [(startX, startY, endX, endY), ...]
    def cascade_detect(self, gray, facial_feat):
        rects = []
        endYscale = 0
        if facial_feat == "mouth":
            rects = self.mouthCascade.detectMultiScale(gray, 1.7, 11)
            endYscale = 0.25

        elif facial_feat == "nose":
            rects = self.noseCascade.detectMultiScale(gray, 1.3, 5)
            endYscale = 0.25

        newRects = self.calculate_coords(rects, endYscale)
        return newRects
