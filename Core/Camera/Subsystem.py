"""
TODO(LUIS): Camera Module Readme right here
"""
import cv2
import imutils
import time

from imutils.video import VideoStream


class Subsystem:
    def __init__(self, type, name, camera_path, storage_path):
        self.camera_type = type  # string
        # TODO(LUIS): Decide on how to support following attributes
        self.camera_name = name
        self.camera_path = camera_path
        self.camera_storage = storage_path
        self.webcam = None
        self.ipcam = None

    def initialize(self):
        """
        :return: an integer, 1 if shutdown is successful, else 0
        """
        print("Hello this is the camera subsystem")

        # TODO(LUIS): add logic to choose between webcam and IP camera
        if self.camera_type == "WEB":
            # initialize the video stream and allow the camera sensor to warm up
            self.webcam = VideoStream(src=0).start()
            time.sleep(2.0)
            return 1

        elif self.camera_type == "IP":
            # TODO(LUIS): Currently, this is a hardcoded value provided by device manufacturers. How can we change this?
            # Armcrest PROHD 1080P
            # https://support.amcrest.com/hc/en-us/articles/360052688931-Accessing-Amcrest-Products-Using-RTSP

            # self.ipcam = cv2.VideoCapture(
            #    'rtsp://admin:creed1826@192.168.1.87:554/cam/realmonitor?channel=1&subtype=0')
            # TODO(LUIS): See camera capture for why this was commented out
            return 1

        else:
            # TODO(LUIS): Return error here or raise an exception?
            return 0

    def shutdown(self):
        """
        :return: an integer, 1 if shutdown is successful, else 0
        """
        if self.camera_type == "WEB":
            self.webcam.stop()
            return 1

        elif self.camera_type == "IP":
            self.ipcam.release()
            return 1

        else:
            # TODO(LUIS): Return error here or raise an exception?
            return 0

    def capture_image(self):
        """
        TODO(LUIS): What is the exact format type for an open cv image?
        :return: Open CV image
        """
        if self.camera_type == "WEB":
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            frame = self.webcam.read()
            frame = imutils.resize(frame, width=400)
            return frame

        elif self.camera_type == "IP":
            # TODO(LUIS): Temporary fix, have to initialize camera stream to get current image
            # else the video capture maintains a buffer of previous frames
            self.ipcam = cv2.VideoCapture(
                'rtsp://admin:creed1826@192.168.1.87:554/cam/realmonitor?channel=1&subtype=0')

            # TODO(LUIS): Handle error, when return_bool is False
            num = 0
            frame = None

            # Try to capture frame 5 times before returning failure
            while True and num < 5:
                num = num + 1
                return_bool, frame = self.ipcam.read()
                if return_bool is True:
                    frame = imutils.resize(frame, width=400)
                    break

            return frame

        else:
            # TODO(LUIS): Return error here or raise an exception?
            pass
