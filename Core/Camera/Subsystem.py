"""
TODO(LUIS): Camera Module Readme right here
"""
from imutils.video import VideoStream
import imutils
import time


class Subsystem:
    def __init__(self, type, name, camera_path, storage_path):
        self.camera_type = type
        self.camera_name = name
        self.camera_path = camera_path
        self.camera_storage = storage_path
        self.camera = None

    def initialize(self):
        print("Hello this is the camera subsystem")
        # TODO(LUIS): add logic to choose between webcam and IP camera
        # initialize the video stream and allow the camera sensor to warm up
        self.camera = VideoStream(src=0).start()
        time.sleep(2.0)

    def shutdown(self):
        self.camera.stop()

    def capture_image(self):
        """
        TODO(LUIS): What is the exact format type for an open cv image?
        :return: Open CV image
        """
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = self.camera.read()
        frame = imutils.resize(frame, width=400)
        return frame
