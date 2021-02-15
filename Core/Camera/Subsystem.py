"""
TODO(LUIS): Camera Module Readme right here
"""
from imutils.video import VideoStream
import imutils
import time


class Subsystem:
    def __init__(self, type, name, camera_path, storage_path):
        self.camera_type = type  # string
        # TODO(LUIS): Decide on how to support following attributes
        self.camera_name = name
        self.camera_path = camera_path
        self.camera_storage = storage_path
        self.camera = None

    def initialize(self):
        """
        :return: an integer, 1 if shutdown is successful, else 0
        """
        print("Hello this is the camera subsystem")

        # TODO(LUIS): add logic to choose between webcam and IP camera
        if self.camera_type == "WEB":
            # initialize the video stream and allow the camera sensor to warm up
            self.camera = VideoStream(src=0).start()
            time.sleep(2.0)
            return 1

        elif self.camera_type == "IP":
            return 1

        else:
            # TODO(LUIS): Return error here or raise an exception?
            return 0

    def shutdown(self):
        """
        :return: an integer, 1 if shutdown is successful, else 0
        """
        if self.camera_type == "WEB":
            self.camera.stop()
            return 1

        elif self.camera_type == "IP":
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
            frame = self.camera.read()
            frame = imutils.resize(frame, width=400)
            return frame

        elif self.camera_type == "IP":
            pass

        else:
            # TODO(LUIS): Return error here or raise an exception?
            pass
