"""
TODO(LUIS): Camera Module Readme right here
"""


class Subsystem:
    def __init__(self, type, name, camera_path, storage_path):
        self.camera_type = type
        self.camera_name = name
        self.camera_path = camera_path
        self.camera_storage = storage_path

    def initialize(self):
        print("Hello this is the camera subsystem")

    def capture_image(self):
        """
        TODO(LUIS): What is the exact format type for an open cv image?
        :return: Open CV image
        """
        pass
