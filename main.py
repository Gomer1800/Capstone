# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import Core.Camera.Subsystem as Camera
import Core.PreProcessing.Subsystem as PreProcessor
import Core.PreProcessing.WithFaceDetect as FaceDetection
import Core.MaskDetection.SubSystem as MaskDetection
import Core.PostProcessing.SubSystem as Postprocessor

import cv2
import os

from tensorflow.keras.models import load_model

# def print_hi(name):
# Use a breakpoint in the code line below to debug your script.
# print(f'Hi this is a test, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # FSM states, for linear control flow
    # TODO(Luis): we need to develop a FSM for parallelism
    states = ["INIT", "CAM", "PRE", "MASK", "LEAK", "FACE", "POST", "DISPLAY"]

    presentState = "INIT"
    nextState = "CAM"
    loop = True

    camera = None
    preprocessor = None
    face_detection = None
    mask_detection = None
    postprocessor = None

    frame = None
    prepared = None
    masknet = None
    predictions = None
    locations = None

    while loop is True:
        if presentState == "INIT":
            # TODO(LUIS): Put all the model setup logic in a function, encapsulate it

            masknet = load_model("./Core/MaskDetection/mask_detector.model")

            prototxtPath = os.path.sep.join(["Core",
                                             "PreProcessing",
                                             "face_detector",
                                             "deploy.prototxt"])

            weightsPath = os.path.sep.join(["Core",
                                            "PreProcessing",
                                            "face_detector",
                                            "res10_300x300_ssd_iter_140000.caffemodel"])

            faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

            camera = Camera.Subsystem(
                type=None,
                name=None,
                camera_path=None,
                storage_path=None
            )
            preprocessor = PreProcessor.Subsystem(
                frame=None
            )
            face_detection = FaceDetection.Subsystem(
                frame=None,
                height=None,
                width=None,
                blob=None,
                faceNet=None
            )
            mask_detection = MaskDetection.SubSystem()
            postprocessor = Postprocessor.SubSystem()

            camera.initialize()
            preprocessor.initialize()
            face_detection.initialize(faceNet)
            mask_detection.initialize()
            postprocessor.initialize()

            nextState = "CAM"

        elif presentState == "CAM":
            frame = camera.capture_image()
            nextState = "PRE"

        elif presentState == "PRE":  # Face Detection Module
            # TODO(Luis):there will be a split in the control flow to jump to Postprocessing early
            face_detection.setFrame(frame)
            faces, locations = face_detection.runFaceDetect()

            # Pre Process Module
            prepared = []
            if len(faces) > 0:
                for face in faces:
                    preprocessor.setFrame(face)
                    modified = preprocessor.prepareFace()
                    cv2.imshow("modified", modified)
                    prepared.append(modified)
            nextState = "MASK"

        elif presentState == "MASK":
            # Mask Detection
            predictions = []
            predictions = mask_detection.runInference(prepared, predictions, masknet)
            nextState = "LEAK"

        elif presentState == "LEAK":
            nextState = "FACE"

        elif presentState == "FACE":
            nextState = "POST"

        elif presentState == "POST":
            print("printing pred outside")
            print(predictions)
            for (box, pred) in zip(locations, predictions):
                (mask, withoutMask) = pred
                # will make a box around face not mask area because these are Kenneth's coordinates
                (startX, startY, endX, endY) = box

                # Post-Processing Module
                #       1) will always yield green box because mask > noMask; probability = 100% mask
                #       2) switch tuple to (0, 1) if testing mask < noMask (red box); probability = 100% no mask
                #       3) to test probability reading, play with different numbers in tuple
                #          ie. (0.38, 0.62) -> probability = 62% no mask
                print("printing pred inside")
                print(pred)
                output_frame = postprocessor.prepareOutputFrame(frame,
                                                                (pred[0], pred[1]),
                                                                startX,
                                                                startY,
                                                                endX, endY)
            nextState = "DISPLAY"

        elif presentState == "DISPLAY":
            while True:
                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key was pressed, break from the loop
                if key == ord("q"):
                    break
            nextState = "CAM"

        else:
            nextState = "INIT"

        presentState = nextState

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
