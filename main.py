# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import Core.Camera.Subsystem as Camera
import Core.PreProcessing.Subsystem as PreProcessor
import Core.PreProcessing.WithFaceDetect as FaceDetection
import Core.MaskDetection.SubSystem as MaskDetection
import Core.PostProcessing.SubSystem as Postprocessor
import Core.FacialFeatureDetection.Subsystem as FacialFeatureDetection
import Core.MaskEvaluate.Subsystem as MaskEvaluate

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as figure
import os
import pandas as pd
import time

from datetime import datetime
from tensorflow.keras.models import load_model

# def print_hi(name):
# Use a breakpoint in the code line below to debug your script.
# print(f'Hi this is a test, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # TODO(LUIS): Use python arg parse in future to get arguments from command line for:
    # camera type

    # FSM states, for linear control flow
    # TODO(Luis): we need to develop a FSM for parallelism
    states = ["INIT", "CAM", "PRE", "MASK", "LEAK", "FACE", "POST", "DISPLAY", "SHUTDOWN"]

    presentState = "INIT"
    nextState = "CAM"
    loop = True

    # Subsystems
    camera = None
    preprocessor = None
    face_detection = None
    mask_detection = None
    facial_feat_detection = None
    mask_eval = None
    postprocessor = None

    # Data
    # TODO(LUIS): Find a way to not have to do this
    frame = None
    prepared = None
    masknet = None
    predictions = None
    locations = None
    shape = None
    mouth_rects = None
    nose_rects = None
    facial_feature_flags = [1, 1, 1]

    # Timing Data
    # Dictionary where keys correspond to subsystems and value is an array of timing data
    # where the index corresponds to a given cycle or image taken
    # TODO(LUIS): Add num_cycles to arg parser branch
    NUM_CYCLE = 20
    # TODO(LUIS): When LEAK detection is implement, add timing data to dictionary
    timing_dict = {
        "CAM": np.zeros((NUM_CYCLE), dtype=float),
        "PRE": np.zeros((NUM_CYCLE), dtype=float),
        "MASK": np.zeros((NUM_CYCLE), dtype=float),
        "FACE": np.zeros((NUM_CYCLE), dtype=float),
        "POST": np.zeros((NUM_CYCLE), dtype=float),
    }
    cycle_counter = 0

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

            predictor_path = "Core/FacialFeatureDetection/shape_predictor_68_face_landmarks.dat"
            mouth_xml = "Core/FacialFeatureDetection/cascade-files/haarcascade_mcs_mouth.xml"
            nose_xml = "Core/FacialFeatureDetection/cascade-files/haarcascade_mcs_nose.xml"

            camera = Camera.Subsystem(
                type="WEB",  # "IP" for ip camera, "WEB" for web camera
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
            facial_feat_detection = FacialFeatureDetection.Subsystem(
                detector=None,
                predictor=None,
                mouthCascade=None,
                noseCascade=None
            )
            mask_eval = MaskEvaluate.Subsystem(
                noseflag=None,
                mouthflag=None,
                shapeflag=None
            )
            postprocessor = Postprocessor.SubSystem()

            camera.initialize()
            preprocessor.initialize()
            face_detection.initialize(faceNet)
            mask_detection.initialize()
            facial_feat_detection.initialize(predictor_path, mouth_xml, nose_xml)
            mask_eval.initialize()
            postprocessor.initialize()

            nextState = "CAM"

        elif presentState == "CAM":
            # start clock
            start = time.time()
            frame = camera.capture_image()
            # stop clock
            end = time.time()
            # record timing data
            timing_dict[presentState][cycle_counter] = end - start
            nextState = "PRE"

        elif presentState == "PRE":  # Face Detection Module
            start = time.time()
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

            end = time.time()
            timing_dict[presentState][cycle_counter] = end - start
            nextState = "MASK"

        elif presentState == "MASK":
            start = time.time()
            # Mask Detection
            if len(prepared) > 0:
                predictions = []
                predictions = mask_detection.runInference(prepared, predictions, masknet)
            end = time.time()
            timing_dict[presentState][cycle_counter] = end - start
            nextState = "LEAK"

        elif presentState == "LEAK":
            nextState = "FACE"

        elif presentState == "FACE":
            start = time.time()
            # Facial Feature Detection
            # TODO(LUIS): Add logic to handle no face detected so we dont waste cycles
            gray = preprocessor.cvtToGRAY(frame)
            shape = facial_feat_detection.detect_facial_landmarks(gray)
            mouth_rects = facial_feat_detection.cascade_detect(gray, "mouth")
            nose_rects = facial_feat_detection.cascade_detect(gray, "nose")
            # Mask Evaluation
            mask_eval.mask_evaluation(nose_rects, mouth_rects, shape)

            end = time.time()
            timing_dict[presentState][cycle_counter] = end - start
            nextState = "POST"

        elif presentState == "POST":
            start = time.time()
            # TODO(LUIS): This array should be maintained by the mask eval subsystem
            facial_feature_flags = [mask_eval.noseflag,
                                    mask_eval.mouthflag,
                                    mask_eval.shapeflag]

            for (box, pred) in zip(locations, predictions):
                (mask, withoutMask) = pred
                # will make a box around face not mask area because these are Kenneth's coordinates
                (startX, startY, endX, endY) = box

                # Post-Processing Module
                #       1) will always yield green box because mask > noMask; probability = 100% mask
                #       2) switch tuple to (0, 1) if testing mask < noMask (red box); probability = 100% no mask
                #       3) to test probability reading, play with different numbers in tuple
                #          ie. (0.38, 0.62) -> probability = 62% no mask
                output_frame = postprocessor.prepareOutputFrame(frame,
                                                                (pred[0], pred[1]),
                                                                startX,
                                                                startY,
                                                                endX, endY,
                                                                shape,
                                                                mouth_rects,
                                                                nose_rects,
                                                                facial_feature_flags)

            end = time.time()
            timing_dict[presentState][cycle_counter] = end - start
            nextState = "DISPLAY"

        elif presentState == "DISPLAY":
            cv2.imshow("Frame", frame)

            # if the `q` key was pressed, break from the loop
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            cycle_counter += 1
            if cycle_counter == NUM_CYCLE:
                nextState = "SHUTDOWN"
            else:
                nextState = "CAM"

        elif presentState == "SHUTDOWN":
            camera.shutdown()
            cv2.destroyAllWindows()
            break

        else:
            nextState = "INIT"

        presentState = nextState

    # Construct pandas dataframes
    timing_data = np.array(
        [timing_dict['CAM'],
         timing_dict['PRE'],
         timing_dict['FACE'],
         timing_dict['MASK'],
         timing_dict['POST']])
    timing_data = np.rot90(timing_data)
    data_frame = pd.DataFrame(timing_data,
                              columns=['CAM', 'PRE', 'FACE', 'MASK', 'POST'])
    data_frame.boxplot()
    """
    # Plot data, follow this guide https://pythonforundergradengineers.com/python-matplotlib-error-bars.html
    # enter raw data
    cam = timing_dict["CAM"]
    pre = timing_dict["PRE"]
    mask = timing_dict["MASK"]
    face = timing_dict["FACE"]
    post = timing_dict["POST"]

    # calculate averages
    cam_mean = np.mean(cam)
    pre_mean = np.mean(pre)
    mask_mean = np.mean(mask)
    face_mean = np.mean(face)
    post_mean = np.mean(post)

    # calculate standard deviation
    cam_std = np.std(cam)
    pre_std = np.std(pre)
    mask_std = np.std(mask)
    face_std = np.std(face)
    post_std = np.std(post)

    # create lists for plots
    subsystems = [
        'Camera',
        'Pre-Processing',
        'Mask Detection',
        'Facial Feature Detection',
        'Post-Processing']
    x_pos = np.arange(len(subsystems))
    means = [
        cam_mean,
        pre_mean,
        mask_mean,
        face_mean,
        post_mean]
    error = [
        cam_std,
        pre_std,
        mask_std,
        face_std,
        post_std]

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, means, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Time (seconds)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(subsystems)
    ax.set_title('Processing Time per Subsystems')
    ax.yaxis.grid(True)

    # Save the figure and show
    timestamp = datetime.now().strftime("%d-%b-%Y (%H_%M_%S.%f)")
    # plt.gcf().set_size_inches(20, 10)
    plt.tight_layout()
    plt.savefig(timestamp + ".png")
    plt.show()

    # save timing dict to csv, w/ timestamp
    timestamp = datetime.now().strftime("%d-%b-%Y (%H_%M_%S.%f)")
    file_name = timestamp + ".csv"
    with open(file_name, "a") as file:
        for subsystem in timing_dict.keys():
            # write subsystem name
            file.write(subsystem + "\n")
            # write data
            np.savetxt(fname=file, X=timing_dict[subsystem], delimiter=",")
    """
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
