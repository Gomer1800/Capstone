# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import Core.Camera.Subsystem as Camera
import Core.PreProcessing.Subsystem as Preprocessor
import Core.MaskDetection.SubSystem as MaskDetection
import Core.PostProcessing.SubSystem as Postprocessor

# def print_hi(name):
# Use a breakpoint in the code line below to debug your script.
# print(f'Hi this is a test, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # FSM states, for linear control flow
    # TODO(Luis): we need to develop a FSM for parallelism
    states = ["INIT", "CAM", "PRE", "MASK", "LEAK", "FACE", "POST"]

    presentState = "INIT"
    nextState = "PRE"
    loop = True

    while loop is True:
        if presentState == "INIT":
            camera = Camera.Subsystem(
                type=None,
                name=None,
                camera_path=None,
                storage_path=None
            )
            preprocessor = Preprocessor.Subsystem(frame=None)
            maskDetector = MaskDetection.SubSystem(face=None)
            postprocessor = Postprocessor.SubSystem()

            camera.initialize()
            preprocessor.initialize()
            maskDetector.initialize()
            nextState = "CAM"

        elif presentState == "PRE":
            nextState = "MASK"

        elif presentState == "PRE":
            # TODO(Luis):there will be a split in the control flow to jump to Postprocessing early
            nextState = "MASK"

        elif presentState == "MASK":
            nextState = "LEAK"

        elif presentState == "LEAK":
            nextState = "FACE"

        elif presentState == "FACE":
            nextState = "POST"

        elif presentState == "POST":
            nextState = "CAM"

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
