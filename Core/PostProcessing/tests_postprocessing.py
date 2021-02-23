import cv2
import os
import Core.PreProcessing.WithFaceDetect as FaceDetection
import Core.PreProcessing.Subsystem as PreProcessor
import Core.Camera.Subsystem as Camera
import Core.PostProcessing.SubSystem as PostProcessing


def main():
    prototxtPath = os.path.sep.join(["..",
                                     "PreProcessing",
                                     "face_detector",
                                     "deploy.prototxt"])

    weightsPath = os.path.sep.join(["..",
                                    "PreProcessing",
                                    "face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])

    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    camera = Camera.Subsystem(
        type='WEB',
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

    postprocessor = PostProcessing.SubSystem()

    camera.initialize()
    preprocessor.initialize()
    face_detection.initialize(faceNet)
    postprocessor.initialize()

    while True:
        frame = camera.capture_image()

        # Face Detection Module
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

        # Drawing box around face location
        for box in locations:
            # will make a box around face not mask area because these are Kenneth's coordinates
            (startX, startY, endX, endY) = box

            # Post-Processing Module
            #       1) will always yield green box because mask > noMask; probability = 100% mask
            #       2) switch tuple to (0, 1) if testing mask < noMask (red box); probability = 100% no mask
            #       3) to test probability reading, play with different numbers in tuple
            #          ie. (0.38, 0.62) -> probability = 62% no mask
            postprocessing = PostProcessing.SubSystem()
            output_frame = postprocessing.prepareOutputFrame(frame, (0.69, 0.31), startX, startY, endX, endY)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    camera.shutdown()


if __name__ == '__main__':
    main()

