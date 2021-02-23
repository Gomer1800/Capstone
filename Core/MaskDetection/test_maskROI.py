import cv2
import os
import Core.PreProcessing.WithFaceDetect as FaceDetection
import Core.PreProcessing.Subsystem as PreProcessor
import Core.Camera.Subsystem as Camera
import Core.MaskDetection.SubSystem as MaskDetection
from tensorflow.keras.models import load_model


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

    camera.initialize()
    preprocessor.initialize()
    face_detection.initialize(faceNet)
    masknet = load_model("mask_detector.model")
    mask_detection = MaskDetection.SubSystem()

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

            # Mask Detection
            predictions = []
            mask_detection.runInference(prepared, predictions, masknet)

        # Drawing box around face location
        for box in locations:
            # first get the coordinates from the face
            (startX, startY, endX, endY) = box

            # resize the face coordinates to get mask ROI
            nstartX, nstartY, nendX, nendY = mask_detection.faceCoordsToMaskROI(startX, startY, endX, endY)

            # display mask ROI if mask is detected
            mask_detection.detectionBox(frame, nstartX, nstartY, nendX,nendY)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    camera.shutdown()


if __name__ == '__main__':
    main()
