import cv2
import os
import Core.PreProcessing.WithFaceDetect as FaceDetection
import Core.PreProcessing.Subsystem as PreProcessor
import Core.Camera.Subsystem as Camera


def main():
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
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

    camera.initialize()
    preprocessor.initialize()
    face_detection.initialize(faceNet)

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
            (startX, startY, endX, endY) = box

            color = (0, 255, 0)  # GREEN outline
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    camera.shutdown()


if __name__ == '__main__':
    main()
