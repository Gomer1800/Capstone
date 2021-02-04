import cv2
import os
import Core.PreProcessing.WithFaceDetect as FaceDetection
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
    camera.initialize()

    while True:
        frame = camera.capture_image()

        # Face Detection Module
        face_detection = FaceDetection.Subsystem(frame, faceNet)
        faces, locations = face_detection.runFaceDetect()

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
