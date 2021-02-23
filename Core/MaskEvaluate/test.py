import cv2
import os
import Core.PreProcessing.WithFaceDetect as FaceDetection
import Core.PreProcessing.Subsystem as PreProcessor
import Core.Camera.Subsystem as Camera
import Core.FacialFeatureDetection.Subsystem as FacialFeatureDetection
import Core.MaskEvaluate.Subsystem as MaskEvaluate


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

    predictor_path = "../FacialFeatureDetection/shape_predictor_68_face_landmarks.dat"
    mouth_xml = "../FacialFeatureDetection/cascade-files/haarcascade_mcs_mouth.xml"
    nose_xml = "../FacialFeatureDetection/cascade-files/haarcascade_mcs_nose.xml"

    camera = Camera.Subsystem(
        type="WEB",
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

    facial_feat_detection = FacialFeatureDetection.Subsystem(
        detector=None,
        predictor=None,
        mouthCascade=None,
        noseCascade=None
    )

    faceDots = 0
    mask_eval = MaskEvaluate.Subsystem(
        noseflag=None,
        mouthflag=None,
        numDots=faceDots,
        shapeflag=None
    )

    camera.initialize()
    preprocessor.initialize()
    face_detection.initialize(faceNet)
    facial_feat_detection.initialize(predictor_path, mouth_xml, nose_xml)

    while True:
        frame = camera.capture_image()

        # Face Detection Module
        # face_detection.setFrame(frame)
        # faces, locations = face_detection.runFaceDetect()

        # Pre Process Module
        # prepared = []
        # if len(faces) > 0:
        #     for face in faces:
        #         preprocessor.setFrame(face)
        #         modified = preprocessor.prepareFace()
        #         cv2.imshow("modified", modified)
        #         prepared.append(modified)

        # Facial Feature Module
        gray = preprocessor.cvtToGRAY(frame)
        shape = facial_feat_detection.detect_facial_landmarks(gray)
        mouth_rects = facial_feat_detection.cascade_detect(gray, "mouth")
        nose_rects = facial_feat_detection.cascade_detect(gray, "nose")

        # Draw facial feature dots
        if len(shape) > 0:
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # Draw nose and mouth using haar cascade
        if len(nose_rects) > 0:
            for (startX, startY, endX, endY) in nose_rects:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)

        if len(mouth_rects) > 0:
            for (startX, startY, endX, endY) in mouth_rects:
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 1)

        # Test if mask evaluation is working
        mask_eval.mask_evaluation(nose_rects, mouth_rects, shape)
        if mask_eval.noseflag == 0:
            print("Nose detected. Please cover up your nose.")
        if mask_eval.mouthflag == 0:
            print("Mouth detected. Please cover up your mouth.")
        if mask_eval.shapeflag == 0:
            print("Facial points detected.")

        # Drawing box around face location
        # for box in locations:
        #     (startX, startY, endX, endY) = box
        #
        #     color = (0, 255, 0)  # GREEN outline
        #     cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    camera.shutdown()


if __name__ == '__main__':
    main()
