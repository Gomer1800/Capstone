import Core.MaskDetection.SubSystem as MaskDetection
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import cv2


def main():
    masknet = load_model("mask_detector.model")
    predictions = []
    faces = []

    face = cv2.imread("testImages/withMask.jpg", 1)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)

    faces.append(face)

    mask_detection = MaskDetection.SubSystem()
    mask_detection.runInference(faces, predictions, masknet)


if __name__ == '__main__':
    main()
