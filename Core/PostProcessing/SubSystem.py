import cv2


class SubSystem:
    def __init__(self):
        # TODO(Jett): this needs more information
        self.frame = None
        self.output = None
        self.mask = None
        self.noMask = None
        self.startX = None
        self.startY = None
        self.endX = None
        self.endY = None

    def initialize(self):
        print("Hello this is the postprocessor subsystem")

    def makeLabel(self, detection):
        mask = detection[0]
        noMask = detection[1]
        label = "Mask" if mask > noMask else "No Mask"
        return label

    def determineLabelColor(self, label):
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        return color

    def setOutput(self, frame):
        self.output = frame

    def probability(self, label, detection):
        mask = detection[0]
        noMask = detection[1]
        finalLabel = "{}: {:.2f}%".format(label, max(mask, noMask) * 100)
        return finalLabel

    def integrate(self, frame, label, color, startX, startY, endX, endY):
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    def prepareOutputFrame(self,
                           frame, detection, startX, startY, endX, endY,  # Mask Detection
                           shape, mouth_rects, nose_rects, facial_feature_flags  # Facial Feature Detection
                           ):
        # Mask Detection
        label = self.makeLabel(detection)
        color = self.determineLabelColor(label)
        finalLabel = self.probability(label, detection)

        # Facial Feature Detection
        # Draw facial feature dots
        if len(shape) > 0:
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        # Draw nose and mouth using haar cascade
        if len(nose_rects) > 0:
            for (nose_startX, nose_startY, nose_endX, nose_endY) in nose_rects:
                cv2.rectangle(frame, (nose_startX, nose_startY), (nose_endX, nose_endY), (0, 255, 0), 1)
                # There are more predictions but we only care about the first one
                break

        if len(mouth_rects) > 0:
            for (mouth_startX, mouth_startY, mouth_endX, mouth_endY) in mouth_rects:
                cv2.rectangle(frame, (mouth_startX, mouth_startY), (mouth_endX, mouth_endY), (0, 255, 0), 1)
                # There are more predictions but we only care about the first one
                break

        # Output Frame
        # TODO(LUIS): Annotate these in the image?
        if facial_feature_flags[0] == 0:
            print("\tNose detected. Please cover up your nose.")
        if facial_feature_flags[1] == 0:
            print("\tMouth detected. Please cover up your mouth.")
        if facial_feature_flags[2] == 0:
            print("\tFacial points detected.")

        self.setOutput(frame)
        self.integrate(self.output, finalLabel, color, startX, startY, endX, endY)
        return self.output

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
