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

    def prepareOutputFrame(self, frame, detection, startX, startY, endX, endY):
        label = self.makeLabel(detection)
        color = self.determineLabelColor(label)
        finalLabel = self.probability(label, detection)
        self.setOutput(frame)
        self.integrate(self.output, finalLabel, color, startX, startY, endX, endY)
        return self.output

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
