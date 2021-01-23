class SubSystem:
    # inputs:    preprocessed image data: unknown atm
    # outputs:   mask detection status: POSSIBLY FLAG?
    #           image data: unknown atm

    def __init__(self, face):
        self.face = face

    def initialize(self):
        print("Hello this is the MaskDetection subsystem")

    # prediction method (adrian rosebrock)
    def prediction(self):
        pass

#        pass the face through the model to determine if the face
# 		# has a mask or not
# 		(mask, withoutMask) = model.predict(face)[0]
#
# 		# determine the class label and color we'll use to draw
# 		# the bounding box and text
# 		label = "Mask" if mask > withoutMask else "No Mask"
# 		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
#
# 		# include the probability in the label
# 		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
