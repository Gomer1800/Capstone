class Subsystem:
    # inputs: mask ROI, facial feature points, nose_rects and mouth_rects (kenneths new facial feature detection)
    # outputs:
    def __init__(self, noseflag, mouthflag, numDots, shapeflag):
        self.noseflag = noseflag
        self.mouthflag = mouthflag
        self.numDots = numDots
        self.shapeflag = shapeflag

    # takes in the list of nose rectangles and mouth rectangles
    # outputs a flag value depending on if nose rect or mouth rect is detected
    def mouth_nose_detection(self, nose_rects, mouth_rects, shape):
        self.noseflag = 1
        self.mouthflag = 1
        self.shapeflag = 1
        if len(nose_rects) > 0:
            self.noseflag = 0
        if len(mouth_rects) > 0:
            self.mouthflag = 0
        if len(shape) > 0:
            self.shapeflag = 0

    # comparing mask roi with nose and mouth facial feature dots
    # ASSUMING mask_roi has (startX, startY, endX, endY)
    # face_pts has (x,y)
    def compare_mask_face(self, mask_roi, face_pts):
        for (x, y) in face_pts:
            if mask_roi[0] <= x or mask_roi[2] >= x or mask_roi[1] >= y or mask_roi[3] <= y:
                self.numDots += 1
            else:
                continue

        if self.numDots >= 9:
            self.flag2 = 0
        else:
            self.flag2 = 1

    # 2 part system
    # tries to detect if any mouth/nose from facial detection
    # if none detected it goes to see comparing mask roi with facial feature points
    # outputs percentage of nose and mouth is covered
    def mask_evaluation(self, nose_rects, mouth_rects, shape):
        #, mask_roi, face_pts
        self.mouth_nose_detection(nose_rects, mouth_rects, shape)
        # else:
        #     self.compare_mask_face(mask_roi, face_pts)
        #     if self.flag2 == 0:
        #         print("Please use the mask to cover up your face more\n")
        #     else:
        #         # 29 is  the number of facial dots associated with nose and mouth
        #         percentage = (1 - (self.numDots / 29)) * 100
        #         print("Mask Coverage: %d", percentage)
