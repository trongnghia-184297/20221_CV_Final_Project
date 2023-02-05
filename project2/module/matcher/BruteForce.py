import cv2


class BruteForce:
    def __init__(self, distance_measure=cv2.NORM_L2, crossCheck=True):
        self.matcher = cv2.BFMatcher(normType=distance_measure, crossCheck=crossCheck)

    def match(self, des1, des2):
        print("Matching...")
        matches = self.matcher.match(des1, des2)
        print("Matches after cross check:{}".format(len(matches)))
        return matches
