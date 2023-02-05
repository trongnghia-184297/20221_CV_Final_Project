import cv2


class SIFT:
    def __init__(self, nOctaveLayers=3, contrastThreshold=0.04, edgeThreshold=10, sigma=1.6):
        self.detector = cv2.SIFT_create(nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold,
                                        edgeThreshold=edgeThreshold, sigma=sigma)

    def detectAndCompute(self, image):
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        print("Image gets {} keypoints".format(len(keypoints)))
        return keypoints, descriptors
