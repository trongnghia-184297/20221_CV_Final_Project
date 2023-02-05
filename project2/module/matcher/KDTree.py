import cv2


class KDTree:
    def __init__(self, numberOfTrees=5, depth=50):
        self.FLANN_INDEX_KDTREE = 1
        self.index_params = dict(algorithm=self.FLANN_INDEX_KDTREE, trees=numberOfTrees)
        self.search_params = dict(checks=depth)
        self.matcher = cv2.FlannBasedMatcher(self.index_params, self.search_params)

    def ratioFilter(self, matches):
        good = []
        for m, n in matches:
            if m.distance < 0.8 * n.distance:
                good.append(m)
        print("Matches after ratio test:{}".format(len(good)))
        return good

    def match(self, des1, des2):
        print("Matching...")
        matches = self.matcher.knnMatch(des1, des2, k=2)
        good_matches = self.ratioFilter(matches)

        return good_matches
