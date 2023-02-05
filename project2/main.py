import argparse
import os
import sys
from itertools import compress

import numpy as np

from module.algorithm import get_algo
from module.matcher import get_matcher
from module.matcher.TransformMatch import transformMatch
from module.utility import *


def main(args):
    img1 = cv2.imread(args.template_img_path, 0)  # template img
    img2 = cv2.imread(args.detect_img_path, 0)  # detect img
    if img1 is None:
        print('Failed to load template image')
        sys.exit(1)

    if img2 is None:
        print('Failed to load detect image')
        sys.exit(1)
    detector = get_algo(args.algorithm)
    matcher = get_matcher(args.matcher)
    print("Using {}-{}".format(args.algorithm, args.matcher))
    kp1, des1 = detector.detectAndCompute(img1)
    kp2, des2 = detector.detectAndCompute(img2)
    matches = matcher.match(des1, des2)
    if len(matches) < int(args.MIN_MATCH_COUNT1):
        print("Not enough good matches are found - {}/{}".format(len(matches), args.MIN_MATCH_COUNT1))
        print("There is not the object in the image")
        sys.exit(1)
    else:
        M, matchesMask = transformMatch(kp1, kp2, matches)
        filterMatches = list(compress(matches, matchesMask))
        occur = len(set([i.trainIdx for i in filterMatches]))
        if occur < int(args.MIN_MATCH_COUNT2):
            print("Not enough unique match points are found - {}/{}".format(occur, args.MIN_MATCH_COUNT2))
            print("There is not the object in the image")
            sys.exit(1)
        else:
            w_min, h_min, w_max, h_max = getBbox(img1, M)
            h, w = img2.shape[:2]
            w_min = np.clip(w_min, 0, w - 1)
            h_min = np.clip(h_min, 0, h - 1)
            w_max = np.clip(w_max, 0, w - 1)
            h_max = np.clip(h_max, 0, h - 1)
            cv2.rectangle(img2, (int(w_min), int(h_min)), (int(w_max), int(h_max)), (255,255,0), 8)
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)
            img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
            cv2.imwrite(os.path.join(args.save_path, "Result.jpg"), img3)
            print("DONE! Result was saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--template_img_path', help="Path to the template image",
                        default='data/templates/IMG_0613.jpeg')
    parser.add_argument('--detect_img_path', help="Path to the detect image", default='data/images/1674896240445.jpg')
    parser.add_argument('--algorithm', help="Current implementation has sift|asift algorithm", default='sift')
    parser.add_argument('--matcher', help="Current implementation has bruteforce|kdtree matcher", default='bruteforce')
    parser.add_argument('--MIN_MATCH_COUNT1', help="Threshold for good matches", default=60)
    parser.add_argument('--MIN_MATCH_COUNT2', help="Threshold for unique matched points", default=30)
    parser.add_argument('--save_path', default='./')
    args = parser.parse_args()
    main(args)
