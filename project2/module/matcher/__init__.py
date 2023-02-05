import sys

from module.matcher.BruteForce import BruteForce
from module.matcher.KDTree import KDTree


def get_matcher(matcher_name):
    # Available algorithm
    matcher_factory = {
        "bruteforce": BruteForce,
        "kdtree": KDTree,
    }
    if matcher_name not in list(matcher_factory.keys()):
        print('Unknown matcher')
        sys.exit(1)
    return matcher_factory[matcher_name]()
