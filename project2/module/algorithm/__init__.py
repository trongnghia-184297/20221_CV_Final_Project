import sys

from module.algorithm.ASIFT import ASIFT
from module.algorithm.SIFT import SIFT


def get_algo(algo_name):
    # Available algorithm
    algorithms_factory = {
        "sift": SIFT,
        "asift": ASIFT,
    }
    if algo_name not in list(algorithms_factory.keys()):
        print('Unknown algorithm')
        sys.exit(1)
    return algorithms_factory[algo_name]()
