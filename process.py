import argparse
from ab_detector.detector import ABDetector


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='', help='source')
    args = parser.parse_args()

    detector = ABDetector()
    detector.process(args)