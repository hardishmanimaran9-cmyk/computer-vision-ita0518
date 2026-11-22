"""Segment pixels using lower and upper BGR threshold values."""

import cv2
import numpy as np

from cv_lab_common import read_image, show_images


def segment_by_threshold(image, lower=(0, 80, 0), upper=(120, 255, 120)):
    mask = cv2.inRange(image, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
    segmented = cv2.bitwise_and(image, image, mask=mask)
    return mask, segmented


if __name__ == "__main__":
    image = read_image()
    mask, segmented = segment_by_threshold(image)
    show_images(("Original", image), ("Threshold mask", mask), ("Segmented image", segmented))
