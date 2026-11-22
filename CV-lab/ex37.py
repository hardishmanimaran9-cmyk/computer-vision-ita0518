"""Remove a chosen foreground color range (HSV color levels)."""

import cv2
import numpy as np

from cv_lab_common import read_image, show_images


def subtract_foreground(image, lower_hsv=(0, 80, 50), upper_hsv=(15, 255, 255)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    foreground_mask = cv2.inRange(
        hsv, np.array(lower_hsv, dtype=np.uint8), np.array(upper_hsv, dtype=np.uint8)
    )
    remaining = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(foreground_mask))
    return remaining, foreground_mask


if __name__ == "__main__":
    image = read_image()
    remaining, mask = subtract_foreground(image)
    show_images(("Original", image), ("Detected foreground", mask), ("Foreground removed", remaining))
