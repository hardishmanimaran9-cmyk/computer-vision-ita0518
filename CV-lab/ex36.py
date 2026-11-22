"""Remove pixels in a chosen background color range (HSV color levels).

Default range targets a green background; adjust the HSV limits for the image.
"""

import cv2
import numpy as np

from cv_lab_common import read_image, show_images


def subtract_background(image, lower_hsv=(35, 40, 40), upper_hsv=(85, 255, 255)):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    background_mask = cv2.inRange(
        hsv, np.array(lower_hsv, dtype=np.uint8), np.array(upper_hsv, dtype=np.uint8)
    )
    foreground = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(background_mask))
    return foreground, background_mask


if __name__ == "__main__":
    image = read_image()
    foreground, mask = subtract_background(image)
    show_images(("Original", image), ("Detected background", mask), ("Background removed", foreground))
