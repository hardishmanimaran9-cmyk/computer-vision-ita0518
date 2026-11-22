

import cv2

from cv_lab_common import read_image, show_images


def erode(image, kernel_size=5, iterations=1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.erode(gray, kernel, iterations=iterations)


if __name__ == "__main__":
    image = read_image()
    show_images(("Original", image), ("Erosion", erode(image)))
