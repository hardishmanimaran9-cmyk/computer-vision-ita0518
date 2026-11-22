"""Morphological top-hat: image minus its opening."""

import cv2

from cv_lab_common import read_image, show_images


def top_hat(image, kernel_size=15):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)


if __name__ == "__main__":
    image = read_image()
    show_images(("Original", image), ("Top hat", top_hat(image)))
