"""Add user-provided text to the input image."""

import cv2

from cv_lab_common import read_image, show_images


def add_text(image, text):
    result = image.copy()
    height, width = image.shape[:2]
    cv2.putText(result, text, (10, max(30, height // 2)), cv2.FONT_HERSHEY_SIMPLEX,
                max(0.5, min(width, height) / 700), (0, 0, 255), 2, cv2.LINE_AA)
    return result


if __name__ == "__main__":
    caption = input("Text to display on the image: ")
    show_images(("Image with text", add_text(read_image(), caption)))
