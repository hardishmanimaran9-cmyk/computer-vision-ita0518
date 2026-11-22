"""Create a white canvas and draw a rectangle on it."""

import cv2
import numpy as np

from cv_lab_common import show_images


def create_rectangle(height, width):
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_x, margin_y = max(1, width // 5), max(1, height // 5)
    cv2.rectangle(canvas, (margin_x, margin_y), (width - margin_x - 1, height - margin_y - 1),
                  (0, 0, 255), 3)
    return canvas


if __name__ == "__main__":
    height = int(input("Image height in pixels: "))
    width = int(input("Image width in pixels: "))
    show_images(("Rectangle", create_rectangle(height, width)))
