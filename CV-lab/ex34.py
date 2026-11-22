"""Create a white canvas and draw a circle on it."""

import cv2
import numpy as np

from cv_lab_common import show_images


def create_circle(height, width):
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = max(1, min(width, height) // 4)
    cv2.circle(canvas, center, radius, (255, 0, 0), 3)
    return canvas


if __name__ == "__main__":
    height = int(input("Image height in pixels: "))
    width = int(input("Image width in pixels: "))
    show_images(("Circle", create_circle(height, width)))
