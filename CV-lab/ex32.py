"""Create a white image with four colored boxes in its corners."""

import numpy as np

from cv_lab_common import show_images


def corner_boxes(height, width):
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    box_h, box_w = max(1, height // 10), max(1, width // 10)
    image[:box_h, :box_w] = (0, 0, 0)
    image[:box_h, width - box_w:] = (255, 0, 0)
    image[height - box_h:, :box_w] = (0, 255, 0)
    image[height - box_h:, width - box_w:] = (0, 0, 255)
    return image


if __name__ == "__main__":
    height = int(input("Image height in pixels: "))
    width = int(input("Image width in pixels: "))
    show_images(("White image with corner boxes", corner_boxes(height, width)))
