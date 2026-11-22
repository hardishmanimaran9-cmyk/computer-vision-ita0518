"""Locate a watch by matching it to a reference watch image with ORB features.

Place a clear watch reference photo in watch_template.jpg. A generic feature
matcher needs a reference image; it cannot identify an arbitrary object by
name without a trained object-recognition model.
"""

import cv2
import numpy as np

from cv_lab_common import image_path, read_image, show_images


def locate_watch(image, template):
    detector = cv2.ORB_create(nfeatures=1500)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key_template, desc_template = detector.detectAndCompute(template_gray, None)
    key_image, desc_image = detector.detectAndCompute(image_gray, None)
    if desc_template is None or desc_image is None:
        raise ValueError("Not enough visual features in the watch reference or input image")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_template, desc_image, k=2)
    good = [first for pair in pairs if len(pair) == 2
            for first, second in [pair] if first.distance < 0.75 * second.distance]
    if len(good) < 4:
        raise ValueError("Watch not located: add a clearer template or use a similar viewpoint")

    source_points = np.float32([key_template[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    target_points = np.float32([key_image[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(source_points, target_points, cv2.RANSAC, 5.0)
    if homography is None or inliers is None or int(inliers.sum()) < 4:
        raise ValueError("Watch reference features did not form a reliable match")
    height, width = template.shape[:2]
    corners = cv2.perspectiveTransform(
        np.float32([[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]]),
        homography,
    )[0].astype(int)
    result = image.copy()
    cv2.polylines(result, [corners], True, (0, 255, 0), 3)
    return result


if __name__ == "__main__":
    template_path = image_path("watch_template.jpg")
    template = cv2.imread(str(template_path))
    if template is None:
        raise FileNotFoundError(f"Add a watch reference image at: {template_path}")
    original = read_image()
    show_images(("Watch matching result", locate_watch(original, template)))
