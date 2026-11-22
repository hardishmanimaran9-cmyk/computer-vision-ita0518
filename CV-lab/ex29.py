"""Detect eyes with the bundled face and eye Haar cascades."""

import cv2

from cv_lab_common import read_image, show_images
from ex27 import face_cascade, load_cascade


def detect_eyes(image):
    eyes_detector = load_cascade("haarcascade_eye.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = image.copy()
    faces = face_cascade().detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    for x, y, width, height in faces:
        face_gray = gray[y:y + height, x:x + width]
        for ex, ey, ew, eh in eyes_detector.detectMultiScale(face_gray, 1.1, 5, minSize=(12, 12)):
            cv2.rectangle(result, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
    return result


if __name__ == "__main__":
    show_images(("Eye detection", detect_eyes(read_image())))
