"""Detect smiles inside faces using bundled Haar cascades."""

import cv2

from cv_lab_common import read_image, show_images
from ex27 import face_cascade, load_cascade


def detect_smiles(image):
    smiles_detector = load_cascade("haarcascade_smile.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = image.copy()
    faces = face_cascade().detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    for x, y, width, height in faces:
        face_gray = gray[y:y + height, x:x + width]
        lower_face = face_gray[height // 2:, :]
        for sx, sy, sw, sh in smiles_detector.detectMultiScale(
                lower_face, scaleFactor=1.7, minNeighbors=20, minSize=(20, 10)):
            cv2.rectangle(result, (x + sx, y + height // 2 + sy),
                          (x + sx + sw, y + height // 2 + sy + sh), (0, 255, 255), 2)
    return result


if __name__ == "__main__":
    show_images(("Smile detection", detect_smiles(read_image())))
