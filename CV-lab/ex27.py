"""Detect and mark faces using OpenCV's bundled Haar cascade."""

from pathlib import Path

import cv2

from cv_lab_common import read_image, show_images


def load_cascade(filename):
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(
            "Haar cascade detection is unavailable in this OpenCV build. "
            "Install an OpenCV 4.x build with CascadeClassifier and Haar cascade XML files."
        )
    path = Path(cv2.data.haarcascades) / filename
    if not path.is_file():
        raise FileNotFoundError(
            f"Required Haar cascade is missing: {path}. Install OpenCV with its Haar cascade data files."
        )
    cascade = cv2.CascadeClassifier(str(path))
    if cascade.empty():
        raise RuntimeError(f"Could not load face cascade: {path}")
    return cascade


def face_cascade():
    return load_cascade("haarcascade_frontalface_default.xml")


def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    result = image.copy()
    for x, y, width, height in faces:
        cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return result, faces


if __name__ == "__main__":
    result, faces = detect_faces(read_image())
    print(f"Faces detected: {len(faces)}")
    show_images(("Face detection", result))
