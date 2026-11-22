"""Count faces in the input image using a Haar cascade."""

from cv_lab_common import read_image
from ex27 import detect_faces


def count_faces(image):
    _, faces = detect_faces(image)
    return len(faces)


if __name__ == "__main__":
    print(f"Faces detected: {count_faces(read_image())}")
