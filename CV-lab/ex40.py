"""Extract OCR text from sampled video frames using Tesseract OCR.

Requires the optional pytesseract Python package and the Tesseract executable.
"""

import cv2
from importlib import import_module

from cv_lab_common import read_video


def extract_video_text(video="input.mp4", frame_interval=30):
    if frame_interval < 1:
        raise ValueError("frame_interval must be at least 1")
    try:
        pytesseract = import_module("pytesseract")
    except ImportError as error:
        raise RuntimeError(
            "Experiment 40 requires pytesseract and the Tesseract OCR application. "
            "Install pytesseract and Tesseract, then rerun."
        ) from error

    capture = read_video(video)
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 25.0
    results = []
    frame_number = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % frame_interval == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                text = pytesseract.image_to_string(gray).strip()
                if text:
                    results.append((frame_number / fps, text))
            frame_number += 1
    finally:
        capture.release()
    return results


if __name__ == "__main__":
    for seconds, text in extract_video_text():
        print(f"[{seconds:.2f}s] {text}")
