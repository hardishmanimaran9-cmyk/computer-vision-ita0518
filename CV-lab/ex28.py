"""Highlight vehicle-sized moving regions in a video with MOG2.

This is a classical motion/size heuristic, not a semantic vehicle classifier;
moving objects of similar size can also be marked.
"""

import cv2

from cv_lab_common import read_video


def detect_vehicle_regions(video="input.mp4"):
    capture = read_video(video)
    background = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=32, detectShadows=True)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            mask = background.apply(frame)
            _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, width, height = cv2.boundingRect(contour)
                ratio = width / max(height, 1)
                if width >= 45 and height >= 25 and 0.8 <= ratio <= 4.5:
                    cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                    cv2.putText(frame, "Moving vehicle-sized region", (x, max(20, y - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.imshow("Vehicle-region detection (q to quit)", frame)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_vehicle_regions()
