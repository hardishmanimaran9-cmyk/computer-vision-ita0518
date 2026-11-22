"""Small helpers shared by the numbered OpenCV lab experiments."""

from pathlib import Path

import cv2


LAB_DIR = Path(__file__).resolve().parent


def image_path(filename: str = "input.jpg") -> Path:
    return LAB_DIR / filename


def read_image(filename: str = "input.jpg", flags: int = cv2.IMREAD_COLOR):
    path = image_path(filename)
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def read_video(filename: str = "input.mp4"):
    path = image_path(filename)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(f"Could not open video: {path}")
    return capture


def show_images(*named_images):
    """Show resizable, screen-sized image windows and wait for a key press."""
    if not named_images:
        return

    max_width = 800
    max_height = 650
    for title, image in named_images:
        height, width = image.shape[:2]
        scale = min(1, max_width / width, max_height / height)
        window_width = max(1, round(width * scale))
        window_height = max(1, round(height * scale))

        cv2.namedWindow(title, cv2.WINDOW_NORMAL)
        cv2.imshow(title, image)
        cv2.resizeWindow(title, window_width, window_height)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
