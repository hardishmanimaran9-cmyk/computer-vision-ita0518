"""Write a video with its frames in reverse order."""

import cv2

from cv_lab_common import image_path, read_video


def reverse_video(source="input.mp4", destination="reversed_video.mp4"):
    capture = read_video(source)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frame_count <= 0:
        capture.release()
        raise ValueError("The source video contains no readable frames")
    fps = fps if fps > 0 else 25.0
    output_path = image_path(destination)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create output video: {output_path}")
    try:
        for frame_index in range(frame_count - 1, -1, -1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise OSError(f"Could not read video frame {frame_index} from {source}")
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    return output_path


if __name__ == "__main__":
    print(f"Saved reversed video to {reverse_video()}")
