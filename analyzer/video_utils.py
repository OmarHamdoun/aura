# analyzer/video_utils.py
import cv2
from PIL import Image

def iter_frames(path_or_device, every_n_frames=30, max_frames=20):
    """
    Yields (PIL_image, timestamp_ms) from a video path or camera device index.
    """
    cap = cv2.VideoCapture(path_or_device)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {path_or_device}")

    count = 0
    yielded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if count % every_n_frames == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield Image.fromarray(rgb), int(cap.get(cv2.CAP_PROP_POS_MSEC))
                yielded += 1
                if yielded >= max_frames:
                    break
            count += 1
    finally:
        cap.release()
