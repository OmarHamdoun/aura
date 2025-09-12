import threading
import time
from typing import Optional, Callable

import cv2
import numpy as np
from PIL import Image


class ThreadedAnalyzerStream:
    """
    Background capture + analysis threads.
    - Capture thread grabs frames and updates latest_frame.
    - Analyze thread runs the VLM periodically and pushes captions.
    """

    def __init__(
        self,
        source,
        describer,
        every_n: int = 2,
        analyze_every: int = 30,
        overlay: bool = True,
        max_width: int = 1920,
        on_caption: Optional[Callable[[str], None]] = None,
        prompt: str = "Give a short caption.",
        max_new_tokens: int = 96,
    ):
        self.source = source
        self.describer = describer
        self.every_n = max(1, int(every_n))
        self.analyze_every = max(1, int(analyze_every))  # frames between analyses
        self.overlay = overlay
        self.max_width = max_width
        self.on_caption = on_caption
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens

        self.cap = None
        self.running = False
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_count = 0

        self.last_caption = ""
        self.last_analyzed_ts = 0.0
        self.last_analyzed_fc = 0  # trigger when fc advanced by >= analyze_every

        self._t_cap = None
        self._t_ana = None

    # ---------- lifecycle ----------
    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")
        self.running = True
        self._t_cap = threading.Thread(target=self._capture_loop, daemon=True)
        self._t_ana = threading.Thread(target=self._analyze_loop, daemon=True)
        self._t_cap.start()
        self._t_ana.start()

    def stop(self):
        self.running = False
        if self._t_cap and self._t_cap.is_alive():
            self._t_cap.join(timeout=1.0)
        if self._t_ana and self._t_ana.is_alive():
            self._t_ana.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.cap = None

    # ---------- threads ----------
    def _capture_loop(self):
        """Continuously capture frames; keep every_n-th as the latest frame."""
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.02)
                continue

            if self.max_width and frame.shape[1] > self.max_width:
                s = self.max_width / frame.shape[1]
                frame = cv2.resize(
                    frame, (int(frame.shape[1] * s), int(frame.shape[0] * s))
                )

            with self.frame_lock:
                self.frame_count += 1
                # Keep only every_n-th frame for analysis/streaming
                if self.frame_count % self.every_n == 0:
                    self.latest_frame = frame.copy()

    def _analyze_loop(self):
        """
        Analyze when we have advanced by >= analyze_every frames since the last analysis.
        Also, never push an empty caption to the UI.
        """
        while self.running:
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                fc = self.frame_count

            if frame is None:
                time.sleep(0.02)
                continue

            # Trigger when enough frames have passed since the last analysis
            if (fc - self.last_analyzed_fc) >= self.analyze_every and (time.time() - self.last_analyzed_ts) > 0.05:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)
                try:
                    caption = self.describer.describe(
                        pil, prompt=self.prompt, max_new_tokens=self.max_new_tokens
                    )
                    caption = (caption or "").strip() or "(no caption generated)"
                    self.last_caption = caption
                    if self.on_caption:
                        self.on_caption(caption)
                except Exception as e:
                    self.last_caption = f"(analyze error: {e})"
                    if self.on_caption:
                        self.on_caption(self.last_caption)

                self.last_analyzed_ts = time.time()
                self.last_analyzed_fc = fc
            else:
                time.sleep(0.01)

    # ---------- read current frame (with overlay) ----------
    def read(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()

        if self.overlay and self.last_caption:
            for i, line in enumerate(_wrap_text(self.last_caption[:240], width=70)):
                y = 32 + i * 24
                cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return frame


# ---------- helpers ----------
def _wrap_text(s, width=60):
    words, line, lines = s.split(), [], []
    for w in words:
        if sum(len(x) for x in line) + len(line) + len(w) > width:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    return lines


def encode_jpeg(frame_bgr, quality=80):
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def mjpeg_generator(stream, fps_limit=20):
    boundary = b"--frame"
    period = 1.0 / max(1, fps_limit)
    try:
        stream.start()
        last = 0.0
        while True:
            frame = stream.read()
            if frame is None:
                time.sleep(0.01)
                continue
            now = time.time()
            if now - last < period:
                time.sleep(0.003)
                continue
            last = now
            jpeg = encode_jpeg(frame, quality=80)
            yield (
                boundary
                + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(jpeg)).encode()
                + b"\r\n\r\n"
                + jpeg
                + b"\r\n"
            )
    finally:
        stream.stop()
