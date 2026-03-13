import base64
import threading
import time
from collections import deque
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
        on_caption: Optional[Callable[[str, str, list, dict], None]] = None,
        prompt: str = "Give a short caption.",
        max_new_tokens: int = 96,
        multi_frames: int = 1,
        include_thumbs: bool = False,
        caption_postprocess: Optional[Callable[[str, str], str]] = None,
    ):
        self.source = source
        self.describer = describer
        self.every_n = max(1, int(every_n))
        self.analyze_every = max(1, int(analyze_every))  # frames between analyses
        self.overlay = overlay
        self.max_width = max_width
        self.on_caption = on_caption
        if isinstance(prompt, (list, tuple)):
            self._prompts = [p for p in prompt if p]
        else:
            self._prompts = [prompt] if prompt else ["Give a short caption."]
        self._prompt_idx = 0
        self._prompt_lock = threading.Lock()
        self.prompt = self._prompts[0]
        self.max_new_tokens = max_new_tokens
        self.multi_frames = max(1, int(multi_frames))
        self.frame_buffer = deque(maxlen=self.multi_frames)
        self.include_thumbs = include_thumbs
        self.caption_postprocess = caption_postprocess
        self.last_thumbs: list = []

        self.cap = None
        self.running = False
        self.frame_lock = threading.Lock()
        self.latest_frame: Optional[np.ndarray] = None
        self.latest_frame_id = 0
        self.latest_frame_ts = 0.0
        self.frame_count = 0
        self.source_fps = None
        self.last_caption_frame: Optional[np.ndarray] = None
        self.last_caption_frame_id = 0
        self.last_caption_frame_ts = 0.0
        self.sync_overlay = isinstance(source, str)

        self.last_caption = ""
        self.last_analyzed_ts = 0.0
        self.last_analyzed_fc = 0  # trigger when fc advanced by >= analyze_every
        self.last_prompt = self.prompt

        self._t_cap = None
        self._t_ana = None

    # ---------- lifecycle ----------
    def start(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {self.source}")
        if isinstance(self.source, str):
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 1:
                self.source_fps = fps
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
        last_ts = 0.0
        while self.running:
            if self.source_fps:
                target = 1.0 / self.source_fps
                now = time.time()
                if last_ts:
                    sleep_for = target - (now - last_ts)
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                last_ts = time.time()
            ok, frame = self.cap.read()
            if not ok:
                # If it's a file source, loop back to start when reaching EOF.
                if isinstance(self.source, str):
                    try:
                        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        if frame_count and frame_count > 0:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            time.sleep(0.01)
                            continue
                    except Exception:
                        pass
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
                    frame_ts = time.time()
                    self.latest_frame = frame.copy()
                    self.latest_frame_id = self.frame_count
                    self.latest_frame_ts = frame_ts
                    self.frame_buffer.append(
                        {
                            "frame": self.latest_frame,
                            "frame_id": self.latest_frame_id,
                            "frame_ts": frame_ts,
                        }
                    )

    def _analyze_loop(self):
        """
        Analyze when we have advanced by >= analyze_every frames since the last analysis.
        Also, never push an empty caption to the UI.
        """
        while self.running:
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                fc = self.frame_count
                frame_id = self.latest_frame_id
                frame_ts = self.latest_frame_ts

            if frame is None:
                time.sleep(0.02)
                continue

            # Trigger when enough frames have passed since the last analysis
            if (fc - self.last_analyzed_fc) >= self.analyze_every and (time.time() - self.last_analyzed_ts) > 0.05:
                use_multi = (
                    self.multi_frames > 1
                    and len(self.frame_buffer) >= self.multi_frames
                    and getattr(self.describer, "supports_multi_image", False)
                )
                if use_multi:
                    frames = []
                    with self.frame_lock:
                        buf_items = list(self.frame_buffer)[-self.multi_frames :]
                    for item in buf_items:
                        f = item["frame"]
                        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(rgb))
                    pil = frames
                    if self.include_thumbs:
                        self.last_thumbs = [
                            _encode_thumb_jpeg(item["frame"], width=180) for item in buf_items
                        ]
                    caption_meta = {
                        "frame_id": buf_items[-1]["frame_id"],
                        "frame_ts": buf_items[-1]["frame_ts"],
                        "frame_ids": [item["frame_id"] for item in buf_items],
                        "frame_tss": [item["frame_ts"] for item in buf_items],
                    }
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    if self.include_thumbs:
                        self.last_thumbs = [_encode_thumb_jpeg(frame, width=220)]
                    caption_meta = {
                        "frame_id": frame_id,
                        "frame_ts": frame_ts,
                        "frame_ids": [frame_id] if frame_id else [],
                        "frame_tss": [frame_ts] if frame_ts else [],
                    }
                prompt = self._next_prompt()
                try:
                    caption = self.describer.describe(
                        pil, prompt=prompt, max_new_tokens=self.max_new_tokens
                    )
                    caption = (caption or "").strip() or "(no caption generated)"
                    final_caption = caption
                    if self.caption_postprocess:
                        try:
                            processed = (self.caption_postprocess(caption, prompt) or "").strip()
                            if processed:
                                final_caption = processed
                        except Exception:
                            # Keep the original caption if post-processing fails.
                            pass
                    self.last_caption = final_caption
                    self.last_prompt = prompt
                    if self.sync_overlay:
                        with self.frame_lock:
                            self.last_caption_frame = frame.copy()
                            self.last_caption_frame_id = caption_meta.get("frame_id") or frame_id
                            self.last_caption_frame_ts = caption_meta.get("frame_ts") or frame_ts
                    if self.on_caption:
                        self.on_caption(final_caption, prompt, self.last_thumbs, caption_meta)
                except Exception as e:
                    self.last_caption = f"(analyze error: {e})"
                    self.last_prompt = prompt
                    if self.sync_overlay:
                        with self.frame_lock:
                            self.last_caption_frame = frame.copy()
                            self.last_caption_frame_id = caption_meta.get("frame_id") or frame_id
                            self.last_caption_frame_ts = caption_meta.get("frame_ts") or frame_ts
                    if self.on_caption:
                        self.on_caption(self.last_caption, prompt, self.last_thumbs, caption_meta)

                self.last_analyzed_ts = time.time()
                self.last_analyzed_fc = fc
            else:
                time.sleep(0.01)

    # ---------- read current frame (with overlay) ----------
    def read(self):
        with self.frame_lock:
            if self.latest_frame is None and self.last_caption_frame is None:
                return None
            if self.sync_overlay and self.last_caption_frame is not None:
                frame = self.last_caption_frame.copy()
            else:
                frame = self.latest_frame.copy()

        if self.overlay and self.last_caption:
            max_chars = _max_chars_for_width(frame.shape[1])
            font_scale = _font_scale_for_width(frame.shape[1])
            line_h = max(16, int(32 * font_scale))
            max_lines = max(1, int((frame.shape[0] - 32) / line_h))
            lines = _wrap_text(self.last_caption, width=max_chars)
            lines = _clamp_lines(lines, max_lines=max_lines)
            for i, line in enumerate(lines):
                y = 32 + i * line_h
                cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def _next_prompt(self):
        with self._prompt_lock:
            if not self._prompts:
                return "Give a short caption."
            prompt = self._prompts[self._prompt_idx % len(self._prompts)]
            self._prompt_idx += 1
            return prompt


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

def _max_chars_for_width(px_width: int):
    return max(26, int(px_width / 14))

def _font_scale_for_width(px_width: int):
    return max(0.5, min(1.2, px_width / 1400))

def _encode_thumb_jpeg(frame_bgr, width=200):
    h, w = frame_bgr.shape[:2]
    if w <= 0 or h <= 0:
        return ""
    scale = width / float(w)
    new_h = max(1, int(h * scale))
    thumb = cv2.resize(frame_bgr, (width, new_h))
    data = encode_jpeg(thumb, quality=70)
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def _clamp_lines(lines, max_lines: int):
    if len(lines) <= max_lines:
        return lines
    kept = lines[:max_lines]
    last = kept[-1]
    kept[-1] = (last[:-1] + "…") if last else "…"
    return kept


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
