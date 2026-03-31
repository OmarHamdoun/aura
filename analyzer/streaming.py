import base64
import threading
import time
from collections import deque
from typing import Optional, Callable

import cv2
import numpy as np
from PIL import Image

# YOLO detector — optional, gracefully disabled if not available
try:
    from .yolo_detector import get_detector as _get_yolo_detector
    _YOLO_AVAILABLE = True
except Exception:
    _YOLO_AVAILABLE = False
    _get_yolo_detector = None


class ThreadedAnalyzerStream:
    """
    Background capture + analysis threads.
    - Capture thread  : grabs frames, updates latest_frame.
    - YOLO thread     : runs on every frame (~1.7ms), draws boxes, feeds policy.
    - VLM  thread     : runs every N frames (slow), produces semantic captions.
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
        yolo_enabled: bool = True,
        yolo_model_path: Optional[str] = None,
        yolo_model_path_b: Optional[str] = None,
        yolo_conf: float = 0.35,
        yolo_conf_b: Optional[float] = None,
        on_yolo: Optional[Callable[[dict], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
    ):
        self.source = source
        self.describer = describer
        self.every_n = max(1, int(every_n))
        self.analyze_every = max(1, int(analyze_every))
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

        # YOLO config
        self.yolo_enabled = yolo_enabled and _YOLO_AVAILABLE
        self.yolo_model_path = yolo_model_path
        self.yolo_model_path_b = yolo_model_path_b
        self.yolo_conf = yolo_conf
        self.yolo_conf_b = yolo_conf_b
        self.on_yolo = on_yolo
        self.on_stop = on_stop
        self._yolo_detector = None
        self._yolo_lock = threading.Lock()

        # YOLO state — latest annotated frame + observation
        self._yolo_frame: Optional[np.ndarray] = None
        self._yolo_obs: Optional[dict] = None
        self._yolo_frame_lock = threading.Lock()

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
        self.last_analyzed_fc = 0
        self.last_prompt = self.prompt

        self._t_cap  = None
        self._t_ana  = None
        self._t_yolo = None

    # ── lifecycle ────────────────────────────────────────────────────
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

        if self.yolo_enabled:
            self._t_yolo = threading.Thread(target=self._yolo_loop, daemon=True)
            self._t_yolo.start()

    def stop(self):
        self.running = False
        for t in (self._t_cap, self._t_ana, self._t_yolo):
            if t and t.is_alive():
                t.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.cap = None
        if self.on_stop:
            try:
                self.on_stop()
            except Exception as e:
                print(f"[stream] on_stop error: {e}", flush=True)

    # ── capture thread ───────────────────────────────────────────────
    def _capture_loop(self):
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
                if isinstance(self.source, str):
                    try:
                        fc = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        if fc and fc > 0:
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
                if self.frame_count % self.every_n == 0:
                    frame_ts = time.time()
                    self.latest_frame = frame.copy()
                    self.latest_frame_id = self.frame_count
                    self.latest_frame_ts = frame_ts
                    self.frame_buffer.append({
                        "frame":    self.latest_frame,
                        "frame_id": self.latest_frame_id,
                        "frame_ts": frame_ts,
                    })

    # ── YOLO thread ──────────────────────────────────────────────────
    def _yolo_loop(self):
        """Runs YOLO on every available frame. ~1.7ms per frame on GPU."""
        if not _YOLO_AVAILABLE or _get_yolo_detector is None:
            return

        detector = _get_yolo_detector(
            model_path=self.yolo_model_path,
            model_path_b=getattr(self, "yolo_model_path_b", None),
            conf=self.yolo_conf,
            conf_b=self.yolo_conf_b,
        )

        last_frame_id = -1
        while self.running:
            with self.frame_lock:
                frame = None if self.latest_frame is None else self.latest_frame.copy()
                fid   = self.latest_frame_id
                fts   = self.latest_frame_ts

            if frame is None or fid == last_frame_id:
                time.sleep(0.005)
                continue

            last_frame_id = fid

            try:
                result = detector.detect(frame)
                annotated = result["annotated"]
                obs       = result["obs"]

                with self._yolo_frame_lock:
                    self._yolo_frame = annotated
                    self._yolo_obs   = obs

                if self.on_yolo:
                    self.on_yolo({
                        "obs": obs,
                        "detections": result.get("detections") or [],
                        "frame_id": fid,
                        "frame_ts": fts,
                        "frame_shape": list(frame.shape[:2]),
                        "frame_bgr": frame,
                        "annotated_bgr": annotated,
                    })

            except Exception as e:
                print(f"[yolo] detect error: {e}", flush=True)
                time.sleep(0.01)

    # ── VLM analyze thread ───────────────────────────────────────────
    def _analyze_loop(self):
        while self.running:
            with self.frame_lock:
                frame    = None if self.latest_frame is None else self.latest_frame.copy()
                fc       = self.frame_count
                frame_id = self.latest_frame_id
                frame_ts = self.latest_frame_ts

            if frame is None:
                time.sleep(0.02)
                continue

            if (fc - self.last_analyzed_fc) >= self.analyze_every and \
               (time.time() - self.last_analyzed_ts) > 0.05:

                use_multi = (
                    self.multi_frames > 1
                    and len(self.frame_buffer) >= self.multi_frames
                    and getattr(self.describer, "supports_multi_image", False)
                )
                if use_multi:
                    with self.frame_lock:
                        buf_items = list(self.frame_buffer)[-self.multi_frames:]
                    pil = [Image.fromarray(cv2.cvtColor(item["frame"], cv2.COLOR_BGR2RGB))
                           for item in buf_items]
                    if self.include_thumbs:
                        self.last_thumbs = [
                            _encode_thumb_jpeg(item["frame"], width=180) for item in buf_items
                        ]
                    caption_meta = {
                        "frame_id":  buf_items[-1]["frame_id"],
                        "frame_ts":  buf_items[-1]["frame_ts"],
                        "frame_ids": [i["frame_id"] for i in buf_items],
                        "frame_tss": [i["frame_ts"] for i in buf_items],
                    }
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(rgb)
                    if self.include_thumbs:
                        self.last_thumbs = [_encode_thumb_jpeg(frame, width=220)]
                    caption_meta = {
                        "frame_id":  frame_id,
                        "frame_ts":  frame_ts,
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
                            pass
                    self.last_caption = final_caption
                    self.last_prompt  = prompt
                    if self.sync_overlay:
                        with self.frame_lock:
                            self.last_caption_frame    = frame.copy()
                            self.last_caption_frame_id = caption_meta.get("frame_id") or frame_id
                            self.last_caption_frame_ts = caption_meta.get("frame_ts") or frame_ts
                    if self.on_caption:
                        self.on_caption(final_caption, prompt, self.last_thumbs, caption_meta)
                except Exception as e:
                    self.last_caption = f"(analyze error: {e})"
                    self.last_prompt  = prompt
                    if self.sync_overlay:
                        with self.frame_lock:
                            self.last_caption_frame    = frame.copy()
                            self.last_caption_frame_id = caption_meta.get("frame_id") or frame_id
                            self.last_caption_frame_ts = caption_meta.get("frame_ts") or frame_ts
                    if self.on_caption:
                        self.on_caption(self.last_caption, prompt, self.last_thumbs, caption_meta)

                self.last_analyzed_ts = time.time()
                self.last_analyzed_fc = fc
            else:
                time.sleep(0.01)

    # ── read frame for MJPEG ─────────────────────────────────────────
    def read(self):
        """
        Returns the best available frame:
        - If YOLO is running: YOLO-annotated frame (boxes drawn) + VLM text overlay
        - Otherwise: plain frame + VLM text overlay
        """
        # Prefer YOLO-annotated frame
        if self.yolo_enabled:
            with self._yolo_frame_lock:
                yolo_frame = None if self._yolo_frame is None else self._yolo_frame.copy()
            if yolo_frame is not None:
                frame = yolo_frame
            else:
                # YOLO not ready yet — fall back to plain frame
                with self.frame_lock:
                    if self.latest_frame is None:
                        return None
                    frame = self.latest_frame.copy()
        else:
            with self.frame_lock:
                if self.latest_frame is None and self.last_caption_frame is None:
                    return None
                if self.sync_overlay and self.last_caption_frame is not None:
                    frame = self.last_caption_frame.copy()
                else:
                    frame = self.latest_frame.copy()

        # VLM caption text overlay (summary only, max 3 lines)
        if self.overlay and self.last_caption:
            overlay_text = _extract_overlay_text(self.last_caption)
            max_chars  = _max_chars_for_width(frame.shape[1])
            font_scale = _font_scale_for_width(frame.shape[1])
            line_h     = max(16, int(32 * font_scale))
            max_lines  = min(3, max(1, int((frame.shape[0] - 32) / line_h)))
            lines      = _wrap_text(overlay_text, width=max_chars)
            lines      = _clamp_lines(lines, max_lines=max_lines)
            bar_h      = len(lines) * line_h + 16
            bar        = frame.copy()
            cv2.rectangle(bar, (0, 0), (frame.shape[1], bar_h), (0, 0, 0), -1)
            cv2.addWeighted(bar, 0.45, frame, 0.55, 0, frame)
            for i, line in enumerate(lines):
                y = 24 + i * line_h
                cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        return frame

    def get_yolo_obs(self) -> Optional[dict]:
        """Return latest YOLO observation dict (for policy use)."""
        with self._yolo_frame_lock:
            return self._yolo_obs

    def _next_prompt(self):
        with self._prompt_lock:
            if not self._prompts:
                return "Give a short caption."
            p = self._prompts[self._prompt_idx % len(self._prompts)]
            self._prompt_idx += 1
            return p


# ── helpers ──────────────────────────────────────────────────────────

def _extract_overlay_text(caption: str) -> str:
    import json as _json, re as _re
    text = (caption or "").strip()
    try:
        obj = _json.loads(text)
        if isinstance(obj, dict) and obj.get("summary"):
            return str(obj["summary"]).strip()
    except Exception:
        pass
    m = _re.search(r'\{.*\}', text, flags=_re.DOTALL)
    if m:
        try:
            obj = _json.loads(m.group(0))
            if isinstance(obj, dict) and obj.get("summary"):
                return str(obj["summary"]).strip()
        except Exception:
            pass
    first = _re.split(r'[.\n]', text)[0].strip()
    return first[:120] if first else text[:120]


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
    data  = encode_jpeg(thumb, quality=70)
    b64   = base64.b64encode(data).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"

def _clamp_lines(lines, max_lines: int):
    if len(lines) <= max_lines:
        return lines
    kept = lines[:max_lines]
    last = kept[-1]
    kept[-1] = (last[:-1] + "…") if last else "…"
    return kept

def encode_jpeg(frame_bgr, quality=80):
    ok, buf = cv2.imencode(".jpg", frame_bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()


def mjpeg_generator(stream, fps_limit=20):
    boundary = b"--frame"
    period   = 1.0 / max(1, fps_limit)
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
