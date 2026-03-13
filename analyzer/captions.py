import threading
import time
import json
import re
from pathlib import Path

class _Buffer:
    def __init__(self):
        self._items = []
        self._id = 0
        self._lock = threading.Lock()

    def add(self, text: str, source: str = "", prompt: str = "", thumbs=None, frame_id=None, frame_ts=None, frame_ids=None, frame_tss=None):
        with self._lock:
            self._id += 1
            self._items.append({
                "id": self._id,
                "ts": int(time.time()),
                "text": text,
                "source": source,
                "prompt": prompt,
                "thumbs": thumbs or [],
                "frame_id": frame_id,
                "frame_ts": frame_ts,
                "frame_ids": frame_ids or ([] if frame_id is None else [frame_id]),
                "frame_tss": frame_tss or ([] if frame_ts is None else [frame_ts]),
            })

    def since(self, after_id: int):
        with self._lock:
            if after_id <= 0:
                return list(self._items[-100:])  # cap history
            return [x for x in self._items if x["id"] > after_id][-100:]

    def clear(self):
        with self._lock:
            self._items.clear()
            self._id = 0

    # NEW: return the last caption text (or None)
    def last_text(self):
        with self._lock:
            if not self._items:
                return None
            return self._items[-1]["text"]


# Global camera buffer
camera_captions = _Buffer()

# Per-video buffers
_video_buffers = {}
_vlock = threading.Lock()

def get_video_buffer(path: str) -> _Buffer:
    with _vlock:
        buf = _video_buffers.get(path)
        if buf is None:
            buf = _Buffer()
            _video_buffers[path] = buf
        return buf


_FILE_LOCK = threading.Lock()


def _safe_name(value: str) -> str:
    text = (value or "unknown").strip()
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    return text[:120] or "unknown"


def append_caption_jsonl(
    base_dir,
    stream_type: str,
    source: str,
    model: str,
    text: str,
    prompt: str,
    thumbs=None,
    frame_id=None,
    frame_ts=None,
    frame_ids=None,
    frame_tss=None,
):
    out_dir = Path(base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{_safe_name(stream_type)}__{_safe_name(model)}__{_safe_name(source)}.jsonl"
    out_path = out_dir / filename
    row = {
        "ts": int(time.time()),
        "stream_type": stream_type,
        "source": source,
        "model": model,
        "text": text,
        "prompt": prompt,
        "thumbs": thumbs or [],
        "frame_id": frame_id,
        "frame_ts": frame_ts,
        "frame_ids": frame_ids or ([] if frame_id is None else [frame_id]),
        "frame_tss": frame_tss or ([] if frame_ts is None else [frame_ts]),
    }
    line = json.dumps(row, ensure_ascii=False)
    with _FILE_LOCK:
        with out_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return str(out_path)
