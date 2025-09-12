import threading
import time

class _Buffer:
    def __init__(self):
        self._items = []
        self._id = 0
        self._lock = threading.Lock()

    def add(self, text: str, source: str = ""):
        with self._lock:
            self._id += 1
            self._items.append({
                "id": self._id,
                "ts": int(time.time()),
                "text": text,
                "source": source,
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
