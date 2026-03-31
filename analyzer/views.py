import json
import os
import threading
import time
import tempfile
import subprocess
import base64
import binascii
import uuid
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from django.http import HttpResponse, HttpResponseBadRequest, StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.conf import settings
from pathlib import Path
from django.views.decorators.csrf import csrf_exempt
import cv2

from .inference import FastVLMDescriber, QwenVLDescriber, MiniInternVL2DriveLMDescriber
from .streaming import ThreadedAnalyzerStream, mjpeg_generator
from .captions import camera_captions, get_video_buffer, append_caption_jsonl
from .policy import decide_core, coerce_obs  # heuristic policy + scene-state normalization

try:
    import mlflow
except Exception:
    mlflow = None


# --------------------------------------------------------------------
# Built-in strict JSON prompt (used when enforce_json=1 in query string)
# --------------------------------------------------------------------
FISHEYE_JSON_PROMPT = """You see a circular fisheye image from a robot (robot at center).
Bearings: 0°=forward, 90°=right, 180°=back, 270°=left. Distances: near≈≤2 m, mid≈2–4 m, far>4 m.
Return STRICT JSON ONLY, no prose:

{
  "summary": "<<=18 words>",
  "obstacles": [
    {
      "name": "<class>",               // e.g., person, chair, toolbox, cable
      "bearing_deg": 0,                // 0..359 (0=fwd)
      "clock": "12",                   // 12/1/2/.../11 (optional mapping)
      "distance": "near|mid|far",
      "notes": "<short note>",
      "confidence": 0.0
    }
  ],
  "move": { "direction": "forward|back|left|right|hold", "reason": "<one line>" },
  "risks": ["trip hazard: cable", "moving person", "cluttered floor"]
}
Use "unknown" if uncertain; keep arrays short and relevant.
"""

SCENE_STATE_JSON_PROMPT = """You are a robot scene-understanding model.
Analyze the current image or image sequence and return STRICT JSON ONLY.
Do not choose an action. Do not include recommendations. Only describe scene state.

{
  "summary": "<<=18 words>",
  "obstacles": [
    {
      "name": "<class>",
      "bearing_deg": 0,
      "clock": "12",
      "position": "left|center|right|unknown",
      "distance": "near|mid|far|unknown",
      "notes": "<short note>",
      "confidence": 0.0
    }
  ],
  "risks": ["trip hazard: cable", "moving person", "narrow passage"]
}

Use "unknown" when uncertain. Keep arrays short and relevant.
"""


# -------------------------------------------------
# Model cache: keep one instance per model key
# -------------------------------------------------
_DESCRIBERS = {}
_DLOCK = threading.Lock()
_MLFLOW_LOCK = threading.Lock()
_MLFLOW_READY = False
_MLFLOW_SEQ = 0
_ALLOWED_DIRECTIONS = {"forward", "back", "left", "right", "hold"}

def get_describer(model_key: str):
    key = (model_key or "internvl2").lower()
    if key not in ("fastvlm", "qwen", "internvl2"):
        key = "internvl2"
    with _DLOCK:
        if key not in _DESCRIBERS:
            if key == "qwen":
                _DESCRIBERS[key] = QwenVLDescriber()
            elif key == "internvl2":
                _DESCRIBERS[key] = MiniInternVL2DriveLMDescriber()
            else:
                _DESCRIBERS[key] = FastVLMDescriber()
        return _DESCRIBERS[key]


def _init_mlflow():
    global _MLFLOW_READY
    if _MLFLOW_READY:
        return True
    if mlflow is None:
        return False
    uri = (getattr(settings, "MLFLOW_TRACKING_URI", "") or "").strip()
    if uri:
        mlflow.set_tracking_uri(uri)
    exp_name = (getattr(settings, "MLFLOW_EXPERIMENT_NAME", "aura-captions") or "aura-captions").strip()
    mlflow.set_experiment(exp_name)
    _MLFLOW_READY = True
    return True


def _persist_caption_mlflow(stream_type: str, source: str, model_key: str, txt: str, prm: str, thumbs, meta=None):
    meta = meta or {}
    global _MLFLOW_SEQ
    if not getattr(settings, "MLFLOW_SAVE_ENABLED", False):
        return
    if not _init_mlflow():
        return
    try:
        with _MLFLOW_LOCK:
            _MLFLOW_SEQ += 1
            seq = _MLFLOW_SEQ
        every_n = max(1, int(getattr(settings, "MLFLOW_LOG_EVERY_N", 1)))
        if seq % every_n != 0:
            return
        now = int(time.time())
        run_prefix = (getattr(settings, "MLFLOW_RUN_NAME_PREFIX", "caption") or "caption").strip()
        run_name = f"{run_prefix}-{stream_type}-{model_key}-{now}-{seq}"
        with mlflow.start_run(run_name=run_name):
            obs = _extract_json_anywhere(txt or "")
            risks = []
            if isinstance(obs, dict):
                raw_risks = obs.get("risks") or []
                if isinstance(raw_risks, list):
                    risks = [str(r).strip() for r in raw_risks if str(r).strip()]
            action_hint = ""
            if isinstance(obs, dict):
                move = obs.get("move") or {}
                if isinstance(move, dict):
                    action_hint = str(move.get("reason") or "").strip()
            driver_warning = ""
            if risks:
                top = "; ".join(risks[:3])
                driver_warning = f"Hazard ahead: {top}. Proceed with caution."
            elif action_hint:
                driver_warning = action_hint[:240]

            mlflow.log_params(
                {
                    "stream_type": stream_type,
                    "source": source,
                    "model": model_key,
                }
            )
            mlflow.log_metric("caption_length", len(txt or ""))
            mlflow.log_metric("thumbs_count", len(thumbs or []))
            mlflow.log_metric("hazard_count", len(risks))
            mlflow.set_tag("has_hazard", "true" if risks else "false")
            payload = {
                "ts": now,
                "stream_type": stream_type,
                "source": source,
                "model": model_key,
                "prompt": prm,
                "text": txt,
                "thumbs": thumbs or [],
                "frame_id": meta.get("frame_id"),
                "frame_ts": meta.get("frame_ts"),
                "frame_ids": meta.get("frame_ids") or [],
                "frame_tss": meta.get("frame_tss") or [],
                "risks": risks,
                "driver_warning": driver_warning,
            }
            mlflow.log_dict(payload, "caption.json")
            if _parse_bool(getattr(settings, "MLFLOW_LOG_THUMB_IMAGES", True), default=True):
                max_thumbs = max(1, int(getattr(settings, "MLFLOW_MAX_THUMB_IMAGES", 1)))
                for idx, thumb in enumerate((thumbs or [])[:max_thumbs]):
                    artifact = _thumb_to_artifact(thumb, idx)
                    if artifact:
                        mlflow.log_artifact(artifact["path"], artifact_path="thumbs")
                        try:
                            os.remove(artifact["path"])
                        except Exception:
                            pass
    except Exception as e:
        print(f"[mlflow] persist failed: {e}", flush=True)


def _thumb_to_artifact(thumb, idx: int):
    if not isinstance(thumb, str):
        return None
    s = thumb.strip()
    if not s.startswith("data:image/"):
        return None
    try:
        header, b64 = s.split(",", 1)
    except ValueError:
        return None
    if ";base64" not in header:
        return None
    mime = header[len("data:"):].split(";")[0].strip().lower()
    ext_map = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    ext = ext_map.get(mime, ".img")
    try:
        blob = base64.b64decode(b64, validate=True)
    except (ValueError, binascii.Error):
        return None
    if not blob:
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, prefix=f"thumb_{idx}_") as f:
        f.write(blob)
        return {"path": f.name}


def _persist_caption(stream_type: str, source: str, model_key: str, txt: str, prm: str, thumbs, meta=None):
    meta = meta or {}
    if getattr(settings, "CAPTIONS_SAVE_ENABLED", False):
        try:
            append_caption_jsonl(
                base_dir=getattr(settings, "CAPTIONS_SAVE_DIR", Path(settings.BASE_DIR) / "media" / "captions"),
                stream_type=stream_type,
                source=source,
                model=model_key,
                text=txt,
                prompt=prm,
                thumbs=thumbs,
                frame_id=meta.get("frame_id"),
                frame_ts=meta.get("frame_ts"),
                frame_ids=meta.get("frame_ids"),
                frame_tss=meta.get("frame_tss"),
            )
        except Exception as e:
            print(f"[captions] persist failed: {e}", flush=True)
    _persist_caption_mlflow(stream_type, source, model_key, txt, prm, thumbs, meta=meta)


def _safe_token(value: str) -> str:
    value = (value or "").strip()
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    cleaned = cleaned.strip("._")
    return (cleaned[:120] or "unknown")


class _HospitalEvalSession:
    def __init__(self, source: str, model_key: str, prompt: str, yolo_enabled: bool):
        self.source = source
        self.model_key = model_key
        self.prompt = prompt
        self.yolo_enabled = bool(yolo_enabled)
        self.started_at = int(time.time())
        self.session_id = f"{self.started_at}-{uuid.uuid4().hex[:8]}"
        self.root_dir = Path(getattr(settings, "HOSPITAL_EVAL_SAVE_DIR", Path(settings.BASE_DIR) / "media" / "hospital_eval")) / (
            f"{_safe_token(Path(source).stem)}__{self.session_id}"
        )
        self.frames_dir = self.root_dir / "frames"
        self.labels_dir = self.root_dir / "labels"
        self.meta_dir = self.root_dir / "meta"
        self._lock = threading.Lock()
        self._class_to_id = {}
        self._run_id = None
        self._mlflow_client = None
        self._caption_count = 0
        self._yolo_frame_count = 0
        self._last_frame_id = 0

        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self._write_json("session.json", {
            "session_id": self.session_id,
            "source": source,
            "model": model_key,
            "prompt": prompt,
            "yolo_enabled": self.yolo_enabled,
            "started_at": self.started_at,
        })
        self._start_mlflow_run()

    def _write_json(self, name: str, payload):
        path = self.meta_dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _append_jsonl(self, name: str, row):
        path = self.meta_dir / name
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return path

    def _start_mlflow_run(self):
        if not getattr(settings, "MLFLOW_SAVE_ENABLED", False):
            return
        if not _init_mlflow():
            return
        try:
            run_name = f"hospital-eval-{_safe_token(Path(self.source).stem)}-{self.session_id}"
            run = mlflow.start_run(run_name=run_name)
            self._run_id = run.info.run_id
            self._mlflow_client = mlflow.tracking.MlflowClient()
            mlflow.set_tags(
                {
                    "eval_type": "hospital_video",
                    "stream_type": "video",
                    "source_path": self.source,
                }
            )
            mlflow.log_params(
                {
                    "source": self.source,
                    "model": self.model_key,
                    "prompt": self.prompt[:500],
                    "yolo_enabled": str(self.yolo_enabled).lower(),
                    "session_id": self.session_id,
                }
            )
            mlflow.end_run()
        except Exception as e:
            print(f"[mlflow] hospital eval start failed: {e}", flush=True)
            self._run_id = None
            self._mlflow_client = None

    def _ensure_class_id(self, name: str) -> int:
        cls_name = str(name or "unknown")
        if cls_name not in self._class_to_id:
            self._class_to_id[cls_name] = len(self._class_to_id)
            self._write_json("classes.json", {
                "class_to_id": self._class_to_id,
                "id_to_class": {str(v): k for k, v in self._class_to_id.items()},
            })
        return self._class_to_id[cls_name]

    def _detections_to_yolo_lines(self, detections, width: int, height: int):
        lines = []
        rows = []
        for det in detections or []:
            box = det.get("box") or []
            if len(box) != 4 or width <= 0 or height <= 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in box]
            x1 = max(0.0, min(x1, width))
            x2 = max(0.0, min(x2, width))
            y1 = max(0.0, min(y1, height))
            y2 = max(0.0, min(y2, height))
            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw <= 0 or bh <= 0:
                continue
            cls_name = str(det.get("class") or "unknown")
            cls_id = self._ensure_class_id(cls_name)
            xc = ((x1 + x2) / 2.0) / float(width)
            yc = ((y1 + y2) / 2.0) / float(height)
            wn = bw / float(width)
            hn = bh / float(height)
            conf = float(det.get("conf") or 0.0)
            lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f} {conf:.6f}")
            rows.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "confidence": conf,
                "xyxy": [x1, y1, x2, y2],
                "xywhn": [xc, yc, wn, hn],
                "model": det.get("model"),
                "distance": det.get("distance"),
                "position": det.get("position"),
            })
        return lines, rows

    def log_caption(self, txt: str, prompt: str, thumbs, meta=None):
        meta = meta or {}
        row = {
            "ts": int(time.time()),
            "type": "caption",
            "frame_id": meta.get("frame_id"),
            "frame_ts": meta.get("frame_ts"),
            "frame_ids": meta.get("frame_ids") or [],
            "frame_tss": meta.get("frame_tss") or [],
            "prompt": prompt,
            "text": txt,
            "thumb_count": len(thumbs or []),
        }
        with self._lock:
            self._caption_count += 1
            path = self._append_jsonl("captions.jsonl", row)
            run_id = self._run_id
            client = self._mlflow_client
            caption_count = self._caption_count
        if run_id and client:
            try:
                ts_ms = int(time.time() * 1000)
                client.log_metric(run_id, "caption_events", caption_count, timestamp=ts_ms, step=caption_count)
                client.log_artifact(run_id, str(path), artifact_path="meta")
            except Exception as e:
                print(f"[mlflow] hospital caption log failed: {e}", flush=True)

    def log_yolo_frame(self, payload: dict):
        if not isinstance(payload, dict):
            return
        obs = payload.get("obs") or {}
        detections = payload.get("detections") or []
        frame_id = int(payload.get("frame_id") or 0)
        frame_ts = payload.get("frame_ts")
        frame_shape = payload.get("frame_shape") or []
        frame_bgr = payload.get("frame_bgr")
        if len(frame_shape) != 2:
            if frame_bgr is None:
                return
            frame_shape = list(frame_bgr.shape[:2])
        height, width = int(frame_shape[0]), int(frame_shape[1])

        label_lines, det_rows = self._detections_to_yolo_lines(detections, width, height)
        with self._lock:
            self._yolo_frame_count += 1
            self._last_frame_id = max(self._last_frame_id, frame_id)

        frame_path = None
        if getattr(settings, "HOSPITAL_EVAL_SAVE_YOLO", True):
            label_path = self.labels_dir / f"{frame_id:06d}.txt"
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")
            if frame_bgr is not None:
                frame_path = self.frames_dir / f"{frame_id:06d}.jpg"
                try:
                    ok, data = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                    if ok:
                        frame_path.write_bytes(data.tobytes())
                except Exception as e:
                    print(f"[hospital-eval] frame save failed: {e}", flush=True)
        else:
            label_path = None
            frame_path = None

        row = {
            "ts": int(time.time()),
            "type": "yolo_frame",
            "frame_id": frame_id,
            "frame_ts": frame_ts,
            "frame_size": {"width": width, "height": height},
            "obs": obs,
            "detections": det_rows,
            "label_path": str(label_path) if label_path else "",
            "frame_path": str(frame_path) if frame_path else "",
        }
        self._append_jsonl("yolo_frames.jsonl", row)

        if self._run_id and self._mlflow_client:
            try:
                step = max(frame_id, self._yolo_frame_count)
                ts_ms = int(time.time() * 1000)
                self._mlflow_client.log_metric(self._run_id, "yolo_frames", self._yolo_frame_count, timestamp=ts_ms, step=step)
                self._mlflow_client.log_metric(self._run_id, "yolo_detections", len(det_rows), timestamp=ts_ms, step=step)
                self._mlflow_client.log_metric(self._run_id, "yolo_risks", len((obs or {}).get("risks") or []), timestamp=ts_ms, step=step)
                self._mlflow_client.log_artifact(self._run_id, str(self.meta_dir / "yolo_frames.jsonl"), artifact_path="meta")
                classes_path = self.meta_dir / "classes.json"
                if classes_path.exists():
                    self._mlflow_client.log_artifact(self._run_id, str(classes_path), artifact_path="meta")
                if label_path and label_path.exists():
                    self._mlflow_client.log_artifact(self._run_id, str(label_path), artifact_path="yolo_labels")
                if frame_path and frame_path.exists():
                    self._mlflow_client.log_artifact(self._run_id, str(frame_path), artifact_path="frames")
            except Exception as e:
                print(f"[mlflow] hospital yolo log failed: {e}", flush=True)

    def finish(self):
        summary = {
            "session_id": self.session_id,
            "source": self.source,
            "model": self.model_key,
            "prompt": self.prompt,
            "caption_count": self._caption_count,
            "yolo_frame_count": self._yolo_frame_count,
            "last_frame_id": self._last_frame_id,
            "class_count": len(self._class_to_id),
            "classes": self._class_to_id,
            "root_dir": str(self.root_dir),
            "finished_at": int(time.time()),
        }
        summary_path = self._write_json("summary.json", summary)
        if self._run_id and self._mlflow_client:
            try:
                ts_ms = int(time.time() * 1000)
                self._mlflow_client.log_metric(self._run_id, "caption_count_final", self._caption_count, timestamp=ts_ms, step=self._caption_count)
                self._mlflow_client.log_metric(self._run_id, "yolo_frame_count_final", self._yolo_frame_count, timestamp=ts_ms, step=self._yolo_frame_count)
                self._mlflow_client.log_metric(self._run_id, "class_count_final", len(self._class_to_id), timestamp=ts_ms, step=max(self._yolo_frame_count, 1))
                self._mlflow_client.log_artifact(self._run_id, str(summary_path), artifact_path="meta")
                captions_path = self.meta_dir / "captions.jsonl"
                yolo_frames_path = self.meta_dir / "yolo_frames.jsonl"
                if captions_path.exists():
                    self._mlflow_client.log_artifact(self._run_id, str(captions_path), artifact_path="meta")
                if yolo_frames_path.exists():
                    self._mlflow_client.log_artifact(self._run_id, str(yolo_frames_path), artifact_path="meta")
                classes_path = self.meta_dir / "classes.json"
                if classes_path.exists():
                    self._mlflow_client.log_artifact(self._run_id, str(classes_path), artifact_path="meta")
            except Exception as e:
                print(f"[mlflow] hospital eval finish failed: {e}", flush=True)


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _get_int(request, name, default, lo, hi):
    try:
        v = int(request.GET.get(name, default))
    except Exception:
        v = default
    return max(lo, min(hi, v))

def _get_str(request, name, default):
    v = request.GET.get(name, "")
    if v is None:
        return default
    v = v.strip()
    return v if v else default

def _maybe_multi_prompt(prompt: str):
    prefix = "NAV_MULTI::"
    if isinstance(prompt, str) and prompt.startswith(prefix):
        rest = prompt[len(prefix):]
        parts = [p.strip() for p in rest.split("||") if p.strip()]
        return parts if parts else ""
    return prompt


def _extract_json_anywhere(text: str):
    """
    Be tolerant: find a JSON object even if the model added prose around it.
    Strategy:
      1) Try json.loads(text).
      2) Scan for the last balanced {...} block and try to parse it.
    Returns dict or None.
    """
    if not text:
        return None
    s = text.strip()
    # direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # find any balanced {...} from the end (most recent object)
    opens = [i for i, ch in enumerate(s) if ch == "{"]
    for start in reversed(opens):
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    cand = s[start:i+1]
                    try:
                        return json.loads(cand)
                    except Exception:
                        break  # try earlier '{'
    return None


def _normalize_action(action_like):
    if not isinstance(action_like, dict):
        return None
    direction = str(action_like.get("direction") or "").strip().lower()
    if direction not in _ALLOWED_DIRECTIONS:
        return None
    reason = str(action_like.get("reason") or "").strip()
    if not reason:
        reason = "Selected safest direction from current observation."
    return {"direction": direction, "reason": reason[:240]}


def _normalize_policy(policy_like):
    if policy_like is None:
        return ""
    text = str(policy_like).strip()
    if not text:
        return ""
    max_chars = max(60, int(getattr(settings, "ACTION_OPENAI_POLICY_MAX_CHARS", 280)))
    return text[:max_chars]


def _resolve_decision_provider(provider_override=None):
    raw = (provider_override or getattr(settings, "ACTION_POLICY_PROVIDER", "heuristic") or "heuristic").strip().lower()
    if raw in {"heuristic", "openai"}:
        return raw
    return "heuristic"


def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _caption_has_hazard(text: str) -> bool:
    clean = str(text or "").strip()
    if not clean:
        return False

    obs = _extract_json_anywhere(clean)
    if isinstance(obs, dict):
        raw_risks = obs.get("risks") or []
        if isinstance(raw_risks, list) and any(str(r).strip() for r in raw_risks):
            return True
        move = obs.get("move") or {}
        if isinstance(move, dict):
            direction = str(move.get("direction") or "").strip().lower()
            if direction in {"hold", "back", "left", "right"}:
                return True

    text_l = clean.lower()
    hazard_terms = (
        "hazard",
        "danger",
        "warning",
        "collision",
        "trip hazard",
        "blocked",
        "obstacle ahead",
        "stop",
        "hold",
        "wait",
        "slow down",
    )
    return any(term in text_l for term in hazard_terms)


def _fallback_policy(action: dict, instruction: str):
    direction = (action or {}).get("direction", "hold")
    reason = (action or {}).get("reason", "")
    prefix = (instruction or "").strip()
    if prefix:
        return _normalize_policy(f"Goal: {prefix}. Move: {direction}. {reason}")
    return _normalize_policy(f"Move: {direction}. {reason}")


def _decide_action_openai(obs_like, instruction: str, include_policy: bool = False):
    api_key = (os.environ.get("OPENAI_API_KEY", "") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    model = (getattr(settings, "ACTION_OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip()
    timeout_s = max(3, int(getattr(settings, "ACTION_OPENAI_TIMEOUT", 15)))
    endpoint = (getattr(settings, "ACTION_OPENAI_URL", "https://api.openai.com/v1/chat/completions") or "").strip()
    if not endpoint:
        raise RuntimeError("ACTION_OPENAI_URL is empty.")

    obs_text = json.dumps(obs_like, ensure_ascii=False) if isinstance(obs_like, dict) else str(obs_like or "")
    output_shape = (
        "Return: {\"direction\":\"...\",\"reason\":\"...\",\"policy\":\"...\"}"
        if include_policy
        else "Return: {\"direction\":\"...\",\"reason\":\"...\"}"
    )
    policy_note = (
        " Also include policy: short 1-3 step safety policy."
        if include_policy
        else ""
    )
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a robot navigation policy. Return strict JSON only with keys: "
                    "direction and reason. direction must be one of: forward, back, left, right, hold."
                    + policy_note
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Instruction: {instruction or 'Navigate safely and avoid near obstacles.'}\n"
                    f"Observation:\n{obs_text}\n"
                    + output_shape
                ),
            },
        ],
    }
    req = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")

    parsed = json.loads(body)
    content = (
        (parsed.get("choices") or [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    action_obj = _extract_json_anywhere(content) if isinstance(content, str) else None
    action = _normalize_action(action_obj)
    if action is None:
        raise RuntimeError("OpenAI response did not contain a valid action JSON.")
    policy = _normalize_policy((action_obj or {}).get("policy")) if include_policy else ""
    return action, policy


def _decide_action(obs_like, instruction: str, include_policy: bool = False, provider_override=None):
    provider = _resolve_decision_provider(provider_override)
    if provider == "openai":
        try:
            action, policy = _decide_action_openai(obs_like, instruction, include_policy=include_policy)
            return action, policy, "openai", ""
        except Exception as e:
            fallback = decide_core(obs_like, instruction)
            policy = _fallback_policy(fallback, instruction) if include_policy else ""
            return fallback, policy, "heuristic", f"OpenAI action failed, heuristic fallback used: {e}"
    fallback = decide_core(obs_like, instruction)
    policy = _fallback_policy(fallback, instruction) if include_policy else ""
    return fallback, policy, "heuristic", ""


def _summarize_caption_openai(text: str, prompt: str):
    api_key = (os.environ.get("OPENAI_API_KEY", "") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    model = (getattr(settings, "CAPTION_SUMMARY_OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip()
    timeout_s = max(3, int(getattr(settings, "CAPTION_SUMMARY_OPENAI_TIMEOUT", 12)))
    endpoint = (getattr(settings, "ACTION_OPENAI_URL", "https://api.openai.com/v1/chat/completions") or "").strip()
    if not endpoint:
        raise RuntimeError("ACTION_OPENAI_URL is empty.")

    max_chars = max(40, int(getattr(settings, "CAPTION_SUMMARY_MAX_CHARS", 180)))
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You summarize visual model outputs for realtime overlay and TTS. "
                    f"Return plain text only, one sentence, max {max_chars} characters."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize the following model output into one clear sentence focused on key scene facts "
                    "and immediate navigation/safety relevance.\n"
                    f"Prompt context: {prompt or 'N/A'}\n"
                    f"Model output:\n{text}"
                ),
            },
        ],
    }
    req = Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    parsed = json.loads(body)
    content = (
        (parsed.get("choices") or [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    out = str(content or "").strip()
    if not out:
        raise RuntimeError("OpenAI summary response was empty.")
    return out[:max_chars]


def _postprocess_caption(text: str, prompt: str):
    provider = (getattr(settings, "CAPTION_SUMMARY_PROVIDER", "none") or "none").strip().lower()
    clean = (text or "").strip()
    if not clean:
        return clean
    if provider != "openai":
        return clean
    try:
        return _summarize_caption_openai(clean, prompt)
    except Exception as e:
        print(f"[caption-summary] openai summary failed, using raw caption: {e}", flush=True)
        return clean


def _latest_obs_from_camera():
    """
    Use the newest camera caption only.
    Returns (obs_dict or None, last_caption_text or "").
    """
    items = camera_captions.since(0)  # last up to 100
    if not items:
        return None, ""
    txt = items[-1].get("text") or ""
    obs = _extract_json_anywhere(txt)
    if isinstance(obs, dict):
        return obs, txt
    return None, txt


def _latest_obs_from_video(path: str):
    """
    Use the newest caption for a given video path only.
    Returns (obs_dict or None, last_caption_text or "").
    """
    buf = get_video_buffer(_resolve_video_path(path))
    items = buf.since(0)
    if not items:
        return None, ""
    txt = items[-1].get("text") or ""
    obs = _extract_json_anywhere(txt)
    if isinstance(obs, dict):
        return obs, txt
    return None, txt


# -------------------------------------------------
# Views (UI + streams)
# -------------------------------------------------
def _resolve_video_path(path: str) -> str:
    if not path:
        return path
    path = path.strip()
    if os.path.isabs(path):
        return path
    candidates = [
        str(Path(settings.BASE_DIR) / path),
        str(Path(settings.MEDIA_ROOT) / path),
    ]
    return next((p for p in candidates if os.path.exists(p)), path)
def index(request):
    return render(request, "analyzer/index.html")

def hospital_eval(request):
    return render(request, "analyzer/hospital_eval.html")


def yolo_editor(request):
    return render(request, "analyzer/yolo_editor.html")


@csrf_exempt
def yolo_editor_save(request):
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    image = request.FILES.get("image")
    labels_text    = (request.POST.get("labels_text")    or "").strip()
    classes_text   = (request.POST.get("classes_text")   or "").strip()
    classes_b_text = (request.POST.get("classes_b_text") or "").strip()
    image_name     = (request.POST.get("image_name")     or "").strip()
    meta_json = (request.POST.get("meta_json") or "").strip()

    if not image and not image_name:
        return HttpResponseBadRequest("Missing image or image_name")

    image_basename = Path(image_name or getattr(image, "name", "image")).name
    stem = _safe_token(Path(image_basename).stem)
    out_root = Path(settings.MEDIA_ROOT) / "yolo_editor" / f"{stem}__{int(time.time())}"
    out_root.mkdir(parents=True, exist_ok=True)

    image_path = None
    if image:
        suffix = Path(getattr(image, "name", "")).suffix or ".jpg"
        image_path = out_root / f"{stem}{suffix}"
        with image_path.open("wb") as f:
            for chunk in image.chunks():
                f.write(chunk)

    labels_path = out_root / f"{stem}.txt"
    labels_path.write_text(labels_text + ("\n" if labels_text else ""), encoding="utf-8")

    classes_path = None
    if classes_text:
        classes_path = out_root / "classes_a.txt"
        classes_path.write_text(classes_text + "\n", encoding="utf-8")
    if classes_b_text:
        (out_root / "classes_b.txt").write_text(classes_b_text + "\n", encoding="utf-8")

    meta_path = None
    if meta_json:
        meta_dir = out_root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta_path = meta_dir / f"{stem}.json"
        try:
            parsed = json.loads(meta_json)
            meta_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
        except json.JSONDecodeError:
            meta_path.write_text(meta_json, encoding="utf-8")

    return JsonResponse(
        {
            "ok": True,
            "saved_dir": str(out_root),
            "image_path": str(image_path) if image_path else "",
            "labels_path": str(labels_path),
            "classes_path": str(classes_path) if classes_path else "",
            "meta_path": str(meta_path) if meta_path else "",
        }
    )


# -------------------------------------------------
# OpenTTS proxy endpoints
# -------------------------------------------------
def _opentts_base():
    return (getattr(settings, "OPENTTS_URL", "") or "").rstrip("/")


def opentts_voices(request):
    base = _opentts_base()
    if not base:
        return JsonResponse({"error": "OPENTTS_URL is not configured."}, status=500)

    params = {}
    for key in ("tts_name", "language", "locale", "gender"):
        val = request.GET.get(key)
        if val:
            params[key] = val

    url = f"{base}/api/voices"
    if params:
        url = f"{url}?{urlencode(params)}"

    try:
        with urlopen(url, timeout=8) as resp:
            data = resp.read()
            ct = resp.headers.get("Content-Type", "application/json")
            return HttpResponse(data, content_type=ct)
    except (URLError, HTTPError) as e:
        return JsonResponse({"error": f"OpenTTS voices failed: {e}"}, status=502)


def opentts_tts(request):
    base = _opentts_base()
    if not base:
        return JsonResponse({"error": "OPENTTS_URL is not configured."}, status=500)

    text = (request.GET.get("text") or "").strip()
    if not text:
        return HttpResponseBadRequest("Missing ?text=...")

    params = {"text": text}
    voice = (request.GET.get("voice") or "").strip()
    if voice:
        params["voice"] = voice

    # Optional OpenTTS parameters (pass-through)
    for key in ("lang", "vocoder", "speakerId", "ssml", "ssmlNumbers", "ssmlDates", "ssmlCurrency", "cache"):
        val = request.GET.get(key)
        if val is not None and val != "":
            params[key] = val

    url = f"{base}/api/tts?{urlencode(params)}"
    try:
        with urlopen(url, timeout=20) as resp:
            data = resp.read()
            ct = resp.headers.get("Content-Type", "audio/wav")
            return HttpResponse(data, content_type=ct)
    except (URLError, HTTPError) as e:
        return JsonResponse({"error": f"OpenTTS tts failed: {e}"}, status=502)


# -------------------------------------------------
# OpenTTS server-side speaking (C-3PO style)
# -------------------------------------------------
_TTS_LOCK = threading.Lock()
_TTS_BUSY = False
_TTS_LAST_TS = 0.0

def _clean_tts_text(text: str) -> str:
    if not text:
        return ""
    return " ".join(text.split()).strip()[:400]

def speak_c3po(text: str):
    if not getattr(settings, "OPENTTS_SPEAK_ENABLED", False):
        return
    if not _caption_has_hazard(text):
        return
    clean = _clean_tts_text(text)
    if not clean:
        return
    global _TTS_LAST_TS, _TTS_BUSY
    min_interval = float(getattr(settings, "OPENTTS_MIN_INTERVAL", 1.5))
    now = time.time()
    if now - _TTS_LAST_TS < min_interval:
        return
    if _TTS_BUSY:
        return

    tts_url = getattr(settings, "OPENTTS_TTS_URL", "")
    if not tts_url:
        return

    def _log(msg: str):
        if getattr(settings, "OPENTTS_DEBUG", False):
            print(f"[opentts] {msg}", flush=True)

    def _run():
        global _TTS_BUSY, _TTS_LAST_TS
        with _TTS_LOCK:
            _TTS_BUSY = True
            try:
                params = {
                    "voice": getattr(settings, "OPENTTS_VOICE", "larynx:harvard"),
                    "lang": getattr(settings, "OPENTTS_LANG", "en"),
                    "format": getattr(settings, "OPENTTS_FORMAT", "wav"),
                    "rate": getattr(settings, "OPENTTS_RATE", 1.0),
                    "effect": getattr(settings, "OPENTTS_EFFECT", "robot"),
                }
                url = f"{tts_url}?{urlencode(params)}"
                payload = clean.encode("utf-8")
                req = Request(url, data=payload, method="POST", headers={"Content-Type": "text/plain; charset=utf-8"})
                with urlopen(req, timeout=15) as resp:
                    data = resp.read()
                if not data:
                    _log("OpenTTS returned empty audio.")
                    return

                keep = getattr(settings, "OPENTTS_SAVE_WAV", False)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
                    f.write(data)
                    wav_path = f.name
                _log(f"WAV saved: {wav_path} ({len(data)} bytes)")

                try:
                    try:
                        subprocess.run(
                            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", wav_path],
                            check=True,
                        )
                    except Exception as e:
                        _log(f"ffplay failed: {e}; trying aplay")
                        subprocess.run(["aplay", "-q", wav_path], check=True)
                finally:
                    if not keep:
                        try:
                            os.remove(wav_path)
                        except Exception:
                            pass
            except Exception as e:
                _log(f"OpenTTS speak error: {e}")
            finally:
                _TTS_LAST_TS = time.time()
                _TTS_BUSY = False

    threading.Thread(target=_run, daemon=True).start()


def stream_camera(request):
    model_key = request.GET.get("model", "fastvlm")
    prompt = _get_str(request, "prompt", "Give a short caption.")
    analyze_every = _get_int(request, "analyze_every", 30, 1, 600)
    every_n = _get_int(request, "every_n", 2, 1, 10)
    max_width = _get_int(request, "max_width", 1920, 320, 3840)
    max_new   = _get_int(request, "max_new_tokens", 96, 8, 256)
    multi_frames = _get_int(request, "multi_frames", 1, 1, 8)

    # Optional: force strict JSON for camera if requested
    enforce_json = request.GET.get("enforce_json", "0").lower() in ("1", "true", "yes", "on")
    if request.GET.get("pipeline", "").strip().lower() == "two_stage":
        if prompt and prompt != "Give a short caption.":
            prompt = f"{prompt}\n\n{SCENE_STATE_JSON_PROMPT}"
        else:
            prompt = SCENE_STATE_JSON_PROMPT
        enforce_json = False
    if enforce_json:
        if prompt and prompt != "Give a short caption.":
            prompt = f"{prompt}\n\n{FISHEYE_JSON_PROMPT}"
        else:
            prompt = FISHEYE_JSON_PROMPT

    prompt = _maybe_multi_prompt(prompt)

    yolo_model_path   = getattr(settings, "YOLO_MODEL_PATH",
        "/home/vision/work/aura/models/hospital_hds/weights/best.pt")
    yolo_model_path_b = getattr(settings, "YOLO_MODEL_PATH_B",
        "/home/vision/work/aura/models/hospital_yolo/weights/best.pt")
    yolo_conf   = float(getattr(settings, "YOLO_CONF", 0.35))
    yolo_conf_b = float(getattr(settings, "YOLO_CONF_B", 0.40))

    stream = ThreadedAnalyzerStream(
        source=0,
        describer=get_describer(model_key),
        every_n=every_n,
        analyze_every=analyze_every,
        overlay=True,
        max_width=max_width,
        on_caption=lambda txt, prm, thumbs, meta: (
            camera_captions.add(
                txt,
                source=f"camera:{model_key}",
                prompt=prm,
                thumbs=thumbs,
                frame_id=meta.get("frame_id"),
                frame_ts=meta.get("frame_ts"),
                frame_ids=meta.get("frame_ids"),
                frame_tss=meta.get("frame_tss"),
            ),
            _persist_caption("camera", "camera", model_key, txt, prm, thumbs, meta=meta),
            speak_c3po(txt),
        ),
        caption_postprocess=_postprocess_caption,
        prompt=prompt,
        max_new_tokens=max_new,
        multi_frames=multi_frames,
        include_thumbs=True,
        yolo_model_path=yolo_model_path,
        yolo_model_path_b=yolo_model_path_b,
        yolo_conf=yolo_conf,
        yolo_conf_b=yolo_conf_b,
    )
    return StreamingHttpResponse(
        mjpeg_generator(stream, fps_limit=20),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


def stream_video(request):
    path = request.GET.get("path")
    if not path:
        return HttpResponseBadRequest("Missing ?path=/abs/path/to/video.mp4")
    path = _resolve_video_path(path)
    if not os.path.exists(path):
        return HttpResponseBadRequest(f"Video not found: {path}")

    model_key    = request.GET.get("model", "fastvlm")
    prompt       = _get_str(request, "prompt", "Give a short caption.")
    analyze_every = _get_int(request, "analyze_every", 30, 1, 600)
    every_n      = _get_int(request, "every_n", 2, 1, 10)
    max_width    = _get_int(request, "max_width", 1920, 320, 3840)
    max_new      = _get_int(request, "max_new_tokens", 96, 8, 256)
    fps_limit    = _get_int(request, "fps", 5, 1, 30)
    multi_frames = _get_int(request, "multi_frames", 1, 1, 8)
    yolo_enabled = _parse_bool(request.GET.get("yolo", "1"), default=True)
    yolo_conf    = float(request.GET.get("yolo_conf", "0.35"))

    # Force strict JSON for video if requested
    enforce_json = request.GET.get("enforce_json", "0").lower() in ("1", "true", "yes", "on")
    if request.GET.get("pipeline", "").strip().lower() == "two_stage":
        if prompt and prompt != "Give a short caption.":
            prompt = f"{prompt}\n\n{SCENE_STATE_JSON_PROMPT}"
        else:
            prompt = SCENE_STATE_JSON_PROMPT
        enforce_json = False
    if enforce_json:
        if prompt and prompt != "Give a short caption.":
            prompt = f"{prompt}\n\n{FISHEYE_JSON_PROMPT}"
        else:
            prompt = FISHEYE_JSON_PROMPT

    buf = get_video_buffer(path)
    prompt = _maybe_multi_prompt(prompt)
    eval_session = _HospitalEvalSession(
        source=path,
        model_key=model_key,
        prompt=prompt if isinstance(prompt, str) else "\n---\n".join(prompt),
        yolo_enabled=yolo_enabled,
    )

    yolo_model_path   = getattr(settings, "YOLO_MODEL_PATH",
        "/home/vision/work/aura/models/hospital_hds/weights/best.pt")
    yolo_model_path_b = getattr(settings, "YOLO_MODEL_PATH_B",
        "/home/vision/work/aura/models/hospital_yolo/weights/best.pt")

    def _on_yolo(payload):
        """Called on every frame with YOLO detections and frame metadata."""
        if not payload:
            return
        obs = payload.get("obs") or {}
        text = json.dumps(obs, ensure_ascii=False)
        frame_id = payload.get("frame_id")
        frame_ts = payload.get("frame_ts")
        buf.add(
            text,
            source=f"{path}:yolo",
            prompt="yolo",
            thumbs=[],
            frame_id=frame_id,
            frame_ts=frame_ts,
            frame_ids=[frame_id] if frame_id is not None else [],
            frame_tss=[frame_ts] if frame_ts is not None else [],
        )
        eval_session.log_yolo_frame(payload)

    stream = ThreadedAnalyzerStream(
        source=path,
        describer=get_describer(model_key),
        every_n=every_n,
        analyze_every=analyze_every,
        overlay=True,
        max_width=max_width,
        on_caption=lambda txt, prm, thumbs, meta: (
            buf.add(
                txt,
                source=f"{path}:{model_key}",
                prompt=prm,
                thumbs=thumbs,
                frame_id=meta.get("frame_id"),
                frame_ts=meta.get("frame_ts"),
                frame_ids=meta.get("frame_ids"),
                frame_tss=meta.get("frame_tss"),
            ),
            _persist_caption("video", path, model_key, txt, prm, thumbs, meta=meta),
            eval_session.log_caption(txt, prm, thumbs, meta=meta),
            speak_c3po(txt),
        ),
        caption_postprocess=_postprocess_caption,
        prompt=prompt,
        max_new_tokens=max_new,
        multi_frames=multi_frames,
        include_thumbs=True,
        yolo_enabled=yolo_enabled,
        yolo_model_path=yolo_model_path,
        yolo_model_path_b=yolo_model_path_b,
        yolo_conf=yolo_conf,
        yolo_conf_b=float(getattr(settings, "YOLO_CONF_B", 0.40)),
        on_yolo=_on_yolo,
        on_stop=eval_session.finish,
    )
    return StreamingHttpResponse(
        mjpeg_generator(stream, fps_limit=fps_limit),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


def captions_camera(request):
    after_raw = request.GET.get("after", "0")
    try:
        after = int(after_raw)
    except ValueError:
        after = 0
    return JsonResponse({"items": camera_captions.since(after)})


def captions_video(request):
    path = request.GET.get("path")
    if not path:
        return HttpResponseBadRequest("Missing ?path=...")
    path = _resolve_video_path(path)
    after_raw = request.GET.get("after", "0")
    try:
        after = int(after_raw)
    except ValueError:
        after = 0
    buf = get_video_buffer(path)
    return JsonResponse({"items": buf.since(after)})


@csrf_exempt
def clear_captions_camera(request):
    camera_captions.clear()
    return JsonResponse({"ok": True})


@csrf_exempt
def clear_captions_video(request):
    path = request.GET.get("path")
    if not path:
        return HttpResponseBadRequest("Missing ?path=...")
    buf = get_video_buffer(_resolve_video_path(path))
    buf.clear()
    return JsonResponse({"ok": True})


# -------------------------------------------------
# Decision endpoints
# -------------------------------------------------
@csrf_exempt
def decide(request):
    """
    POST JSON:
    {
      "obs": { ... VLM JSON ... }  // or string containing JSON
      "instruction": "high-level goal" // optional
    }
    """
    if request.method != "POST":
        return HttpResponseBadRequest("Use POST with JSON body.")

    try:
        payload = json.loads(request.body or "{}")
    except Exception:
        return HttpResponseBadRequest("Invalid JSON body.")

    obs = payload.get("obs")
    instruction = (payload.get("instruction") or "").strip()
    provider_override = payload.get("provider")
    include_policy = _parse_bool(
        payload.get("include_policy"),
        default=bool(getattr(settings, "ACTION_OPENAI_GENERATE_POLICY", False)),
    )
    if obs is None:
        return HttpResponseBadRequest("Missing 'obs' in JSON body.")

    # Accept dict, plain text, or text blob containing JSON.
    if isinstance(obs, str):
        obs_json = _extract_json_anywhere(obs)
        if isinstance(obs_json, dict):
            obs = obs_json

    if not isinstance(obs, (dict, str)):
        return JsonResponse({
            "action": {"direction": "hold", "reason": "No valid observation JSON provided."},
            "note": "Send obs as dict or text."
        })

    scene_state = coerce_obs(obs)
    action, policy, provider, note = _decide_action(
        scene_state, instruction, include_policy=include_policy, provider_override=provider_override
    )
    resp = {"action": action, "instruction": instruction, "provider": provider, "scene_state": scene_state}
    if policy:
        resp["policy"] = policy
    if note:
        resp["note"] = note
    return JsonResponse(resp)


def decide_camera(request):
    """
    GET: optionally ?instruction=...
    Tries JSON from the last camera captions; if none, falls back to free-text.
    """
    instruction = (request.GET.get("instruction") or "").strip()
    include_policy = _parse_bool(
        request.GET.get("include_policy"),
        default=bool(getattr(settings, "ACTION_OPENAI_GENERATE_POLICY", False)),
    )
    provider_override = request.GET.get("provider")
    obs, last_txt = _latest_obs_from_camera()

    # If we found a JSON object, use it
    if isinstance(obs, dict):
        scene_state = coerce_obs(obs)
        action, policy, provider, note = _decide_action(
            scene_state, instruction, include_policy=include_policy, provider_override=provider_override
        )
        return JsonResponse({
            "source": "json",
            "obs": obs,
            "scene_state": scene_state,
            "action": action,
            **({"policy": policy} if policy else {}),
            "provider": provider,
            **({"note": note} if note else {}),
        })

    # No JSON? Fall back to free text (policy.py can parse text)
    if last_txt:
        scene_state = coerce_obs(last_txt)
        action, policy, provider, note = _decide_action(
            scene_state, instruction, include_policy=include_policy, provider_override=provider_override
        )
        return JsonResponse({
            "source": "text",
            "last_caption": last_txt,
            "scene_state": scene_state,
            "action": action,
            **({"policy": policy} if policy else {}),
            "provider": provider,
            "note": ("No strict JSON found; parsed from free text. " + note).strip()
        })

    # Nothing captured yet
    return JsonResponse({
        "action": {"direction": "hold", "reason": "No camera captions yet."},
        "note": "Start/Restart the camera stream and wait for a caption."
    })

def decide_video(request):
    """
    GET ?path=...&instruction=...
    Tries JSON for that video; if none, falls back to free-text.
    """
    path = request.GET.get("path")
    if not path:
        return HttpResponseBadRequest("Missing ?path=...")
    path = _resolve_video_path(path)

    instruction = (request.GET.get("instruction") or "").strip()
    include_policy = _parse_bool(
        request.GET.get("include_policy"),
        default=bool(getattr(settings, "ACTION_OPENAI_GENERATE_POLICY", False)),
    )
    provider_override = request.GET.get("provider")
    obs, last_txt = _latest_obs_from_video(path)

    if isinstance(obs, dict):
        scene_state = coerce_obs(obs)
        action, policy, provider, note = _decide_action(
            scene_state, instruction, include_policy=include_policy, provider_override=provider_override
        )
        return JsonResponse({
            "source": "json",
            "obs": obs,
            "scene_state": scene_state,
            "action": action,
            **({"policy": policy} if policy else {}),
            "provider": provider,
            **({"note": note} if note else {}),
        })

    if last_txt:
        scene_state = coerce_obs(last_txt)
        action, policy, provider, note = _decide_action(
            scene_state, instruction, include_policy=include_policy, provider_override=provider_override
        )
        return JsonResponse({
            "source": "text",
            "last_caption": last_txt,
            "scene_state": scene_state,
            "action": action,
            **({"policy": policy} if policy else {}),
            "provider": provider,
            "note": ("No strict JSON found; parsed from free text. " + note).strip()
        })

    return JsonResponse({
        "action": {"direction": "hold", "reason": "No video captions yet."},
        "note": "Open/Restart the video stream and wait for a caption."
    })
