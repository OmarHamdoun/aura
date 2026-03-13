import json
import os
import threading
import time
import tempfile
import subprocess
import base64
import binascii
from urllib.parse import urlencode
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from django.http import HttpResponse, HttpResponseBadRequest, StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.conf import settings
from pathlib import Path
from django.views.decorators.csrf import csrf_exempt

from .inference import FastVLMDescriber, QwenVLDescriber, MiniInternVL2DriveLMDescriber
from .streaming import ThreadedAnalyzerStream, mjpeg_generator
from .captions import camera_captions, get_video_buffer, append_caption_jsonl
from .policy import decide_core  # heuristic policy

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


def _decide_action(obs_like, instruction: str, include_policy: bool = False):
    provider = (getattr(settings, "ACTION_POLICY_PROVIDER", "heuristic") or "heuristic").strip().lower()
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
    if enforce_json:
        prompt = FISHEYE_JSON_PROMPT

    prompt = _maybe_multi_prompt(prompt)
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

    model_key = request.GET.get("model", "fastvlm")
    prompt = _get_str(request, "prompt", "Give a short caption.")
    analyze_every = _get_int(request, "analyze_every", 30, 1, 600)
    every_n = _get_int(request, "every_n", 2, 1, 10)
    max_width = _get_int(request, "max_width", 1920, 320, 3840)
    max_new   = _get_int(request, "max_new_tokens", 96, 8, 256)
    fps_limit = _get_int(request, "fps", 5, 1, 30)
    multi_frames = _get_int(request, "multi_frames", 1, 1, 8)

    # Force strict JSON for video if requested
    enforce_json = request.GET.get("enforce_json", "0").lower() in ("1", "true", "yes", "on")
    if enforce_json:
        prompt = FISHEYE_JSON_PROMPT

    buf = get_video_buffer(path)
    prompt = _maybe_multi_prompt(prompt)
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
            speak_c3po(txt),
        ),
        caption_postprocess=_postprocess_caption,
        prompt=prompt,
        max_new_tokens=max_new,
        multi_frames=multi_frames,
        include_thumbs=True,
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

    action, policy, provider, note = _decide_action(obs, instruction, include_policy=include_policy)
    resp = {"action": action, "instruction": instruction, "provider": provider}
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
    obs, last_txt = _latest_obs_from_camera()

    # If we found a JSON object, use it
    if isinstance(obs, dict):
        action, policy, provider, note = _decide_action(obs, instruction, include_policy=include_policy)
        return JsonResponse({
            "source": "json",
            "obs": obs,
            "action": action,
            **({"policy": policy} if policy else {}),
            "provider": provider,
            **({"note": note} if note else {}),
        })

    # No JSON? Fall back to free text (policy.py can parse text)
    if last_txt:
        action, policy, provider, note = _decide_action(last_txt, instruction, include_policy=include_policy)
        return JsonResponse({
            "source": "text",
            "last_caption": last_txt,
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
    obs, last_txt = _latest_obs_from_video(path)

    if isinstance(obs, dict):
        action, policy, provider, note = _decide_action(obs, instruction, include_policy=include_policy)
        return JsonResponse({
            "source": "json",
            "obs": obs,
            "action": action,
            **({"policy": policy} if policy else {}),
            "provider": provider,
            **({"note": note} if note else {}),
        })

    if last_txt:
        action, policy, provider, note = _decide_action(last_txt, instruction, include_policy=include_policy)
        return JsonResponse({
            "source": "text",
            "last_caption": last_txt,
            "action": action,
            **({"policy": policy} if policy else {}),
            "provider": provider,
            "note": ("No strict JSON found; parsed from free text. " + note).strip()
        })

    return JsonResponse({
        "action": {"direction": "hold", "reason": "No video captions yet."},
        "note": "Open/Restart the video stream and wait for a caption."
    })
