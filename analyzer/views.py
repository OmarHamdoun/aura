import json
import threading
from django.http import HttpResponseBadRequest, StreamingHttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .inference import FastVLMDescriber, QwenVLDescriber
from .streaming import ThreadedAnalyzerStream, mjpeg_generator
from .captions import camera_captions, get_video_buffer
from .policy import decide_core  # heuristic policy


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

def get_describer(model_key: str):
    key = (model_key or "fastvlm").lower()
    if key not in ("fastvlm", "qwen"):
        key = "fastvlm"
    with _DLOCK:
        if key not in _DESCRIBERS:
            if key == "qwen":
                _DESCRIBERS[key] = QwenVLDescriber()
            else:
                _DESCRIBERS[key] = FastVLMDescriber()
        return _DESCRIBERS[key]


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


def _latest_obs_from_camera():
    """
    Search the last ~100 camera captions (newest first) for a parseable JSON object.
    Returns (obs_dict or None, last_caption_text or "").
    """
    items = camera_captions.since(0)  # last up to 100
    if not items:
        return None, ""
    for it in reversed(items):
        txt = it.get("text") or ""
        obs = _extract_json_anywhere(txt)
        if isinstance(obs, dict):
            return obs, txt
    # none parseable; return the very last text for debugging
    return None, items[-1].get("text") or ""


def _latest_obs_from_video(path: str):
    """
    Search the last ~100 captions for a given video path.
    Returns (obs_dict or None, last_caption_text or "").
    """
    buf = get_video_buffer(path)
    items = buf.since(0)
    if not items:
        return None, ""
    for it in reversed(items):
        txt = it.get("text") or ""
        obs = _extract_json_anywhere(txt)
        if isinstance(obs, dict):
            return obs, txt
    return None, items[-1].get("text") or ""


# -------------------------------------------------
# Views (UI + streams)
# -------------------------------------------------
def index(request):
    return render(request, "analyzer/index.html")


def stream_camera(request):
    model_key = request.GET.get("model", "fastvlm")
    prompt = _get_str(request, "prompt", "Give a short caption.")
    analyze_every = _get_int(request, "analyze_every", 30, 1, 600)
    every_n = _get_int(request, "every_n", 2, 1, 10)
    max_width = _get_int(request, "max_width", 1920, 320, 3840)
    max_new   = _get_int(request, "max_new_tokens", 96, 8, 256)

    # Optional: force strict JSON for camera if requested
    enforce_json = request.GET.get("enforce_json", "0").lower() in ("1", "true", "yes", "on")
    if enforce_json:
        prompt = FISHEYE_JSON_PROMPT

    stream = ThreadedAnalyzerStream(
        source=0,
        describer=get_describer(model_key),
        every_n=every_n,
        analyze_every=analyze_every,
        overlay=True,
        max_width=max_width,
        on_caption=lambda txt: camera_captions.add(txt, source=f"camera:{model_key}"),
        prompt=prompt,
        max_new_tokens=max_new,
    )
    return StreamingHttpResponse(
        mjpeg_generator(stream, fps_limit=20),
        content_type="multipart/x-mixed-replace; boundary=frame",
    )


def stream_video(request):
    path = request.GET.get("path")
    if not path:
        return HttpResponseBadRequest("Missing ?path=/abs/path/to/video.mp4")

    model_key = request.GET.get("model", "fastvlm")
    prompt = _get_str(request, "prompt", "Give a short caption.")
    analyze_every = _get_int(request, "analyze_every", 30, 1, 600)
    every_n = _get_int(request, "every_n", 2, 1, 10)
    max_width = _get_int(request, "max_width", 1920, 320, 3840)
    max_new   = _get_int(request, "max_new_tokens", 96, 8, 256)

    # Force strict JSON for video if requested
    enforce_json = request.GET.get("enforce_json", "0").lower() in ("1", "true", "yes", "on")
    if enforce_json:
        prompt = FISHEYE_JSON_PROMPT

    buf = get_video_buffer(path)
    stream = ThreadedAnalyzerStream(
        source=path,
        describer=get_describer(model_key),
        every_n=every_n,
        analyze_every=analyze_every,
        overlay=True,
        max_width=max_width,
        on_caption=lambda txt: buf.add(txt, source=f"{path}:{model_key}"),
        prompt=prompt,
        max_new_tokens=max_new,
    )
    return StreamingHttpResponse(
        mjpeg_generator(stream, fps_limit=20),
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
    buf = get_video_buffer(path)
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

    # Accept either dict or a text blob that contains JSON
    if isinstance(obs, str):
        obs = _extract_json_anywhere(obs)

    if not isinstance(obs, dict):
        return JsonResponse({
            "action": {"direction": "hold", "reason": "No valid observation JSON provided."},
            "note": "Send the VLM's structured JSON in 'obs' or ensure your prompt outputs strict JSON."
        })

    action = decide_core(obs, instruction)
    return JsonResponse({"action": action, "instruction": instruction})


def decide_camera(request):
    """
    GET: optionally ?instruction=...
    Tries JSON from the last camera captions; if none, falls back to free-text.
    """
    instruction = (request.GET.get("instruction") or "").strip()
    obs, last_txt = _latest_obs_from_camera()

    # If we found a JSON object, use it
    if isinstance(obs, dict):
        return JsonResponse({
            "source": "json",
            "obs": obs,
            "action": decide_core(obs, instruction)
        })

    # No JSON? Fall back to free text (policy.py can parse text)
    if last_txt:
        action = decide_core(last_txt, instruction)
        return JsonResponse({
            "source": "text",
            "last_caption": last_txt,
            "action": action,
            "note": "No strict JSON found; parsed from free text."
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

    instruction = (request.GET.get("instruction") or "").strip()
    obs, last_txt = _latest_obs_from_video(path)

    if isinstance(obs, dict):
        return JsonResponse({
            "source": "json",
            "obs": obs,
            "action": decide_core(obs, instruction)
        })

    if last_txt:
        action = decide_core(last_txt, instruction)
        return JsonResponse({
            "source": "text",
            "last_caption": last_txt,
            "action": action,
            "note": "No strict JSON found; parsed from free text."
        })

    return JsonResponse({
        "action": {"direction": "hold", "reason": "No video captions yet."},
        "note": "Open/Restart the video stream and wait for a caption."
    })
