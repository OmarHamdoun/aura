"""
yolo_detector.py — GPAS dual-model YOLO detector
Runs two models in parallel on every frame:
  - Model A (HDS):          58 classes — corridor navigation: staff, patient,
                            wheelchair, iv_pole, utility_cart, ventilator ...
  - Model B (hospital_yolo): 13 classes — OR/procedure equipment: anesthesia
                            machine, C-arm, patient table, saline stand ...

Both run on CPU to keep GPU free for the VLM.
Results are merged, de-duplicated by IoU, and fed into policy.py.
"""

import threading
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np

# ── Priority classification (across both models) ─────────────────────

# HOLD: robot must stop immediately when any of these are near+ahead
HOLD_CLASSES = {
    # HDS people
    "patient", "person", "staff", "visitor", "worker", "human",
    "wheel_chair",
    # OR model
    "Patient table",  # patient likely on it
}

# CAUTION: route around, slow down
CAUTION_CLASSES = {
    # HDS equipment
    "iv_pole", "utility_cart", "ventilator", "infusion_pump",
    "syringe_pump", "hospital_bed", "operating_bed", "xray_bed",
    "exam_table", "overbed_table", "bedside_table",
    "panda_baby_warmer", "incubator", "sequential_compression",
    "electrosurgical_unit", "breathing_tube",
    # OR model equipment
    "Anesthesia machine", "C-Arm", "Medicine Trolley",
    "Theater suction trolley", "Trolley", "Saline stand",
    "Machine", "stand",
}

# ── Colour palette: BGR ──────────────────────────────────────────────
# HDS model — warm colours
_HDS_COLORS = {
    "patient":      (0,   60, 255),   # red
    "person":       (0,  100, 255),
    "staff":        (0,  200, 100),   # green
    "visitor":      (0,  180, 200),   # teal
    "wheel_chair":  (0,   60, 240),   # bright red
    "iv_pole":      (0,  160, 220),   # amber
    "utility_cart": (0,  120, 200),
    "ventilator":   (0,  100, 200),
    "hospital_bed": (0,  140, 180),
}
# OR model — cool colours so boxes are visually distinct
_OR_COLORS = {
    "Anesthesia machine":      (200,  80,   0),   # blue
    "C-Arm":                   (220, 120,   0),   # blue-violet
    "Patient table":           (180,  40,   0),   # deep blue = hold
    "Saline stand":            (200, 160,  40),   # sky blue
    "Medicine Trolley":        (180, 140,  20),
    "Theater suction trolley": (160, 120,  20),
    "Trolley":                 (160, 100,  20),
    "Machine":                 (180, 100,  20),
    "Bin":                     (140, 140, 140),   # gray
    "Chair":                   (160, 160, 100),
    "Door":                    (120, 120, 120),
    "Foot stool":              (140, 120,  80),
    "stand":                   (160, 140,  60),
}
_DEFAULT_COLOR = (160, 160, 160)


def _box_color(cls_name: str) -> Tuple[int, int, int]:
    return _HDS_COLORS.get(cls_name,
           _OR_COLORS.get(cls_name, _DEFAULT_COLOR))


def _distance_from_box(box_h: float, frame_h: float, cls_name: str) -> str:
    ratio = box_h / max(frame_h, 1)
    large = {"hospital_bed", "operating_bed", "xray_bed",
             "Patient table", "Anesthesia machine", "C-Arm"}
    if cls_name in large:
        if ratio > 0.45: return "near"
        if ratio > 0.18: return "mid"
        return "far"
    if ratio > 0.35: return "near"
    if ratio > 0.15: return "mid"
    return "far"


def _position_from_box(cx: float, frame_w: float) -> str:
    r = cx / max(frame_w, 1)
    if r < 0.35: return "left"
    if r > 0.65: return "right"
    return "center"


def _iou(a: List[int], b: List[int]) -> float:
    """Intersection-over-Union for two [x1,y1,x2,y2] boxes."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / max(area_a + area_b - inter, 1)


def _deduplicate(detections: List[Dict], iou_thr: float = 0.45) -> List[Dict]:
    """
    Remove duplicate boxes from two models.
    When IoU > threshold keep the higher-confidence detection.
    """
    kept = []
    for det in sorted(detections, key=lambda d: -d["conf"]):
        duplicate = False
        for k in kept:
            if _iou(det["box"], k["box"]) > iou_thr:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept


# ── Single-model loader ──────────────────────────────────────────────

class _SingleModel:
    def __init__(self, model_path: str, conf: float, device: str,
                 max_det: int, tag: str):
        self.model_path = model_path
        self.conf = conf
        self.device = device
        self.max_det = max_det
        self.tag = tag
        self._model = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()

    def _get(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            try:
                from ultralytics import YOLO
                print(f"[yolo:{self.tag}] Loading: {self.model_path}", flush=True)
                self._model = YOLO(self.model_path)
                print(f"[yolo:{self.tag}] Ready.", flush=True)
            except Exception as e:
                print(f"[yolo:{self.tag}] Load failed: {e}", flush=True)
        return self._model

    def predict(self, frame_bgr: np.ndarray) -> List[Dict]:
        model = self._get()
        if model is None:
            return []
        try:
            with self._lock:
                results = model.predict(
                    frame_bgr,
                    conf=self.conf,
                    device=self.device,
                    max_det=self.max_det,
                    verbose=False,
                )[0]
        except Exception as e:
            print(f"[yolo:{self.tag}] predict error: {str(e)[:80]}", flush=True)
            return []

        detections = []
        h, w = frame_bgr.shape[:2]
        boxes = results.boxes
        if boxes is None or not len(boxes):
            return detections

        names = model.names
        for i in range(len(boxes)):
            cls_id   = int(boxes.cls[i].item())
            conf_val = float(boxes.conf[i].item())
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[i].tolist()]
            cls_name = names.get(cls_id, str(cls_id))
            cx = (x1 + x2) / 2
            bx_h = y2 - y1
            detections.append({
                "class":    cls_name,
                "conf":     round(conf_val, 2),
                "box":      [x1, y1, x2, y2],
                "position": _position_from_box(cx, w),
                "distance": _distance_from_box(bx_h, h, cls_name),
                "moving":   False,
                "model":    self.tag,
            })
        return detections


# ── Dual-model detector ──────────────────────────────────────────────

class HospitalYOLODetector:
    """
    Runs two YOLO models on every frame and merges results.
    Model A: HDS (58 classes — corridor)
    Model B: hospital_yolo (13 classes — OR equipment)
    Both run on CPU; results are merged and de-duplicated.
    """

    def __init__(
        self,
        # Model A — HDS corridor model
        model_path: str = "/home/vision/work/aura/models/hospital_hds/weights/best.pt",
        # Model B — OR equipment model
        model_path_b: Optional[str] = "/home/vision/work/aura/models/hospital_yolo/weights/best.pt",
        conf: float = 0.35,
        conf_b: Optional[float] = None,       # if None uses same as conf
        device: str = "cpu",
        max_det: int = 30,
    ):
        self.device = device
        self._model_a = _SingleModel(
            model_path, conf, device, max_det, tag="hds"
        )
        self._model_b = _SingleModel(
            model_path_b, conf_b or conf, device, max_det, tag="or"
        ) if model_path_b else None

    def detect(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Run both models, merge + deduplicate, draw boxes, build obs dict.
        """
        h, w = frame_bgr.shape[:2]

        # ── Run both models (sequentially on CPU) ─────────────────────
        dets_a = self._model_a.predict(frame_bgr)
        dets_b = self._model_b.predict(frame_bgr) if self._model_b else []

        # Merge and deduplicate
        all_dets = _deduplicate(dets_a + dets_b, iou_thr=0.45)

        # ── Draw boxes ────────────────────────────────────────────────
        annotated = frame_bgr.copy()
        obstacles = []
        risks     = []

        for det in all_dets:
            cls_name = det["class"]
            conf_val = det["conf"]
            x1, y1, x2, y2 = det["box"]
            position = det["position"]
            distance = det["distance"]
            model_tag = det.get("model", "")

            color = _box_color(cls_name)
            thickness = 3 if cls_name in HOLD_CLASSES else 2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label with model tag so you can see which model fired
            label = f"{cls_name} {conf_val:.2f}"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1
            )
            lx = x1
            ly = max(y1 - 4, th + 4)
            cv2.rectangle(annotated,
                          (lx, ly - th - 4), (lx + tw + 6, ly + 2),
                          color, -1)
            cv2.putText(annotated, label, (lx + 3, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (255, 255, 255), 1, cv2.LINE_AA)

            # ── policy obstacle ───────────────────────────────────────
            obstacles.append({
                "name":       cls_name,
                "position":   position,
                "distance":   distance,
                "moving":     False,
                "confidence": conf_val,
                "model":      model_tag,
            })

            # ── risk flags ────────────────────────────────────────────
            if cls_name in HOLD_CLASSES and distance == "near":
                risks.append(f"near {cls_name} {position}")
            if cls_name in ("breathing_tube", "Saline stand") and distance in ("near", "mid"):
                risks.append(f"{cls_name} visible — patient may be attached")
            if cls_name == "Anesthesia machine" and distance == "near":
                risks.append("anesthesia machine nearby — active procedure possible")

        obs = {
            "summary":      _make_summary(all_dets),
            "obstacles":    obstacles,
            "risks":        risks,
            "people_count": sum(1 for d in all_dets if d["class"] in HOLD_CLASSES),
            "path_clear":   not any(
                d["class"] in (HOLD_CLASSES | CAUTION_CLASSES)
                and d["position"] == "center"
                and d["distance"] == "near"
                for d in all_dets
            ),
            "source": "yolo_dual",
            "model_a_count": len(dets_a),
            "model_b_count": len(dets_b),
        }

        return {
            "detections": all_dets,
            "obs":        obs,
            "annotated":  annotated,
        }


# ── helpers ──────────────────────────────────────────────────────────

def _empty_obs() -> Dict[str, Any]:
    return {
        "summary": "", "obstacles": [], "risks": [],
        "people_count": 0, "path_clear": True, "source": "yolo_dual",
    }


def _make_summary(detections: List[Dict]) -> str:
    if not detections:
        return "No objects detected."
    counts: Dict[str, int] = {}
    for d in detections:
        counts[d["class"]] = counts.get(d["class"], 0) + 1
    parts = [f"{v} {k}" for k, v in sorted(counts.items(), key=lambda x: -x[1])]
    return ("Detected: " + ", ".join(parts[:6]))[:120]


# ── singleton ─────────────────────────────────────────────────────────
_DETECTOR: Optional[HospitalYOLODetector] = None
_DETECTOR_LOCK = threading.Lock()


def get_detector(
    model_path:   Optional[str] = None,
    model_path_b: Optional[str] = None,
    conf:  float = 0.35,
    conf_b: Optional[float] = None,
) -> HospitalYOLODetector:
    global _DETECTOR
    if _DETECTOR is not None:
        return _DETECTOR
    with _DETECTOR_LOCK:
        if _DETECTOR is None:
            path_a = model_path or \
                "/home/vision/work/aura/models/hospital_hds/weights/best.pt"
            path_b = model_path_b or \
                "/home/vision/work/aura/models/hospital_yolo/weights/best.pt"
            _DETECTOR = HospitalYOLODetector(
                model_path=path_a,
                model_path_b=path_b,
                conf=conf,
                conf_b=conf_b,
            )
    return _DETECTOR
