import json
import math
import re
from typing import Dict, Any, List, Tuple, Union

# ---------- helpers to map bearings/positions ----------

def _sector_from_bearing(bearing_deg: float) -> str:
    """Map a 0..359 bearing to a coarse sector."""
    b = ((bearing_deg or 0.0) % 360.0)
    if b <= 30 or b >= 330:
        return "forward"
    if 30 < b < 150:
        return "right"
    if 210 < b < 330:
        return "left"
    return "back"


def _is_near(dist_label: str) -> bool:
    return (dist_label or "").lower().startswith("near")


def _clock_to_bearing(clock_str: str) -> float:
    """
    Convert '1..12 o'clock' into degrees:
      12 -> 0°, 3 -> 90°, 6 -> 180°, 9 -> 270°
    """
    try:
        c = int(clock_str)
    except Exception:
        return 0.0
    c = ((c - 12) % 12) or 12  # keep 1..12
    return 0.0 if c == 12 else float(c * 30)


# ---------- robust parsing from any caption format ----------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)

def _extract_json_anywhere(text: str) -> Dict[str, Any]:
    """
    Try to find a JSON object anywhere inside a text blob and parse it.
    Returns {} if none or invalid.
    """
    if not isinstance(text, str):
        return {}
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return {}
    block = m.group(0)
    try:
        obj = json.loads(block)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


# Some common object words we care about (extend as you wish)
_OBJ_WORDS = r"(person|drone|chair|table|box|toolbox|bench|cart|cable|monitor|screen|pc|laptop|robot|door|tripod|shelf|rack|stool)"

def _from_free_text(text: str) -> Dict[str, Any]:
    """
    Build a best-effort observation dict from unstructured text.

    Heuristics supported (any order in the sentence):
      - bearing <deg> / <deg> deg / <deg>°
      - <clock> o'clock / oclock
      - position left|center|right
      - distance near|mid|far (or hints like 'within 2 m' -> near)
    """
    if not isinstance(text, str):
        return {"summary": "", "obstacles": [], "risks": []}

    obstacles: List[Dict[str, Any]] = []

    # Split into short clauses
    clauses = re.split(r"[;\n\.]+", text)
    for cl in clauses:
        cln = cl.strip()
        if not cln:
            continue

        name_m = re.search(_OBJ_WORDS, cln, flags=re.IGNORECASE)
        if not name_m:
            continue
        name = name_m.group(1).lower()

        # bearing in degrees
        bearing = None
        m_deg = re.search(r"\b(\d{1,3})\s*(?:deg|°|degrees?)\b", cln, flags=re.IGNORECASE)
        if not m_deg:
            m_bear = re.search(r"\bbearing\s*[:=]?\s*(\d{1,3})\b", cln, flags=re.IGNORECASE)
            if m_bear:
                m_deg = m_bear
        if m_deg:
            try:
                val = int(m_deg.group(1))
                if 0 <= val <= 359:
                    bearing = float(val)
            except Exception:
                pass

        # clock direction
        clock = None
        m_clock = re.search(r"\b(\d{1,2})\s*(?:o'?clock|oclock)\b", cln, flags=re.IGNORECASE)
        if m_clock and bearing is None:
            clock = m_clock.group(1)
            bearing = _clock_to_bearing(clock)

        # position coarse
        pos = None
        m_pos = re.search(r"\b(left|right|center|centre)\b", cln, flags=re.IGNORECASE)
        if m_pos:
            pos = m_pos.group(1).lower()
            if pos == "centre":
                pos = "center"

        # distance label
        dist = None
        m_dist = re.search(r"\b(near|mid|far)\b", cln, flags=re.IGNORECASE)
        if m_dist:
            dist = m_dist.group(1).lower()
        else:
            # simple numeric hint => near if mentions <= 2 m
            m_m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", cln, flags=re.IGNORECASE)
            if m_m:
                try:
                    meters = float(m_m.group(1))
                    if meters <= 2.0:
                        dist = "near"
                    elif meters <= 4.0:
                        dist = "mid"
                    else:
                        dist = "far"
                except Exception:
                    pass

        obstacles.append({
            "name": name,
            **({"bearing_deg": float(bearing)} if bearing is not None else {}),
            **({"clock": str(clock)} if clock is not None else {}),
            **({"position": pos} if pos else {}),
            "distance": dist or "unknown",
            "confidence": 0.0
        })

    summary = text.strip()
    if len(summary) > 160:
        summary = summary[:157] + "..."
    return {"summary": summary, "obstacles": obstacles, "risks": []}


def coerce_obs(obs_like: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    """
    Accept dict or raw text. Try JSON inside text; otherwise parse free-text.
    Always return a dict with at least: {"summary": str, "obstacles": [], "risks": []}
    """
    # Already a dict with obstacles/risk
    if isinstance(obs_like, dict):
        summary = obs_like.get("summary") or ""
        obstacles = obs_like.get("obstacles") or []
        risks = obs_like.get("risks") or []
        return {"summary": summary, "obstacles": obstacles, "risks": risks}

    # Text path
    text = str(obs_like or "")
    as_json = _extract_json_anywhere(text)
    if as_json:
        return coerce_obs(as_json)  # normalize shape recursively
    return _from_free_text(text)


# ---------- the simple rule-based policy ----------

class SimpleHeuristicPolicy:
    """
    Rule-based policy over an 'observation' dict with keys:
      summary, obstacles:[{name, bearing_deg?|position?, distance?}], risks:[]
    Returns: {"direction": "forward|left|right|back|hold", "reason": "..."}
    """

    @staticmethod
    def decide(obs_like: Union[Dict[str, Any], str], instruction: str = "") -> Dict[str, str]:
        obs = coerce_obs(obs_like)
        obstacles: List[Dict[str, Any]] = obs.get("obstacles") or []

        counts = {"forward": 0, "left": 0, "right": 0, "back": 0}
        has_person_ahead = False
        has_drone_ahead = False

        for o in obstacles:
            name = (o.get("name") or "").lower()
            dist = (o.get("distance") or "").lower()

            if "bearing_deg" in o and isinstance(o["bearing_deg"], (int, float)):
                sector = _sector_from_bearing(float(o["bearing_deg"]))
            else:
                pos = (o.get("position") or "").lower()
                if pos == "left":
                    sector = "left"
                elif pos == "right":
                    sector = "right"
                elif pos == "center":
                    sector = "forward"
                else:
                    sector = "forward"

            if _is_near(dist):
                counts[sector] += 1
                if sector == "forward" and "person" in name:
                    has_person_ahead = True
                if sector == "forward" and "drone" in name:
                    has_drone_ahead = True

        if has_drone_ahead or has_person_ahead:
            who = "drone" if has_drone_ahead else "person"
            return {"direction": "hold", "reason": f"Near {who} ahead; waiting to avoid collision."}

        order = ["forward", "left", "right", "back"]
        best = min(order, key=lambda s: (counts[s], order.index(s)))

        if counts[best] == 0 and best == "forward":
            return {"direction": "forward", "reason": "Clear path ahead."}
        elif counts[best] == 0:
            return {"direction": best, "reason": f"No near obstacles to the {best}."}
        else:
            return {"direction": best, "reason": f"Fewer near obstacles to the {best} (counts={counts})."}


def decide_core(obs_like: Union[Dict[str, Any], str], instruction: str = "") -> Dict[str, str]:
    """
    Entry point used by views: now accepts dict OR raw text.
    """
    return SimpleHeuristicPolicy.decide(obs_like, instruction)
