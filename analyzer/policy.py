import json
import re
from typing import Dict, Any, List, Union

# ---------- bearing / position helpers ----------

def _sector_from_bearing(bearing_deg: float) -> str:
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

def _is_near_or_mid(dist_label: str) -> bool:
    d = (dist_label or "").lower()
    return d.startswith("near") or d.startswith("mid")

def _clock_to_bearing(clock_str: str) -> float:
    try:
        c = int(clock_str)
    except Exception:
        return 0.0
    c = ((c - 12) % 12) or 12
    return 0.0 if c == 12 else float(c * 30)


# ---------- hospital entity classification ----------

# Entities that require a full HOLD when near+ahead (vulnerable / high-risk)
_HOLD_ENTITIES = {
    # people
    "person", "patient", "staff", "nurse", "doctor", "visitor",
    "worker", "child", "elderly", "people", "human",
    # mobility aids — HDS model
    "wheelchair", "wheelchair user", "person in wheelchair",
    "wheel_chair",
    "walker", "rollator", "crutch", "crutches",
    "gurney", "stretcher", "hospital bed", "bed",
    # OR model — patient likely present
    "patient table",
    # autonomous agents
    "drone", "robot",
}

# Entities that block but don't require hold
_OBSTACLE_ENTITIES = {
    # HDS model
    "iv pole", "iv stand", "infusion pole", "drip stand",
    "supply cart", "medication cart", "crash cart", "resuscitation cart",
    "equipment cart", "laundry cart", "food cart", "trolley", "cart",
    "monitor", "vital signs monitor", "ecg machine",
    "wheelchair", "wheel_chair", "chair", "stool", "bench",
    "table", "desk", "box", "toolbox", "crate",
    "door", "cable", "wire",
    "wet floor sign", "cone", "barrier",
    "shelf", "rack",
    # OR model (hospital_yolo 13-class)
    "anesthesia machine", "c-arm", "medicine trolley",
    "theater suction trolley", "saline stand",
    "machine", "stand", "bin", "foot stool",
    "utility_cart", "ventilator", "infusion_pump",
    "syringe_pump", "operating_bed", "xray_bed",
    "exam_table", "overbed_table", "bedside_table",
    "panda_baby_warmer", "incubator",
}

# Risk keywords that trigger an immediate hold
_HIGH_RISK_KEYWORDS = {
    "emergency", "fall", "fallen", "collapsed", "unconscious",
    "spill", "puddle", "wet floor", "blood", "fluid",
    "fire", "smoke", "alarm",
    "collision", "crash",
    # OR/procedure room specific
    "active procedure", "surgery", "intubated", "ventilated",
    "resuscitation", "code blue", "do not enter",
    "patient attached", "anesthesia",
}

_CAUTION_KEYWORDS = {
    "moving", "approaching", "running", "rushing", "fast",
    "narrow", "tight", "crowded", "busy",
    "cable", "wire", "cord", "trip",
    "open door", "swinging",
}


def _name_matches(name: str, entity_set: set) -> bool:
    name_l = name.lower()
    return any(e in name_l for e in entity_set)


def _urgency(obs: Dict[str, Any]) -> str:
    explicit = (obs.get("urgency") or "").lower()
    # normalise "med" (from VLM prompt shorthand) to "medium"
    if explicit == "med":
        explicit = "medium"
    if explicit in ("high", "medium", "low"):
        return explicit
    name = (obs.get("name") or "").lower()
    dist = (obs.get("distance") or "").lower()
    moving = obs.get("moving", False)
    if _name_matches(name, _HOLD_ENTITIES) and _is_near(dist):
        return "high"
    if _name_matches(name, _HOLD_ENTITIES) and _is_near_or_mid(dist) and moving:
        return "high"
    if _name_matches(name, _HOLD_ENTITIES) and _is_near_or_mid(dist):
        return "medium"
    if _name_matches(name, _OBSTACLE_ENTITIES) and _is_near(dist):
        return "medium"
    return "low"


def _is_high_urgency(obs: Dict[str, Any]) -> bool:
    """True if obstacle is high urgency — used directly in policy decide()."""
    return _urgency(obs) == "high"


# ---------- JSON extraction ----------

_JSON_BLOCK_RE = re.compile(r"\{.*\}", flags=re.DOTALL)

def _extract_json_anywhere(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {}
    m = _JSON_BLOCK_RE.search(text)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


# ---------- free-text parser ----------

_OBJ_WORDS = (
    r"(person|patient|staff|nurse|doctor|visitor|child|elderly|people|human"
    r"|wheelchair|walker|rollator|crutch(?:es)?"
    r"|gurney|stretcher|hospital\s+bed|bed"
    r"|iv\s+(?:pole|stand|drip)|infusion\s+pole|drip\s+stand"
    r"|(?:supply|medication|crash|equipment|laundry|food)\s+cart|trolley|cart"
    r"|monitor|drone|robot"
    r"|cable|wire|cord"
    r"|wet\s+floor|spill|cone|barrier"
    r"|door|chair|table|box|shelf|rack|stool|bench)"
)

def _from_free_text(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {"summary": "", "obstacles": [], "risks": []}

    obstacles: List[Dict[str, Any]] = []
    text_lower = text.lower()

    risks = []
    for kw in _HIGH_RISK_KEYWORDS:
        if kw in text_lower:
            risks.append(kw)
    for kw in _CAUTION_KEYWORDS:
        if kw in text_lower and kw not in risks:
            risks.append(kw)

    clauses = re.split(r"[;\n\.]+", text)
    for cl in clauses:
        cln = cl.strip()
        if not cln:
            continue
        name_m = re.search(_OBJ_WORDS, cln, flags=re.IGNORECASE)
        if not name_m:
            continue
        name = name_m.group(1).lower()

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

        clock = None
        m_clock = re.search(r"\b(\d{1,2})\s*(?:o'?clock|oclock)\b", cln, flags=re.IGNORECASE)
        if m_clock and bearing is None:
            clock = m_clock.group(1)
            bearing = _clock_to_bearing(clock)

        pos = None
        m_pos = re.search(r"\b(left|right|center|centre)\b", cln, flags=re.IGNORECASE)
        if m_pos:
            pos = m_pos.group(1).lower()
            if pos == "centre":
                pos = "center"

        dist = None
        m_dist = re.search(r"\b(near|mid|far)\b", cln, flags=re.IGNORECASE)
        if m_dist:
            dist = m_dist.group(1).lower()
        else:
            m_m = re.search(r"(\d+(?:\.\d+)?)\s*m\b", cln, flags=re.IGNORECASE)
            if m_m:
                try:
                    meters = float(m_m.group(1))
                    dist = "near" if meters <= 2.0 else ("mid" if meters <= 5.0 else "far")
                except Exception:
                    pass

        moving = bool(re.search(
            r"\b(moving|approaching|walking|running|rolling)\b", cln, flags=re.IGNORECASE
        ))

        obstacles.append({
            "name": name,
            **({"bearing_deg": float(bearing)} if bearing is not None else {}),
            **({"clock": str(clock)} if clock is not None else {}),
            **({"position": pos} if pos else {}),
            "distance": dist or "unknown",
            "moving": moving,
            "confidence": 0.0,
        })

    summary = text.strip()
    if len(summary) > 160:
        summary = summary[:157] + "..."
    return {"summary": summary, "obstacles": obstacles, "risks": risks}


# ---------- coerce ----------

def coerce_obs(obs_like: Union[Dict[str, Any], str]) -> Dict[str, Any]:
    if isinstance(obs_like, dict):
        return {
            "summary":      obs_like.get("summary") or "",
            "obstacles":    obs_like.get("obstacles") or [],
            "risks":        obs_like.get("risks") or [],
            "people_count": obs_like.get("people_count", 0),
            "path_clear":   obs_like.get("path_clear", True),
        }
    text = str(obs_like or "")
    as_json = _extract_json_anywhere(text)
    if as_json:
        return coerce_obs(as_json)
    return _from_free_text(text)


# ---------- hospital-aware heuristic policy ----------

class SimpleHeuristicPolicy:
    """
    Hospital-aware rule-based navigation policy.

    Priority order:
      1. HOLD  — vulnerable entity (person/patient/wheelchair…) near+ahead
      2. HOLD  — high-risk keywords in risks list (emergency, fall, spill…)
      3. HOLD  — path explicitly marked not clear
      4. ROUTE — obstacle ahead, try left/right/back
      5. MOVE  — forward if clear
    """

    @staticmethod
    def decide(obs_like: Union[Dict[str, Any], str], instruction: str = "") -> Dict[str, str]:
        obs = coerce_obs(obs_like)
        obstacles: List[Dict[str, Any]] = obs.get("obstacles") or []
        risks: List[str] = obs.get("risks") or []
        path_clear: bool = obs.get("path_clear", True)

        counts = {"forward": 0, "left": 0, "right": 0, "back": 0}
        hold_reasons: List[str] = []

        for o in obstacles:
            name = (o.get("name") or "").lower()
            dist = (o.get("distance") or "unknown").lower()
            moving = bool(o.get("moving", False))

            if "bearing_deg" in o and isinstance(o["bearing_deg"], (int, float)):
                sector = _sector_from_bearing(float(o["bearing_deg"]))
            else:
                pos = (o.get("position") or "").lower()
                sector = {"left": "left", "right": "right", "center": "forward"}.get(pos, "forward")

            if _is_near(dist):
                counts[sector] += 1

            if sector == "forward":
                # Check explicit urgency field first (from VLM JSON)
                if _urgency(o) == "high":
                    hold_reasons.append(f"high urgency {name} ahead ({dist})")
                elif _name_matches(name, _HOLD_ENTITIES):
                    if _is_near(dist):
                        hold_reasons.append(f"near {name} ahead")
                    elif _is_near_or_mid(dist) and moving:
                        hold_reasons.append(f"moving {name} approaching")

        # Risk-based hold
        risk_text = " ".join(str(r) for r in risks).lower()
        for kw in _HIGH_RISK_KEYWORDS:
            if kw in risk_text:
                hold_reasons.append(f"risk detected: {kw}")
                break

        # Explicit path blocked
        if not path_clear:
            hold_reasons.append("path marked as blocked")

        if hold_reasons:
            reason_str = "; ".join(hold_reasons[:2])
            return {
                "direction": "hold",
                "reason": f"Holding: {reason_str}. Waiting for clearance.",
                "urgency": "high",
            }

        order = ["forward", "left", "right", "back"]
        best = min(order, key=lambda s: (counts[s], order.index(s)))

        if counts[best] == 0 and best == "forward":
            return {"direction": "forward", "reason": "Path ahead is clear.", "urgency": "low"}
        elif counts[best] == 0:
            return {"direction": best, "reason": f"Routing {best} — fewer obstacles.", "urgency": "medium"}
        else:
            return {"direction": best, "reason": f"All paths have obstacles; taking least-blocked ({best}).", "urgency": "medium"}


def decide_core(obs_like: Union[Dict[str, Any], str], instruction: str = "") -> Dict[str, str]:
    """Entry point used by views."""
    return SimpleHeuristicPolicy.decide(obs_like, instruction)
