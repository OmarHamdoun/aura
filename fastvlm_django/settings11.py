from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes", "on")

# ── Security ─────────────────────────────────────────────────────────
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-secret-key-change-me")
DEBUG      = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = os.environ.get("DJANGO_ALLOWED_HOSTS", "*").split(",")

# ── Apps ─────────────────────────────────────────────────────────────
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "analyzer",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF      = "fastvlm_django.urls"
WSGI_APPLICATION  = "fastvlm_django.wsgi.application"
ASGI_APPLICATION  = "fastvlm_django.asgi.application"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

# ── Database ─────────────────────────────────────────────────────────
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME":   BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

# ── Internationalisation ──────────────────────────────────────────────
LANGUAGE_CODE = "en-us"
TIME_ZONE     = "Europe/Amsterdam"
USE_I18N      = True
USE_TZ        = True

# ── Static / Media ───────────────────────────────────────────────────
STATIC_URL      = "static/"
STATIC_ROOT     = BASE_DIR / "staticfiles"
STATICFILES_DIRS = []

MEDIA_URL  = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"


# ════════════════════════════════════════════════════════════════════
#  AURA  —  Hospital Robot Configuration
# ════════════════════════════════════════════════════════════════════

# ── YOLO detector ────────────────────────────────────────────────────
# Path to your trained hospital model weights
YOLO_MODEL_PATH = os.environ.get(
    "YOLO_MODEL_PATH",
    "/home/vision/work/aura/models/hospital_hds/weights/best.pt",
)

# Detection confidence threshold (0.35 = best F1 from your training results)
# Increase to 0.5 for fewer false positives, decrease to 0.25 for higher recall
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.35"))

# GPU device: "0" = first GPU, "cpu" = CPU only
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "0")

# Max detections per frame (30 is plenty for a hospital corridor)
YOLO_MAX_DET = int(os.environ.get("YOLO_MAX_DET", "30"))

# ── VLM models ───────────────────────────────────────────────────────
# Path override for Mini-InternVL2-DriveLM (leave empty to use default HuggingFace download)
MINI_INTERNVL2_DRIVELM_PATH = os.environ.get(
    "MINI_INTERNVL2_DRIVELM_PATH",
    "/home/vision/work/Mini-InternVL2-2B-DA-DriveLM",
)

# ── Caption persistence ───────────────────────────────────────────────
# Save captions to JSONL files on disk (useful for evaluation / audit trail)
CAPTIONS_SAVE_ENABLED = _env_flag("CAPTIONS_SAVE_ENABLED", "0")
CAPTIONS_SAVE_DIR     = BASE_DIR / "media" / "captions"

# ── MLflow experiment tracking ────────────────────────────────────────
# Set MLFLOW_TRACKING_URI to enable (e.g. "http://localhost:5000")
MLFLOW_SAVE_ENABLED     = _env_flag("MLFLOW_SAVE_ENABLED", "0")
MLFLOW_TRACKING_URI     = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT_NAME  = os.environ.get("MLFLOW_EXPERIMENT_NAME", "aura-hospital")
MLFLOW_RUN_NAME_PREFIX  = os.environ.get("MLFLOW_RUN_NAME_PREFIX", "caption")
MLFLOW_LOG_EVERY_N      = int(os.environ.get("MLFLOW_LOG_EVERY_N", "1"))
MLFLOW_LOG_THUMB_IMAGES = _env_flag("MLFLOW_LOG_THUMB_IMAGES", "1")
MLFLOW_MAX_THUMB_IMAGES = int(os.environ.get("MLFLOW_MAX_THUMB_IMAGES", "1"))

# ── Action / Navigation policy ────────────────────────────────────────
# "heuristic" = fast rule-based (default, no API key needed)
# "openai"    = GPT-4o-mini for smarter decisions (requires OPENAI_API_KEY)
ACTION_POLICY_PROVIDER = os.environ.get("ACTION_POLICY_PROVIDER", "heuristic")

# OpenAI settings (only used when ACTION_POLICY_PROVIDER = "openai")
ACTION_OPENAI_MODEL           = os.environ.get("ACTION_OPENAI_MODEL", "gpt-4o-mini")
ACTION_OPENAI_URL             = os.environ.get("ACTION_OPENAI_URL", "https://api.openai.com/v1/chat/completions")
ACTION_OPENAI_TIMEOUT         = int(os.environ.get("ACTION_OPENAI_TIMEOUT", "15"))
ACTION_OPENAI_GENERATE_POLICY = _env_flag("ACTION_OPENAI_GENERATE_POLICY", "0")
ACTION_OPENAI_POLICY_MAX_CHARS = int(os.environ.get("ACTION_OPENAI_POLICY_MAX_CHARS", "280"))

# Caption summarisation via OpenAI (condenses long VLM output for overlay)
# "none" = raw caption, "openai" = summarise via GPT
CAPTION_SUMMARY_PROVIDER      = os.environ.get("CAPTION_SUMMARY_PROVIDER", "none")
CAPTION_SUMMARY_OPENAI_MODEL  = os.environ.get("CAPTION_SUMMARY_OPENAI_MODEL", "gpt-4o-mini")
CAPTION_SUMMARY_OPENAI_TIMEOUT = int(os.environ.get("CAPTION_SUMMARY_OPENAI_TIMEOUT", "12"))
CAPTION_SUMMARY_MAX_CHARS     = int(os.environ.get("CAPTION_SUMMARY_MAX_CHARS", "180"))

# ── Text-to-speech (OpenTTS) ──────────────────────────────────────────
# Set OPENTTS_URL to enable (e.g. "http://localhost:5500")
# Robot will speak hazard alerts aloud
OPENTTS_URL           = os.environ.get("OPENTTS_URL", "")
OPENTTS_SPEAK_ENABLED = _env_flag("OPENTTS_SPEAK_ENABLED", "0")
OPENTTS_TTS_URL       = os.environ.get("OPENTTS_TTS_URL", "")
OPENTTS_VOICE         = os.environ.get("OPENTTS_VOICE", "larynx:harvard")
OPENTTS_LANG          = os.environ.get("OPENTTS_LANG", "en")
OPENTTS_FORMAT        = os.environ.get("OPENTTS_FORMAT", "wav")
OPENTTS_RATE          = float(os.environ.get("OPENTTS_RATE", "1.0"))
OPENTTS_EFFECT        = os.environ.get("OPENTTS_EFFECT", "robot")
OPENTTS_MIN_INTERVAL  = float(os.environ.get("OPENTTS_MIN_INTERVAL", "1.5"))
OPENTTS_SAVE_WAV      = _env_flag("OPENTTS_SAVE_WAV", "0")
OPENTTS_DEBUG         = _env_flag("OPENTTS_DEBUG", "0")
