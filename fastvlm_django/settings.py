# fastvlm_django/settings.py
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = "dev-secret-key-change-me"
DEBUG = True
ALLOWED_HOSTS = ["*"]  # tighten for production

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

ROOT_URLCONF = "fastvlm_django.urls"

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

WSGI_APPLICATION = "fastvlm_django.wsgi.application"
ASGI_APPLICATION = "fastvlm_django.asgi.application"

DATABASES = {
    "default": {"ENGINE": "django.db.backends.sqlite3", "NAME": BASE_DIR / "db.sqlite3"}
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Europe/Amsterdam"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = []

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# OpenTTS config
OPENTTS_URL = os.environ.get("OPENTTS_URL", "http://localhost:5500")
OPENTTS_TTS_URL = os.environ.get("OPENTTS_TTS_URL", f"{OPENTTS_URL}/api/tts")
OPENTTS_VOICE = os.environ.get("OPENTTS_VOICE", "larynx:harvard")
OPENTTS_LANG = os.environ.get("OPENTTS_LANG", "en")
OPENTTS_FORMAT = os.environ.get("OPENTTS_FORMAT", "wav")
OPENTTS_RATE = float(os.environ.get("OPENTTS_RATE", "1.0"))
OPENTTS_EFFECT = os.environ.get("OPENTTS_EFFECT", "robot")
OPENTTS_MIN_INTERVAL = float(os.environ.get("OPENTTS_MIN_INTERVAL", "1.5"))
OPENTTS_SPEAK_ENABLED = os.environ.get("OPENTTS_SPEAK_ENABLED", "true").lower() in ("1", "true", "yes", "on")
OPENTTS_SAVE_WAV = os.environ.get("OPENTTS_SAVE_WAV", "false").lower() in ("1", "true", "yes", "on")
OPENTTS_DEBUG = os.environ.get("OPENTTS_DEBUG", "true").lower() in ("1", "true", "yes", "on")

# Caption persistence
CAPTIONS_SAVE_ENABLED = os.environ.get("CAPTIONS_SAVE_ENABLED", "true").lower() in ("1", "true", "yes", "on")
CAPTIONS_SAVE_DIR = Path(os.environ.get("CAPTIONS_SAVE_DIR", BASE_DIR / "media" / "captions"))

# MLflow caption logging
MLFLOW_SAVE_ENABLED = os.environ.get("MLFLOW_SAVE_ENABLED", "false").lower() in ("1", "true", "yes", "on")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "aura-captions")
MLFLOW_RUN_NAME_PREFIX = os.environ.get("MLFLOW_RUN_NAME_PREFIX", "caption")
MLFLOW_LOG_EVERY_N = int(os.environ.get("MLFLOW_LOG_EVERY_N", "1"))
MLFLOW_LOG_THUMB_IMAGES = os.environ.get("MLFLOW_LOG_THUMB_IMAGES", "true").lower() in ("1", "true", "yes", "on")
MLFLOW_MAX_THUMB_IMAGES = int(os.environ.get("MLFLOW_MAX_THUMB_IMAGES", "1"))

# Action policy configuration
ACTION_POLICY_PROVIDER = os.environ.get("ACTION_POLICY_PROVIDER", "heuristic").strip().lower()
ACTION_OPENAI_MODEL = os.environ.get("ACTION_OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
ACTION_OPENAI_URL = os.environ.get("ACTION_OPENAI_URL", "https://api.openai.com/v1/chat/completions").strip()
ACTION_OPENAI_TIMEOUT = int(os.environ.get("ACTION_OPENAI_TIMEOUT", "15"))
ACTION_OPENAI_GENERATE_POLICY = os.environ.get("ACTION_OPENAI_GENERATE_POLICY", "false").lower() in ("1", "true", "yes", "on")
ACTION_OPENAI_POLICY_MAX_CHARS = int(os.environ.get("ACTION_OPENAI_POLICY_MAX_CHARS", "280"))

# Optional caption summarization (used for overlay + OpenTTS speech)
CAPTION_SUMMARY_PROVIDER = os.environ.get("CAPTION_SUMMARY_PROVIDER", "none").strip().lower()
CAPTION_SUMMARY_OPENAI_MODEL = os.environ.get("CAPTION_SUMMARY_OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
CAPTION_SUMMARY_OPENAI_TIMEOUT = int(os.environ.get("CAPTION_SUMMARY_OPENAI_TIMEOUT", "12"))
CAPTION_SUMMARY_MAX_CHARS = int(os.environ.get("CAPTION_SUMMARY_MAX_CHARS", "180"))

# Local model path for Mini-InternVL2-2B-DA-DriveLM
MINI_INTERNVL2_DRIVELM_PATH = os.environ.get(
    "MINI_INTERNVL2_DRIVELM_PATH",
    "/home/vision/work/Mini-InternVL2-2B-DA-DriveLM",
)
os.environ.setdefault("MINI_INTERNVL2_DRIVELM_PATH", MINI_INTERNVL2_DRIVELM_PATH)
