#!/usr/bin/env bash
set -euo pipefail

# Run MLflow + Django app together.
# Usage:
#   ./run_with_mlflow.sh
#   MLFLOW_TRACKING_URI=http://127.0.0.1:5001 ./run_with_mlflow.sh
#   USE_PROJECT_VENV=true ./run_with_mlflow.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Environment selection:
# - USE_PROJECT_VENV=true: force activate project .venv
# - USE_PROJECT_VENV=false: never activate project .venv
# - USE_PROJECT_VENV=auto (default): use active env if present, else project .venv if available
USE_PROJECT_VENV_MODE="${USE_PROJECT_VENV:-auto}"
if [[ "${USE_PROJECT_VENV_MODE}" =~ ^(1|true|yes|on)$ ]]; then
  if [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source ".venv/bin/activate"
    echo "Environment: project .venv (forced)"
  else
    echo "Warning: USE_PROJECT_VENV=true but .venv/bin/activate not found; using current environment."
  fi
elif [[ "${USE_PROJECT_VENV_MODE}" =~ ^(0|false|no|off)$ ]]; then
  echo "Environment: current shell (USE_PROJECT_VENV=false)"
else
  if [[ -n "${VIRTUAL_ENV:-}" || -n "${CONDA_PREFIX:-}" ]]; then
    echo "Environment: current shell (detected active env)"
  elif [[ -f ".venv/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source ".venv/bin/activate"
    echo "Environment: project .venv (auto fallback)"
  else
    echo "Environment: current shell (no active env and no project .venv)"
  fi
fi

if ! command -v mlflow >/dev/null 2>&1; then
  echo "mlflow command not found. Install deps first: pip install -r requirements.txt"
  exit 1
fi

if ! python -c "import django" >/dev/null 2>&1; then
  echo "Django is not installed in this environment. Install deps first: pip install -r requirements.txt"
  exit 1
fi

MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_BACKEND_STORE_URI="${MLFLOW_BACKEND_STORE_URI:-sqlite:///${ROOT_DIR}/mlflow.db}"
MLFLOW_ARTIFACT_ROOT="${MLFLOW_ARTIFACT_ROOT:-${ROOT_DIR}/mlruns}"
MLFLOW_HOST="${MLFLOW_HOST:-127.0.0.1}"
OPENTTS_AUTOSTART="${OPENTTS_AUTOSTART:-true}"
OPENTTS_PORT="${OPENTTS_PORT:-5500}"
OPENTTS_IMAGE="${OPENTTS_IMAGE:-synesthesiam/opentts:en}"
OPENTTS_CONTAINER_NAME="${OPENTTS_CONTAINER_NAME:-aura-opentts}"

export MLFLOW_SAVE_ENABLED="${MLFLOW_SAVE_ENABLED:-true}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://${MLFLOW_HOST}:${MLFLOW_PORT}}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-aura-captions}"
export MLFLOW_RUN_NAME_PREFIX="${MLFLOW_RUN_NAME_PREFIX:-caption}"
export MLFLOW_LOG_EVERY_N="${MLFLOW_LOG_EVERY_N:-1}"
export MLFLOW_LOG_THUMB_IMAGES="${MLFLOW_LOG_THUMB_IMAGES:-true}"
export MLFLOW_MAX_THUMB_IMAGES="${MLFLOW_MAX_THUMB_IMAGES:-1}"
export ACTION_POLICY_PROVIDER="${ACTION_POLICY_PROVIDER:-heuristic}"
export ACTION_OPENAI_MODEL="${ACTION_OPENAI_MODEL:-gpt-4o-mini}"
export ACTION_OPENAI_TIMEOUT="${ACTION_OPENAI_TIMEOUT:-15}"
export ACTION_OPENAI_GENERATE_POLICY="${ACTION_OPENAI_GENERATE_POLICY:-false}"
export ACTION_OPENAI_POLICY_MAX_CHARS="${ACTION_OPENAI_POLICY_MAX_CHARS:-280}"
export CAPTION_SUMMARY_PROVIDER="${CAPTION_SUMMARY_PROVIDER:-openai}"
export CAPTION_SUMMARY_OPENAI_MODEL="${CAPTION_SUMMARY_OPENAI_MODEL:-gpt-4o-mini}"
export CAPTION_SUMMARY_OPENAI_TIMEOUT="${CAPTION_SUMMARY_OPENAI_TIMEOUT:-12}"
export CAPTION_SUMMARY_MAX_CHARS="${CAPTION_SUMMARY_MAX_CHARS:-180}"

if [[ "${ACTION_POLICY_PROVIDER}" == "openai" || "${CAPTION_SUMMARY_PROVIDER}" == "openai" ]] && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "Warning: OpenAI provider enabled but OPENAI_API_KEY is not set; OpenAI paths will fall back to local behavior."
fi

mkdir -p "${MLFLOW_ARTIFACT_ROOT}"

cleanup() {
  if [[ -n "${MLFLOW_PID:-}" ]]; then
    kill "${MLFLOW_PID}" >/dev/null 2>&1 || true
  fi
  if [[ "${OPENTTS_STARTED_BY_SCRIPT:-false}" == "true" ]]; then
    docker stop "${OPENTTS_CONTAINER_NAME}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${OPENTTS_AUTOSTART}" =~ ^(1|true|yes|on)$ ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "Warning: docker not found; OpenTTS container was not started."
  else
    RUNNING_ID="$(docker ps -q -f "name=^/${OPENTTS_CONTAINER_NAME}$" 2>/dev/null || true)"
    if [[ -n "${RUNNING_ID}" ]]; then
      echo "OpenTTS already running in container: ${OPENTTS_CONTAINER_NAME}"
    else
      EXISTING_ID="$(docker ps -aq -f "name=^/${OPENTTS_CONTAINER_NAME}$" 2>/dev/null || true)"
      if [[ -n "${EXISTING_ID}" ]]; then
        echo "Starting existing OpenTTS container: ${OPENTTS_CONTAINER_NAME}"
        if docker start "${OPENTTS_CONTAINER_NAME}" >/dev/null 2>&1; then
          OPENTTS_STARTED_BY_SCRIPT=true
        else
          echo "Warning: failed to start existing OpenTTS container ${OPENTTS_CONTAINER_NAME}."
        fi
      else
        echo "Starting OpenTTS on 0.0.0.0:${OPENTTS_PORT} using image ${OPENTTS_IMAGE}"
        if docker run -d --rm --name "${OPENTTS_CONTAINER_NAME}" -p "${OPENTTS_PORT}:5500" "${OPENTTS_IMAGE}" >/dev/null 2>&1; then
          OPENTTS_STARTED_BY_SCRIPT=true
        else
          echo "Warning: failed to run OpenTTS container (docker run -it -p ${OPENTTS_PORT}:5500 ${OPENTTS_IMAGE})."
        fi
      fi
    fi
  fi
fi

echo "Starting MLflow server on ${MLFLOW_HOST}:${MLFLOW_PORT}"
mlflow server \
  --host "${MLFLOW_HOST}" \
  --port "${MLFLOW_PORT}" \
  --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
  --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}" &
MLFLOW_PID=$!

sleep 2
echo "MLflow tracking URI: ${MLFLOW_TRACKING_URI}"
echo "Action provider: ${ACTION_POLICY_PROVIDER} (model: ${ACTION_OPENAI_MODEL})"
echo "Action policy generation: ${ACTION_OPENAI_GENERATE_POLICY}"
echo "Caption summary provider: ${CAPTION_SUMMARY_PROVIDER} (model: ${CAPTION_SUMMARY_OPENAI_MODEL})"
echo "Starting Django app on 0.0.0.0:8000"
python manage.py runserver 0.0.0.0:8000
