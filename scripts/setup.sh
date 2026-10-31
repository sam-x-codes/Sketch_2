#!/usr/bin/env bash
# Sketch2Shape setup — local NVIDIA box OR Kaggle GPU notebook (2× T4).
#
# Kaggle (Internet ON, Accelerator = GPU T4 x2):
#   Add this repo as a Dataset, then in a notebook cell:
#     !bash /kaggle/input/<dataset-name>/scripts/setup.sh
#   Or if the repo is already in /kaggle/working:
#     !bash scripts/setup.sh
#
# Optional Kaggle secret: HF_TOKEN  (needed for gated SDXL weights)
# Optional:    S2S_SKIP_WEIGHTS=1  to skip the long HF download
#              S2S_SKIP_PAINT_CUDA=1  to skip Hunyuan rasterizer compile
set -euo pipefail

log()  { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

is_kaggle() {
  [[ -d /kaggle ]] || [[ -n "${KAGGLE_KERNEL_RUN_TYPE:-}" ]] || [[ -n "${KAGGLE_URL_BASE:-}" ]]
}

writable() {
  local d="$1"
  mkdir -p "$d" 2>/dev/null || return 1
  touch "$d/.s2s_write_test" 2>/dev/null || return 1
  rm -f "$d/.s2s_write_test"
  return 0
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Resolve a writable project root (Kaggle datasets are read-only)
# ---------------------------------------------------------------------------
if is_kaggle; then
  DEST_ROOT="${S2S_KAGGLE_ROOT:-/kaggle/working/Sketch-main}"
  if ! writable "$SRC_ROOT"; then
    log "Kaggle dataset is read-only — copying repo to $DEST_ROOT"
    mkdir -p "$DEST_ROOT"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete \
        --exclude '.venv' --exclude '.venv-instantmesh' \
        --exclude 'outputs' --exclude 'third_party' \
        --exclude '__pycache__' --exclude '.git' \
        "$SRC_ROOT/" "$DEST_ROOT/"
    else
      cp -a "$SRC_ROOT"/. "$DEST_ROOT"/
    fi
    PROJECT_ROOT="$DEST_ROOT"
  else
    PROJECT_ROOT="$SRC_ROOT"
  fi
  export HF_HOME="${HF_HOME:-/kaggle/working/hf_home}"
  export S2S_OUTPUT_DIR="${S2S_OUTPUT_DIR:-/kaggle/working/outputs}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  export S2S_BACKEND="${S2S_BACKEND:-auto}"
  export S2S_TEXTURE_PAINT="${S2S_TEXTURE_PAINT:-1}"
  export S2S_OCTREE="${S2S_OCTREE:-384}"
  export S2S_MC_RESOLUTION="${S2S_MC_RESOLUTION:-384}"
  export S2S_TEXTURE_SIZE="${S2S_TEXTURE_SIZE:-2048}"
  export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
  mkdir -p "$HF_HOME" "$S2S_OUTPUT_DIR"
else
  PROJECT_ROOT="$SRC_ROOT"
fi

cd "$PROJECT_ROOT"
[[ -d sketch2shape ]] || die "Cannot find sketch2shape/ under $PROJECT_ROOT"

log "Project root: $PROJECT_ROOT"
if is_kaggle; then
  echo "Environment: Kaggle"
else
  echo "Environment: local"
fi

# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
if is_kaggle; then
  PYTHON_BIN="${PYTHON_BIN:-python}"
else
  PYTHON_BIN=""
  for candidate in python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
        PYTHON_BIN="$candidate"
        break
      fi
    fi
  done
  [[ -n "$PYTHON_BIN" ]] || die "Need Python 3.10+"
fi
echo "Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

HAS_CUDA=0
if command -v nvidia-smi >/dev/null 2>&1; then
  HAS_CUDA=1
  nvidia-smi -L || true
fi

# ---------------------------------------------------------------------------
# Hugging Face token (Kaggle secret HF_TOKEN, or env HF_TOKEN)
# ---------------------------------------------------------------------------
if [[ -z "${HF_TOKEN:-}" && -z "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
  SECRET="$("$PYTHON_BIN" - <<'PY' 2>/dev/null || true
try:
    from kaggle_secrets import UserSecretsClient
    print(UserSecretsClient().get_secret("HF_TOKEN") or "")
except Exception:
    pass
PY
)"
  if [[ -n "${SECRET:-}" ]]; then
    export HF_TOKEN="$SECRET"
    echo "HF_TOKEN loaded from Kaggle secrets"
  fi
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}"
  "$PYTHON_BIN" - <<'PY' || true
import os
try:
    from huggingface_hub import login
    login(token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"), add_to_git_credential=False)
    print("Hugging Face login: ok")
except Exception as exc:
    print("Hugging Face login skipped:", exc)
PY
fi

# ---------------------------------------------------------------------------
# Install Python packages
# Kaggle already ships a CUDA PyTorch that matches the image — do NOT replace it.
# ---------------------------------------------------------------------------
if is_kaggle; then
  log "Installing Python deps into the Kaggle kernel (keeping preinstalled PyTorch)"
  "$PYTHON_BIN" -m pip install -q --upgrade pip wheel setuptools
  "$PYTHON_BIN" -m pip install -q -r "$PROJECT_ROOT/requirements.txt"
  "$PYTHON_BIN" -m pip install -q onnxruntime-gpu || warn "onnxruntime-gpu skipped"
  "$PYTHON_BIN" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())"
  if ! "$PYTHON_BIN" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)"; then
    warn "This Kaggle session has no GPU. Switch Accelerator to GPU T4 x2 and re-run."
  fi
else
  if [[ ! -d .venv ]]; then
    log "Creating .venv"
    "$PYTHON_BIN" -m venv .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate
  PYTHON_BIN="$(command -v python)"
  "$PYTHON_BIN" -m pip install -q --upgrade pip wheel setuptools
  if [[ "$HAS_CUDA" -eq 1 ]]; then
    log "Installing PyTorch CUDA 12.1"
    "$PYTHON_BIN" -m pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    "$PYTHON_BIN" -m pip install -q xformers || warn "xformers skipped"
  else
    log "No NVIDIA GPU — CPU PyTorch (3D stage needs CUDA)"
    "$PYTHON_BIN" -m pip install -q torch torchvision torchaudio
  fi
  "$PYTHON_BIN" -m pip install -q -r "$PROJECT_ROOT/requirements.txt"
fi

# ---------------------------------------------------------------------------
# Third-party 3D code (shape only; paint CUDA ops are optional)
# ---------------------------------------------------------------------------
log "Cloning Hunyuan3D-2 and TripoSR"
mkdir -p third_party
if [[ ! -d third_party/Hunyuan3D-2/hy3dgen ]]; then
  git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git third_party/Hunyuan3D-2
else
  echo "Hunyuan3D-2 already present"
fi
if [[ ! -d third_party/TripoSR/tsr ]]; then
  git clone --depth 1 https://github.com/stabilityai/TripoSR.git third_party/TripoSR
else
  echo "TripoSR already present"
fi

if [[ "$HAS_CUDA" -eq 1 && "${S2S_SKIP_PAINT_CUDA:-0}" != "1" ]]; then
  log "Optional Hunyuan paint CUDA extensions (safe to fail — colour bake still works)"
  set +e
  (
    cd third_party/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
    "$PYTHON_BIN" setup.py install
  )
  (
    cd third_party/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
    "$PYTHON_BIN" setup.py install
  )
  set -e
fi

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
if [[ "${S2S_SKIP_WEIGHTS:-0}" == "1" ]]; then
  warn "Skipping weight download (S2S_SKIP_WEIGHTS=1)"
else
  log "Downloading model weights (10–20 min, ~15 GB). Re-run this script if it drops."
  "$PYTHON_BIN" "$PROJECT_ROOT/scripts/download_weights.py" \
    || warn "Weight download incomplete — re-run: $PYTHON_BIN scripts/download_weights.py"
fi

# Persist Kaggle env for later notebook cells (`!bash` does not export into Python).
if is_kaggle; then
  cat > "$PROJECT_ROOT/kaggle.env.sh" <<EOF
export HF_HOME="${HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE}"
export S2S_OUTPUT_DIR="${S2S_OUTPUT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
export S2S_BACKEND="${S2S_BACKEND}"
export S2S_TEXTURE_PAINT="${S2S_TEXTURE_PAINT}"
export S2S_OCTREE="${S2S_OCTREE}"
export S2S_MC_RESOLUTION="${S2S_MC_RESOLUTION}"
export S2S_TEXTURE_SIZE="${S2S_TEXTURE_SIZE}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/third_party/Hunyuan3D-2:${PROJECT_ROOT}/third_party/TripoSR:\${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"
EOF
  cat > /kaggle/working/s2s_paths.txt <<EOF
PROJECT_ROOT=${PROJECT_ROOT}
HF_HOME=${HF_HOME}
S2S_OUTPUT_DIR=${S2S_OUTPUT_DIR}
EOF
fi

log "Setup finished"
echo "Project:  $PROJECT_ROOT"
if is_kaggle; then
  echo
  echo "Next, in a notebook cell:"
  echo "  import os, sys"
  echo "  os.chdir('$PROJECT_ROOT')"
  echo "  sys.path.insert(0, '$PROJECT_ROOT')"
  echo "  # then either:"
  echo "  # !python run_pipeline.py --input examples/inputs/chair.png --prompt 'oak chair'"
  echo "  # or:"
  echo "  # from app import build_app; build_app().launch(share=True)"
  echo
  echo "If SDXL download was gated, add Kaggle secret HF_TOKEN (Hugging Face token) and re-run."
else
  echo "Activate:  source $PROJECT_ROOT/.venv/bin/activate"
  echo "Web app:   python app.py"
  echo "CLI:       python run_pipeline.py --input examples/inputs/chair.png --prompt 'wooden chair'"
fi
if [[ "$HAS_CUDA" -eq 0 ]]; then
  echo
  warn "No NVIDIA GPU in this session. 3D reconstruction will not run."
fi
