#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  ChickpeaAI — Full Pipeline Runner
#  Usage:
#    bash run_pipeline.sh           # generate data + train + start API
#    bash run_pipeline.sh api       # only start API (model must exist)
#    bash run_pipeline.sh train     # only train (dataset must exist)
#    bash run_pipeline.sh generate  # only generate synthetic dataset
# ─────────────────────────────────────────────────────────────────

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="$SCRIPT_DIR/backend"
DATASET="$SCRIPT_DIR/dataset"
MODELS="$SCRIPT_DIR/models"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RESET='\033[0m'

step() { echo -e "\n${CYAN}▶ $1${RESET}"; }
ok()   { echo -e "${GREEN}✓ $1${RESET}"; }
warn() { echo -e "${YELLOW}⚠ $1${RESET}"; }

# ── Generate synthetic dataset ─────────────────────────────────────
generate_data() {
  step "Generating synthetic chickpea dataset (100 images/class)..."
  python3 "$BACKEND/dataset_generator.py" "$DATASET" 100
  ok "Dataset generated at: $DATASET"
}

# ── Train model ────────────────────────────────────────────────────
train_model() {
  step "Training Hybrid CNN-RNN model..."
  python3 "$BACKEND/train.py" \
    --data_dir "$DATASET" \
    --model_dir "$MODELS" \
    --epochs 30 \
    --batch_size 16 \
    --lr 0.001 \
    --patience 7
  ok "Model saved to: $MODELS/best_model.pth"
}

# ── Start API ──────────────────────────────────────────────────────
start_api() {
  step "Starting Flask API server..."
  echo ""
  echo "  ┌──────────────────────────────────────────┐"
  echo "  │  API:      http://localhost:5000          │"
  echo "  │  Frontend: http://localhost:5000          │"
  echo "  │                                           │"
  echo "  │  Press Ctrl+C to stop the server         │"
  echo "  └──────────────────────────────────────────┘"
  echo ""
  cd "$BACKEND" && python3 app.py
}

# ── Main ───────────────────────────────────────────────────────────
case "${1:-all}" in
  generate)
    generate_data ;;
  train)
    [ -d "$DATASET/train" ] || { warn "No dataset found. Run: bash run_pipeline.sh generate"; exit 1; }
    train_model ;;
  api)
    start_api ;;
  all|*)
    echo ""
    echo "╔══════════════════════════════════════════════╗"
    echo "║     ChickpeaAI — Chickpea Disease Detection  ║"
    echo "║         Hybrid CNN-RNN Deep Learning         ║"
    echo "╚══════════════════════════════════════════════╝"
    generate_data
    train_model
    start_api ;;
esac
