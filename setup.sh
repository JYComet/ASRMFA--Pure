#!/usr/bin/env bash
set -euo pipefail

echo "============================================"
echo "  Chinese MFA Pipeline - One-Click Setup"
echo "============================================"
echo ""

# Check conda
if ! command -v conda &>/dev/null; then
    echo "[ERROR] conda not found."
    echo "  Download Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "[OK] conda found."

# Step 1: Create conda environment
echo ""
echo "[1/3] Creating mfa_chinese environment from environment.yml..."
echo "  (This may take 10-20 minutes on first run)"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
conda env create -f "$SCRIPT_DIR/environment.yml" --force 2>&1 || {
    echo "[WARNING] conda env create had issues."
    echo "  Try manual: conda env create -f environment.yml"
}

# Step 2: Convert pinyin dict to IPA format
echo ""
echo "[2/3] Generating MFA IPA dictionary..."
eval "$(conda shell.bash hook)"
conda activate mfa_chinese
python "$SCRIPT_DIR/scripts/convert_dict_to_ipa.py" || {
    echo "[WARNING] IPA dict generation failed."
    echo "  Run manually: python scripts/convert_dict_to_ipa.py"
}

# Step 3: Download pretrained models (MFA + Qwen3) from official sources
echo ""
echo "[3/3] Downloading pretrained models (MFA acoustic/g2p/dictionary + Qwen3)..."
echo "  Qwen3 weights are several GB - this can take a while."
export MFA_ROOT_DIR="$SCRIPT_DIR/models/mfa"

python "$SCRIPT_DIR/scripts/download_models.py" || {
    echo "[WARNING] Model download incomplete."
    echo "  Re-run to resume (already-downloaded models are skipped):"
    echo "    python scripts/download_models.py"
}
echo ""
echo "  Behind a slow/blocked HuggingFace, use a mirror:"
echo "    HF_ENDPOINT=https://hf-mirror.com python scripts/download_models.py"

echo ""
echo "============================================"
echo "  Setup Complete!"
echo ""
echo "  Environment:  mfa_chinese"
echo "  Configuration: config.yaml"
echo "  Dictionary:    dict/mfa_ipa.dict"
echo "  Models:        models/mfa/  +  models/qwen3/"
echo ""
echo "  Quick start:"
echo "    1. Edit config.yaml - set 'workspace' path"
echo "    2. conda activate mfa_chinese"
echo "    3. python scripts/run_pipeline.py --data-dir /path/to/audio"
echo ""
echo "  Check what is installed:"
echo "    python scripts/download_models.py --check"
echo ""
echo "  See README.md for detailed usage."
echo "============================================"
