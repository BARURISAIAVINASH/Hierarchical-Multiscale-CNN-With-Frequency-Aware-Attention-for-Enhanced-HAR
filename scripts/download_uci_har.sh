#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${ROOT_DIR}/data/uci_har"
ZIP_PATH="${DATA_DIR}/UCI_HAR_Dataset.zip"

mkdir -p "${DATA_DIR}"

echo "[1/3] Downloading UCI HAR zip..."
curl -L "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip" -o "${ZIP_PATH}"

echo "[2/3] Unzipping..."
unzip -q -o "${ZIP_PATH}" -d "${DATA_DIR}"

echo "[3/3] Done. Listing:"
ls -la "${DATA_DIR}"
