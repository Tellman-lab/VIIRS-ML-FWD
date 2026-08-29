#!/bin/bash
# Example usage script for the inference pipeline

# Stop immediately on error
set -e

# Default root path (optional, can be overridden)
ROOT_PATH=$(pwd)

# Example command
python Inference.py \
  --folderNames "yourdatafolder" \
  --rootPath "$ROOT_PATH" \
  --modelWeightsPath "ModelWeights/model" \
  --downloadModelWeights y \
  --laadsEmail "yournasaearthdata@email.com" \
  --laadsApikey "yourearthdatatoken" \
  --downloadOrder y \
  --prepFiles y \
  --resampleFiles y \
  --chipFiles y \
  --maskClouds 3 \
  --overwriteInf y \
  --mosaicInf y \
  --deleteInfChips n \
  --device 0 \
  --bs 64 \
  --imageBuffer 64 \
  --gradientMethod linear
