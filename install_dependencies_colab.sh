#!/bin/bash
# Script per installare le dipendenze da zero su Google Colab
# Eseguire in una cella del notebook con: !bash install_dependencies_colab.sh
# Installa tutto partendo da un ambiente pulito

# Step 1: Aggiorna pip
pip install -q --upgrade pip

# Step 2: Aggiorna numpy (questo può rompere pandas)
pip install -q --upgrade --force-reinstall numpy>=2.0

# Step 3: Reinstalla pandas per compatibilità con numpy 2.0
pip install -q --upgrade --force-reinstall pandas

# Step 4: Installa le altre librerie
pip install -q --upgrade \
  torch \
  transformers==4.44.2 \
  peft==0.12.0 \
  accelerate==0.33.0 \
  datasets \
  bitsandbytes \
  sentencepiece \
  fsspec \
  matplotlib \
  scikit-learn \
  plotly

