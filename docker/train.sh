#!/bin/bash
set -e

# 1. Run training
python -u docker/train.py --config docker/train.yaml

# 2. Fix permissions so host user can access checkpoints
chown -R 1000:1000 /app/tmp_model_checkpoints /app/model_checkpoints
chown 1000:1000 training_log.csv
