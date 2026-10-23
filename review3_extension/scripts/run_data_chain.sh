#!/bin/bash
# Waits for tracking to finish, then builds features, relation pairs and temporal sequences for the NEW videos.
cd "$HOME/Downloads/cv_work" || exit 1
while pgrep -f 06_run_tracking.py > /dev/null; do sleep 10; done
echo "== tracking done $(date)"
python3 scripts/07_extract_features.py 2>&1 | grep -v Warning | tail -3
echo "== features done $(date)"
python3 scripts/08_prepare_relation_data.py 2>&1 | grep -v Warning | tail -6
echo "== relation data done $(date)"
python3 scripts/10_prepare_temporal_data.py 2>&1 | grep -v Warning | tail -5
echo "== temporal data done $(date)"
