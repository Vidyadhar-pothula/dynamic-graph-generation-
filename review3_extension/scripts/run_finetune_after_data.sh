#!/bin/bash
cd "$HOME/Downloads/cv_work" || exit 1
while ! grep -q "temporal data done" logs/data_chain.log 2>/dev/null; do
  if ! pgrep -f run_data_chain.sh >/dev/null && ! pgrep -f 06_run_tracking.py >/dev/null; then echo "data chain ended without finishing"; break; fi
  sleep 30
done
if grep -q "temporal data done" logs/data_chain.log 2>/dev/null; then
  echo "== starting relation + temporal fine-tune $(date)"
  python3 scripts/12b_finetune_relation_temporal.py --epochs 10 --device cpu 2>&1 | grep -v -i warning
  echo "== fine-tune finished $(date)"
fi
