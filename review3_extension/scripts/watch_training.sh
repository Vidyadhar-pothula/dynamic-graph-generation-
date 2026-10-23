#!/bin/bash
# Live view of the YOLO fine-tune progress. Ctrl-C closes this view; training keeps running.
LOG="$HOME/Downloads/cv_work/logs/finetune.log"
while true; do
  clear
  echo "YOLO fine-tune on old + new VidOR data   (Ctrl-C closes this view, training keeps running)"
  echo
  tr '\r' '\n' < "$LOG" | tail -n 4
  sleep 3
done
