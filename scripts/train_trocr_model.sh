#!/bin/bash

DATA_DIR='datasets'
LOGS_DIR='logs'
export PYTHONPATH=$PYTHONPATH:src

python3 src/trocr_model/train.py --log_dir $LOGS_DIR --log_name train_trocr.txt --out_dir trocr \
  --data_dir $DATA_DIR --label_files gt_cyrillic.csv
