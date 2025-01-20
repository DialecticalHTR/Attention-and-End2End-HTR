#!/bin/bash

DATA_DIR='datasets'
LOGS_DIR='logs'
export PYTHONPATH=$PYTHONPATH:src

python3 src/attention_model/test.py --log_dir $LOGS_DIR --log_name ocr_attention.txt --write_errors \
   --saved_model attention_hkr/best_cer.pth --eval_stage test --batch_size 64 --images_path datasets/img_cyrillic
