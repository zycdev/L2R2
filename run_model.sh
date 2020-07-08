#!/usr/bin/env bash

set -e

python predict.py \
  --input_file /data/anli.jsonl \
  --output_file /results/predictions.lst \
  --max_seq_len 128

#--cuda
