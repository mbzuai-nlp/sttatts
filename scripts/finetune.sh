#!/bin/bash

export WANDB_NAME=wandb_project_name
ASR_DIR= #path to asr .tsv,.txt manifest files, should include dict.txt
TTS_DIR= #path to tts .tsv,.txt manifest files, should include dict.txt
SAVE_DIR=
TRAIN_SET= # | separated list of train sets e.g train|train
VALID_SET= # | separated list of validation sets
BPE_TOKENIZER= #path to sentencepiece tokenizer
USER_DIR=sttatts/artst
CHECKPOINT_PATH= #path to the checkpoint to finetune from
WANDB_PROJECT= #wandb run name


mkdir -p ${SAVE_DIR}

fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --asr-dir ${ASR_DIR} \
  --tts-dir ${TTS_DIR} \
  --distributed-world-size 4 \
  --distributed-port 0 \
  --ddp-backend legacy_ddp \
  --user-dir ${USER_DIR} \
  --log-format json \
  --seed 1 \
  --fp16 \
  \
  --task artst \
  --t5-task multitask \
  --sample-rate 16000 \
  --num-workers 6 \
  --max-tokens 3200000 \
  --update-freq 6 \
  --bpe-tokenizer ${BPE_TOKENIZER} \
  \
  --criterion artst \
  --report-accuracy \
  --use-guided-attn-loss \
  --zero-infinity \
  --ce-weight 0.5 \
  --ctc-weight 0.5 \
  \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-08 \
  --weight-decay 0.1 \
  --lr 0.0001 \
  --lr-scheduler tri_stage \
  --phase-ratio "[0.25, 0.4, 0.35]" \
  --final-lr-scale 0.05 \
  --clip-norm 25 \
  --sentence-avg \
  \
  --max-update 400000 \
  --max-text-positions 600 \
  --min-speech-sample-size 1056 \
  --max-speech-sample-size 480256 \
  --max-speech-positions 1876 \
  --required-batch-size-multiple 1 \
  --save-interval-updates 10000 \
  --skip-invalid-size-inputs-valid-test \
  \
  --arch artst_transformer_base_asr \
  --share-input-output-embed \
  --find-unused-parameters \
  --bert-init \
  --relative-position-embedding \
  --validate-after-updates 150000 \
  --warmup-steps 30000 \
  --ignore-unused-valid-subsets \
  --validate-interval 100 \
  \
  --keep-last-epochs 3 \
  --feature-grad-mult 1.0 \
  --finetune-from-model ${CHECKPOINT_PATH} \
  --wandb-project ${WANDB_PROJECT}


