#!/bin/bash

#SBATCH --job-name=t5_new_large            # Job name
#SBATCH --output=logs/t5_new_large.%A_%a.txt   # Standard output and error log
#SBATCH --nodes=1                   # Run all processes on a single node
#SBATCH --gres=gpu:2          # Number of GPUs (per node)
#SBATCH -p it-hpc                      # Use the it-dept partition
#SBATCH --time=12:00:00             # Specify the time needed for your experiment

export WANDB_NAME=t5-new-large
DATA_PATH=/fsx/hyperpod-input-datasets/AROA6GBMFKRI2VWQAUGYI:Hawau.Toyin@mbzuai.ac.ae
OUTPUT_ROOT=/fsx/hyperpod-input-datasets/AROA6GBMFKRI2VWQAUGYI:Hawau.Toyin@mbzuai.ac.ae/results
DATA_ROOT=$DATA_PATH/sttatts_en
ASR_DIR=$DATA_PATH/librispeech_cleaned_manifest
TTS_DIR=$DATA_PATH/libritts_cleaned_manifest
SAVE_DIR=$OUTPUT_ROOT/sttatts/models_english_150K_t5_new_large/
TRAIN_SET="train-100-360-500|train-100-360-500"
VALID_SET="dev-clean|dev-clean"
BPE_TOKENIZER=$DATA_PATH/sttatts_en/spm_char.model
USER_DIR=$DATA_PATH/t5/artst
CHECKPOINT_PATH=$DATA_PATH/sttatts_en/speecht5_base.pt
WANDB_PROJECT=joint_training_en
 
hostname 
 
mkdir -p ${SAVE_DIR}

fairseq-train ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --asr-dir ${ASR_DIR} \
  --tts-dir ${TTS_DIR} \
  --distributed-world-size 2 \
  --distributed-port 0 \
  --ddp-backend legacy_ddp \
  --user-dir ${USER_DIR} \
  --log-format json \
  --batch-ratio "[0.5,0.5]" \
  --sample-ratios "[0.5,0.5]" \
  --seed 1 \
  --fp16 \
  \
  --task artst \
  --t5-task multitask \
  --sample-rate 16000 \
  --num-workers 4 \
  --max-tokens 3200000 \
  --update-freq 4 \
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
  --max-update 160000 \
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
  --validate-after-updates 10000 \
  --warmup-steps 10000 \
  --ignore-unused-valid-subsets \
  --validate-interval 50 \
  \
  --keep-last-epochs 3 \
  --feature-grad-mult 1.0 \
  --finetune-from-model ${CHECKPOINT_PATH} \
  --wandb-project ${WANDB_PROJECT}


# change from 150K updates to 160K




