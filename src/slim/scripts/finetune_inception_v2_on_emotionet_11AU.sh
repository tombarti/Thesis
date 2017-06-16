#!/bin/bash
#
# This script performs the following operations:
# 1. Fine-tunes an InceptionV1 model on the Flowers training set.
# 2. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/{name_of_script}.sh
set -e

# set model
MODEL=inception_v2

# select which dataset to train or eval on
DATASET=emotionet

# set optimiser
OPTIMISER=$2

# Where the pre-trained InceptionV1 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=~/Thesis/tmp/${MODEL}/checkpoint

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=~/Thesis/tmp/${MODEL}/train/${OPTIMISER}

# Where the evaluation checkpoint and logs will be saved to
EVAL_DIR=~/Thesis/tmp/${MODEL}/eval/${OPTIMISER}

# Where the dataset is saved to.
DATASET_DIR=~/Thesis/data/records

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir -p ${PRETRAINED_CHECKPOINT_DIR}
  echo "created directory" ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/${MODEL}.ckpt ]; then
  echo "Downloading ckpt file: vgg_16.ckpt"
  wget http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz
  tar -xvf inception_v2_2016_08_28.tar.gz
  mv ${MODEL}.ckpt ${PRETRAINED_CHECKPOINT_DIR}/${MODEL}.ckpt
  rm inception_v2_2016_08_28.tar.gz
fi

case $1 in
  train)
    if [ ! -d "$TRAIN_DIR" ]; then
      mkdir -p ${TRAIN_DIR}
      echo "created directory" ${TRAIN_DIR}
    fi
    # Fine-tune only the new layers for 2000 steps.
    python train_image_classifier.py \
      --train_dir=${TRAIN_DIR} \
      --dataset_name=${DATASET} \
      --dataset_split_name=train \
      --dataset_dir=${DATASET_DIR} \
      --model_name=${MODEL} \
      --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/${MODEL}.ckpt \
      --checkpoint_exclude_scopes=InceptionV2/Logits,InceptionV2/AuxLogits \
      --trainable_scopes=InceptionV2/Logits,InceptionV2/AuxLogits \
      --max_number_of_steps=5000 \
      --batch_size=32 \
      --learning_rate=0.01 \
      --save_interval_secs=600 \
      --save_summaries_secs=600 \
      --log_every_n_steps=100 \
      --optimizer=${OPTIMISER} \
      --weight_decay=0.00004
    ;;
  evaluate)
    if [ ! -d "$EVAL_DIR" ]; then
      mkdir -p ${EVAL_DIR}
      echo "created directory" ${EVAL_DIR}
    fi

    # Run evaluation.
    python eval_image_classifier.py \
      --checkpoint_path=${TRAIN_DIR} \
      --eval_dir=${EVAL_DIR} \
      --dataset_name=${DATASET} \
      --dataset_split_name=validation \
      --dataset_dir=${DATASET_DIR} \
      --model_name=${MODEL}
    ;;
esac

# Fine-tune all the new layers for 1000 steps.
#python train_image_classifier.py \
 # --train_dir=${TRAIN_DIR}/all \
 # --dataset_name=flowers \
 # --dataset_split_name=train \
 # --dataset_dir=${DATASET_DIR} \
 # --checkpoint_path=${TRAIN_DIR} \
 # --model_name=inception_v1 \
 # --max_number_of_steps=1000 \
 # --batch_size=32 \
 # --learning_rate=0.001 \
 # --save_interval_secs=60 \
 # --save_summaries_secs=60 \
 # --log_every_n_steps=100 \
 # --optimizer=rmsprop \
 # --weight_decay=0.00004

# Run evaluation.
#python eval_image_classifier.py \
 # --checkpoint_path=${TRAIN_DIR}/all \
 # --eval_dir=${TRAIN_DIR}/all \
 # --dataset_name=flowers \
 # --dataset_split_name=validation \
 # --dataset_dir=${DATASET_DIR} \
 # --model_name=inception_v1

