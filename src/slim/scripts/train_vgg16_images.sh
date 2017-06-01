#!/bin/bash
#
# This script performs the following operations:
# 1. Fine-tunes an InceptionV1 model on the Flowers training set.
# 2. Evaluates the model on the Flowers validation set.
#
# Usage:
# cd slim
# ./slim/scripts/finetune_inception_v1_on_flowers.sh
set -e

# Where the pre-trained InceptionV1 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=~/Thesis/data/checkpoints/vgg_16

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=~/Thesis/tmp/vgg16/train

# Where the dataset is saved to.
DATASET_DIR=~/Thesis/data/records

# Download the pre-trained checkpoint.
if [ ! -d "$PRETRAINED_CHECKPOINT_DIR" ]; then
  mkdir ${PRETRAINED_CHECKPOINT_DIR}
  echo "created directory" ${PRETRAINED_CHECKPOINT_DIR}
fi
if [ ! -f ${PRETRAINED_CHECKPOINT_DIR}/vgg_16.ckpt ]; then
  echo "Downloading ckpt file: vgg_16.ckpt"
  wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
  tar -xvf vgg_16_2016_08_28.tar.gz
  mv vgg_16.ckpt ${PRETRAINED_CHECKPOINT_DIR}/vgg_16.ckpt
  rm vgg_16_2016_08_28.tar.gz
fi

# Fine-tune only the new layers for 2000 steps.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=images \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=vgg_16 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/vgg_16.ckpt \
  --checkpoint_exclude_scopes=vgg_16/fc6,vgg_16/fc7,vgg_16/fc8 \
  --trainable_scopes=vgg_16/fc6,vgg_16/fc7,vgg_16/fc8 \
  --max_number_of_steps=100 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --save_interval_secs=600 \
  --save_summaries_secs=600 \
  --log_every_n_steps=100 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

# Run evaluation.
#python eval_image_classifier.py \
 # --checkpoint_path=${TRAIN_DIR} \
 # --eval_dir=${TRAIN_DIR} \
 # --dataset_name=flowers \
 # --dataset_split_name=validation \
 # --dataset_dir=${DATASET_DIR} \
 # --model_name=inception_v1

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
