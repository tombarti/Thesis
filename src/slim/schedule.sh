#!/bin/bash

./scripts/finetune_inception_v2_on_emotionet.sh evaluate rmsprop
./scripts/finetune_inception_v2_on_emotionet.sh evaluate momentum
./scripts/finetune_inception_v2_on_emotionet.sh evaluate sgd
