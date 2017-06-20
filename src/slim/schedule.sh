#!/bin/bash

./scripts/finetune_vgg_16_on_emotionet_11AU.sh train rmsprop 32 0.1
./scripts/finetune_vgg_16_on_emotionet_11AU.sh train rmsprop 32 0.01
./scripts/finetune_vgg_16_on_emotionet_11AU.sh train rmsprop 32 0.001
