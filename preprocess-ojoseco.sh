#!/bin/bash
python preprocess_rupturas_test.py -i ../../UBOT/dataset/train+5videos/ -o ./datasets/ojo-seco-new/test/
python preprocess_rupturas_train.py -i ../../UBOT/dataset/ojo_seco_sin_ruptura/ -o ./datasets/ojo-seco-new/train/
mkdir ./datasets/ojo-seco-new/test/good
find ./datasets/ojo-seco-new/train/good/ -name "*.png" | shuf | head -n 109 | xargs -i mv {} ./datasets/ojo-seco-new/test/good/
rm -rf ./datasets/ojo-seco-new/test/mask
