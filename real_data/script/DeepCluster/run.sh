#!/bin/bash

DIR="~/imagenet_val"
ARCH="vgg16"
LR=0.05
WD=-5
K=1000
WORKERS=8
EXP="~/exp_val"

mkdir -p ${EXP}

python main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --sobel --verbose --workers ${WORKERS}
