#!/bin/bash
declare -a model=("FCN" "ResNet" "Inception" "FrameworkResNetAtt")
declare -a model=("FrameworkResNetAtt")
declare -a InFramework=("1")

for m in ${model[@]}; do
  for in in ${InFramework[@]}; do
    sbatch InferenceTimeDeep.sh $m $in
  done
done