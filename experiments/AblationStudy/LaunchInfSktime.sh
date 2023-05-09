#!/bin/bash
declare -a model=("Arsenal" "Rocket")
declare -a InFramework=("0" "1")

for m in ${model[@]}; do
  for in in ${InFramework[@]}; do
    sbatch InferenceTimeSktime.sh $m $in
  done
done