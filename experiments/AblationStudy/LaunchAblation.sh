#!/bin/bash
declare -a ablation=("att" "fixed" "learnable")
declare -a dataset=("CER" "EDF1")

for a in ${ablation[@]}; do
  for d in ${dataset[@]}; do
    sbatch AblationStudy.sh $a $d
  done
done