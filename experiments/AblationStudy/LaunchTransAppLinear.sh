#!/bin/bash
declare -a dataset=("CER" "EDF1")

for d in ${dataset[@]}; do
  sbatch TransAppLinear.sh $d
done