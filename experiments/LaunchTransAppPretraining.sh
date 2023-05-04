#!/bin/bash
declare -a type_embed=("0" "1")
declare -a dim_model=("64" "96" "128")

for type in ${type_embed[@]}; do
  for d in ${dim_model[@]}; do
    sbatch TransAppPretraining.sh $type $d
  done
done
