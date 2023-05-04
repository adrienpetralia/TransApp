#!/bin/bash
declare -a all_case=("cooker_case" "dishwasher_case" "desktopcomputer_case" "ecs_case" "pluginheater_case" "tumbledryer_case" "tv_greater21inch_case" "tv_less21inch_case" "laptopcomputer_case")
declare -a models=("TransApp" "TransAppPT")
declare -a dim_model=("64" "96" "128")
declare -a frac=("1")

for case in ${all_case[@]}; do
  for model in ${models[@]}; do
    for d_model in ${dim_model[@]}; do
      for f in ${frac[@]}; do
        sbatch TransAppClassif.sh $case $model $d_model $f
      done
    done
  done
done
