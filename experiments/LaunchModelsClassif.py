#!/bin/bash
#declare -a all_case=("cooker_case" "dishwasher_case" "desktopcomputer_case" "ecs_case" "pluginheater_case" "tumbledryer_case" "tv_greater21inch_case" "tv_less21inch_case" "laptopcomputer_case")
#declare -a models=("Inception" "FCN" "ResNet" "ResNetAtt" "FrameworkResNetAtt")

for case in ${all_case[@]}; do
  for model in ${models[@]}; do
    sbatch RunModelsClassif.sh $case $model
  done
done
