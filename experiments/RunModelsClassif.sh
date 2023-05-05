#!/bin/bash
#SBATCH --partition=partition
#SBATCH --time=3-00:00:00
#SBATCH --mem=0
#SBATCH -N 1
#SBATCH -o ./job_outputs/ModelsComparaison.out
#SBATCH -e ./job_outputs/ModelsComparaison.err

. /projets/datascience_retd/dist/miniconda37/bin/activate pytorch-1.8
python3 RunModelsClassif.py $1 $2
