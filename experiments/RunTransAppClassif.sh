#!/bin/bash
#SBATCH --partition=partition
#SBATCH --time=3-00:00:00
#SBATCH --mem=0
#SBATCH -N 1
#SBATCH -o ./job_outputs/TransAppClf.out
#SBATCH -e ./job_outputs/TransAppClf.err

. /projets/datascience_retd/dist/miniconda37/bin/activate pytorch-1.8
python3 RunTransAppClassif.py $1 $2 $3 $4
