#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --cpus-per-task=16
#SBATCH --ntasks=1
#SBATCH --mem=16384
#SBATCH --output=/scratch/jroth/supercomputer/text_outputs/exp%01a_stdout_%A.txt
#SBATCH --error=/scratch/jroth/supercomputer/text_outputs/exp%01a_stderr_%A.txt
#SBATCH --time=96:00:00
#SBATCH --job-name=hparam
#SBATCH --mail-user=jay.c.rothenberger@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/jroth/AI2ES-DL/
#SBATCH --nodelist=c830
#SBATCH --array=[0-48%4]
. /home/fagg/tf_setup.sh
conda activate tf
python run.py --pkl experiments/experiment-1672696707554358.pkl --lscratch $LSCRATCH --id $SLURM_ARRAY_TASK_ID