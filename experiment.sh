#!/bin/bash
#SBATCH --partition=ai2es
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --mem=8196
#SBATCH --output=/scratch/jroth/supercomputer/text_outputs/exp%01a_stdout_%A.txt
#SBATCH --error=/scratch/jroth/supercomputer/text_outputs/exp%01a_stderr_%A.txt
#SBATCH --time=48:00:00
#SBATCH --job-name=G1
#SBATCH --mail-user=jay.c.rothenberger@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/scratch/jroth/AI2ES-DL/
#SBATCH --nodelist=c732,c731
#SBATCH --array=[0]
. /home/fagg/tf_setup.sh
conda activate tf
python run.py --pkl experiments/experiment-16675944129479291.pkl --lscratch $LSCRATCH --id $SLURM_ARRAY_TASK_ID