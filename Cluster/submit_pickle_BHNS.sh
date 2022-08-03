#!/bin/bash
#SBATCH -c 96               # Number of cores (-c)
#SBATCH -t 0-15:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared             # Partition to submit to
#SBATCH --mem=184G          # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
module load Anaconda/5.0.1-fasrc02

source activate Exploring_Uncertainties

python pickle_BHNS.py