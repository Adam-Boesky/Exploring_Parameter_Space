#!/bin/bash
#SBATCH --job-name=combine_h5s 				# job name
#SBATCH --nodes=1 					# Number of nodes
#SBATCH --ntasks=10 				# Number of cores
#SBATCH --output=/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/combine_h5s.out 				# output storage file
#SBATCH --error=/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/combine_h5s.err 					# error storage file
#SBATCH --time=07:00:00 					# Runtime (hr:min:second)
#SBATCH --mem=48G 					# Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH -p shared
#SBATCH --mail-user=aboesky@college.harvard.edu 				# Send email to user
#SBATCH --mail-type=FAIL			#
#
#Load Conda environment
module load Anaconda/5.0.1-fasrc02
source activate Exploring_Uncertainties
#
#Print some stuff on screen
echo $SLURM_JOB_ID
echo $SLURM_JOB_NAME
echo $SLURM_ARRAY_TASK_ID
#
#Set variables
export QT_QPA_PLATFORM=offscreen # To avoid the X Display error
export COMPAS_ROOT_DIR=/n/home04/aboesky/pgk/COMPAS
export OUTPUT_DIRNAME=/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/final_alpha_CE_beta/output_alpha_CE_0.1_beta_0.5/
export PARAM1_LABEL=alpha_CE
export PARAM2_LABEL=beta
export PARAM1_VAL=0.1
export PARAM2_VAL=0.5
#
# Run your job
python /n/home04/aboesky/berger/Exploring_Parameter_Space/Cluster/h5copy.py  /n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/final_alpha_CE_beta/output_alpha_CE_0.1_beta_0.5/ -r 2 -b 20 -o ../../COMPAS_Output.h5
