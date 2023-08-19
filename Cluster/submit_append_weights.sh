#!/bin/bash
#SBATCH --job-name=append_weights 				# job name
#SBATCH --nodes=1 					# Number of nodes
#SBATCH --ntasks=10 				# Number of cores
#SBATCH --output=/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/append_weights.out 				# output storage file
#SBATCH --error=/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/append_weights.err 					# error storage file
#SBATCH --time=05:00:00 					# Runtime (hr:min:second)
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
export OUTPUT_DIRNAME=/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/final_sigma_remnant_prescription/output_sigma_30_remnant_prescription_R/
export PARAM1_LABEL=sigma
export PARAM2_LABEL=remnant_prescription
export PARAM1_VAL=30
export PARAM2_VAL=R
#
# Run your job
python /n/home04/aboesky/berger/Exploring_Parameter_Space/Cluster/append_weights_run.py 
