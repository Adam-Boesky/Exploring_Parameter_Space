#!/bin/bash
#SBATCH --job-name=sw_run 				# job name
#SBATCH --nodes=1 					# Number of nodes
#SBATCH --ntasks=20 				# Number of cores
#SBATCH --output=/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/sw_run.out 				# output storage file
#SBATCH --error=/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/sw_run.err 					# error storage file
#SBATCH --time=100:00:00 					# Runtime (hr:min:second)
#SBATCH --mem=64G 					# Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH -p shared
#SBATCH --mail-user=aboesky@college.harvard.edu 				# Send email to user
#SBATCH --mail-type=FAIL			#
#
#Load Conda environment
module load gcc/12.1.0-fasrc01 gsl/2.6-fasrc01 hdf5/1.10.6-fasrc01
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
export CPP_INCLUDE_PATH=/n/home04/aboesky/pgk/boost/include
export LD_LIBRARY_PATH=/n/home04/aboesky/pgk/boost/lib:$LD_LIBRARY_PATH
export COMPAS_ROOT_DIR=/n/home04/aboesky/pgk/COMPAS
export CONFIG_FILEPATH=/n/home04/aboesky/berger/Exploring_Parameter_Space/Configuration_Files/kick_sigma_remnant_mass_config_files/sigma_30_remnant_mass_R.yaml
export OUTPUT_DIRNAME=/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/final_sigma_remnant_prescription/output_sigma_30_remnant_prescription_R/
export NUM_SYSTEMS=20000000
export NUM_CORES=20
export NUM_PER_CORE=1000000
export PARAM1_LABEL=sigma
export PARAM2_LABEL=remnant_prescription
export PARAM1_VAL=30
export PARAM2_VAL=R
#
# Run your job
python /n/home04/aboesky/berger/Exploring_Parameter_Space/Cluster/custom_sw_run.py
