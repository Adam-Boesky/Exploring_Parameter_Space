#!/bin/bash
#
#SBATCH -p shared
#SBATCH -n 1
#SBATCH --mem-per-cpu=5G
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -o /n/home04/aboesky/berger/Exploring_Parameter_Space/Uploading_Data/logs/myoutput_\%j.out
#SBATCH -e /n/home04/aboesky/berger/Exploring_Parameter_Space/Uploading_Data/logs/myerrors_\%j.err

cd /Users/adamboesky/Research/PRISE/exploring_parameter_space/Uploading_Data

echo 'Running!'
python3 run_uploader.py $ALPHA $BETA $FPATH
echo 'Done running!'
