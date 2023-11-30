#!/bin/bash
#
#SBATCH -p shared
#SBATCH -n 1
#SBATCH --mem-per-cpu=5G
#SBATCH -t 1-00:00 # time (D-HH:MM)
#SBATCH -o /n/home04/aboesky/berger/Exploring_Parameter_Space/Uploading_Data/logs/myoutput_\%j.out
#SBATCH -e /n/home04/aboesky/berger/Exploring_Parameter_Space/Uploading_Data/logs/myerrors_\%j.err

cd /n/home04/aboesky/berger/Exploring_Parameter_Space/Uploading_Data

echo 'Running!'
python3 run_uploader.py $SIGMA $RMP $FPATH
echo 'Done running!'
