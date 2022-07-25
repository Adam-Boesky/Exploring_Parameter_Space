#!/bin/bash
#SBATCH -c 48               # Number of cores (-c)
#SBATCH -t 0-14:00          # Runtime in D-HH:MM, minimum of 10 minutes
#SBATCH -p shared           # Partition to submit to
#SBATCH --mem=24000         # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o myoutput_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e myerrors_%j.err  # File to which STDERR will be written, %j inserts jobid
#SBATCH --test-only         # ONLY TEST THIS SCRIPT

module load perl/5.26.1-fasrc01 #Load Perl module
perl -e 'print "Hi there.\n"'