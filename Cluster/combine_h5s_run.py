import os
import sys

# Append paths to access the scripts we need
sys.path.append('/n/home04/aboesky/berger/Exploring_Parameter_Space/Data_Generation')

from combine_h5s import condense_h5s


def get_env_params():

    # Get the output directory name
    output_dirname = os.environ.get('OUTPUT_DIRNAME')

    return output_dirname
    


def main():
    
    # Get the environment variables just declared by the sbatch submission script
    output_dirname = get_env_params()

    # Combine the h5s
    print('**** Writing complete.h5 in ' + output_dirname)
    ce_alpha_interface = condense_h5s(output_dir=output_dirname)
    print('Finished condensing!')

if __name__ == '__main__':
    sys.exit(main())