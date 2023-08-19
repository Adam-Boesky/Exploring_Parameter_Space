import os
import sys

# Append paths to access the scripts we need
sys.path.append('/n/home04/aboesky/berger/Exploring_Parameter_Space/Data_Generation')

from append_weights import create_weighted_file


def get_env_params():

    # Get the output directory name
    output_dirname = os.environ.get('OUTPUT_DIRNAME')

    return output_dirname
    


def main():
    
    # Get the environment variables just declared by the sbatch submission script
    output_dirname = get_env_params()

    # Combine the h5s
    print('**** Writing ' + 'COMPAS_Output_Weighted.h5' + ' in ' + output_dirname)
    ce_alpha_interface = create_weighted_file(data_dir=output_dirname, Stroopwafel_name='samples.csv', Raw_COMPAS_name='COMPAS_Output.h5', Weights_COMPAS_name='COMPAS_Output_Weighted.h5')
    print('Done writing ' + 'COMPAS_Output_Weighted.h5' + ' in ' + output_dirname + ' !!!')

if __name__ == '__main__':
    sys.exit(main())