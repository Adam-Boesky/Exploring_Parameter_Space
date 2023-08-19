import os, sys

# Append paths to access the scripts we need
sys.path.append('/n/home04/aboesky/berger/Exploring_Parameter_Space/Data_Generation')

from pickle_rates import calcuate_rates_and_pickle


def get_env_params():

    # Get the output directory name
    output_dirname = os.environ.get('OUTPUT_DIRNAME')

    # Get the type of DCO that we want
    dco_type = os.environ.get('DCO_TYPE')

    # Get variable labels
    param1_label = os.environ.get('PARAM1_LABEL')               # the label of the first variable
    param2_label = os.environ.get('PARAM2_LABEL')               # the label of the second variable
    # Get variable values
    param1_val = os.environ.get('PARAM1_VAL')                   # the value of the first variable
    param2_val = os.environ.get('PARAM2_VAL')                   # the value of the second variable

    return output_dirname, dco_type, param1_label, param2_label, param1_val, param2_val


def main():
    # Get the enviornment parameters
    output_dirname, dco_type, param1_label, param2_label, param1_val, param2_val = get_env_params()

    # The path for the compas result file
    compas_results_path = output_dirname + 'COMPAS_Output_Weighted.h5'

    # Get the rates, boostraped errors, and pickle the results
    calcuate_rates_and_pickle(alpha_val=param1_val, beta_val=param2_val, dco_type=dco_type, mssfr_prescription='MSSFR_111', output_files_path=output_dirname, compas_results_path=compas_results_path)



if __name__ == '__main__':
    sys.exit(main())