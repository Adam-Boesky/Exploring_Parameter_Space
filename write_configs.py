import os
import sys
import yaml
import shutil
import numpy as np

COMPAS_ROOT_DIR = os.getenv('COMPAS_ROOT_DIR')
sys.path.append(os.path.dirname(COMPAS_ROOT_DIR))


class ConfigWriter:
    """
    This class writes yaml files to make compas configuration for a number of varied parameters easier

    ...

    Attributes
    ----------
    config : dict
        a dictionary whose keys are the parameters and whose values are the parameter value of the configuration file

    param : str
        the parameter that we wish to vary

    choice_type : str
        the ChoiceType section that the parameter will be under in our configuration file

    Methods
    -------
    write_configs(param, vals, filepath='/Users/adamboesky/Research/PRISE/exploring_parameter_space')
        writes a number of configuration files with the varied parameters in a folder in the given file path
    """

    def __init__(self, param, choice_type='numericalChoices'):

        self.param = param
        self.param_name = param[2:].replace("-", "_")
        self.choice_type = choice_type
        # Get the default config file from the compas directory
        with open(COMPAS_ROOT_DIR + "/utils/preProcessing/compasConfigDefault.yaml", "r") as default_yaml:
            self.config = yaml.load(default_yaml, Loader=yaml.FullLoader)


    def write_configs(self, vals, path='/Users/adamboesky/Research/PRISE/exploring_parameter_space'):
        
        # Create folder to dump yaml files into
        dir_path = os.path.join(path, self.param_name + '_config_files')
        if os.path.exists(dir_path) and os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
            os.mkdir(dir_path)
        else:
            os.mkdir(dir_path)

        # Vary over each parameter
        for val in list(vals):
            self.config[self.choice_type][self.param] = float(val)

            # Dump the config files into the folder
            if isinstance(val, str):
                with open(dir_path + '/config_' + self.param_name + '_' + val + '.yaml', 'w') as file:
                    yaml.dump(self.config, file)
            else:
                with open(dir_path + '/config_' + self.param_name + '_' + str(val)[:4] + '.yaml', 'w') as file:
                    yaml.dump(self.config, file)



def main():
    alpha_configs = ConfigWriter('--common-envelope-alpha')
    alpha_configs.write_configs(np.linspace(0.1, 10, num=10))



if __name__ == '__main__':
    sys.exit(main())
