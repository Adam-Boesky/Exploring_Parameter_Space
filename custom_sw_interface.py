#!/usr/bin/env python

import os, sys
import pandas as pd
import shutil
import time
import numpy as np
from sqlalchemy import null
from stroopwafel import sw, classes, prior, sampler, distributions, constants, utils
import argparse
import h5py


# TODO fix issues with adaptive sampling
# TODO add in functionality for alternative runSubmit names and locations

#######################################################
### 
### For User Instructions, see 'docs/sampling.md'
### 
#######################################################

class SwInterface():
    
    def __init__(self, config_file, param, val, num_systems = 1000, num_per_core = 1000, num_cores = 1, random_seed_base = 0):

        ### Set default stroopwafel inputs - these are overwritten by any command-line arguments

        self.COMPAS_ROOT_DIR = os.environ.get('COMPAS_ROOT_DIR')
        self.compas_executable = os.path.join(self.COMPAS_ROOT_DIR, 'src/COMPAS')   # Location of the executable      # Note: overrides runSubmit + compasConfigDefault.yaml value
        self.config_file = config_file        # Stroopwafel configuration file
        self.num_systems = 10000                  # Number of binary systems to evolve                                             # Note: overrides runSubmit + compasConfigDefault.yaml value
        self.output_folder = 'output_' + param + '/output_' + val + '/'           # Location of output folder (relative to cwd)                                     # Note: overrides runSubmit + compasConfigDefault.yaml value
        self.random_seed_base = 0                # The initial random seed to increment from                                       # Note: overrides runSubmit + compasConfigDefault.yaml value

        self.num_cores = num_cores                       # Number of cores to parallelize over 
        self.num_per_core = 10000                  # Number of binaries per batch
        self.mc_only = False                      # Exclude adaptive importance sampling (currently not implemented, leave set to True)
        self.run_on_hpc = False                  # Run on slurm based cluster HPC

        self.output_filename = val + '_samples.csv'     # output filename for the stroopwafel samples
        self.debug = True                       # show COMPAS output/errors

        self.commandOptions =  null

    def create_dimensions(self):
        """
        This Function that will create all the dimensions for stroopwafel, a dimension is basically one of the variables you want to sample
        Invoke the Dimension class to create objects for each variable. Look at the Dimension class definition in classes.py for more.
        It takes the name of the dimension, its max and min value. 
        The Sampler class will tell how to sample this dimension. Similarly, prior tells it how it calculates the prior. You can find more of these in their respective modules
        OUT:
            As Output, this should return a list containing all the instances of Dimension class.
        """
        m1 = classes.Dimension('--initial-mass-1', 5, 50, sampler.kroupa, prior.kroupa)
        q = classes.Dimension('q', 0, 1, sampler.uniform, prior.uniform, should_print = False)
        a = classes.Dimension('-a', .01, 10, sampler.flat_in_log, prior.flat_in_log) # I think this is off?
        # kick_velocity_random_1 = classes.Dimension('Kick_Velocity_Random_1', 0, 1, sampler.uniform, prior.uniform)
        # kick_theta_1 = classes.Dimension('Kick_Theta_1', -np.pi / 2, np.pi / 2, sampler.uniform_in_cosine, prior.uniform_in_cosine)
        #kick_phi_1 = classes.Dimension('Kick_Phi_1', 0, 2 * np.pi, sampler.uniform, prior.uniform)
        #kick_velocity_random_2 = classes.Dimension('Kick_Velocity_Random_2', 0, 1, sampler.uniform, prior.uniform)
        #kick_theta_2 = classes.Dimension('Kick_Theta_2', -np.pi / 2, np.pi / 2, sampler.uniform_in_cosine, prior.uniform_in_cosine)
        #kick_phi_2 = classes.Dimension('Kick_Phi_2', 0, 2 * np.pi, sampler.uniform, prior.uniform)
        #return [m1, q, a, kick_velocity_random_1, kick_theta_1, kick_phi_1, kick_velocity_random_2, kick_theta_2, kick_phi_2]
        return [m1, q, a]

    def update_properties(self, locations, dimensions):
        """
        This function is not mandatory, it is required only if you have some dependent variable. 
        For example, if you want to sample Mass_1 and q, then Mass_2 is a dependent variable which is product of the two.
        Similarly, you can assume that Metallicity_2 will always be equal to Metallicity_1
        IN:
            locations (list(Location)) : A list containing objects of Location class in classes.py. 
            You can play with them and update whatever fields you like or add more in the property (which is a dictionary)
        OUT: Not Required
        """
        m1 = dimensions[0]
        q = dimensions[1]
        for location in locations:
            location.properties['--initial-mass-2'] = location.dimensions[m1] * location.dimensions[q]
            location.properties['--metallicity'] = location.properties['--metallicity'] = constants.METALLICITY_SOL
            location.properties['--eccentricity'] = 0
            #location.properties['Kick_Mean_Anomaly_1'] = np.random.uniform(0, 2 * np.pi, 1)[0]
            #location.properties['Kick_Mean_Anomaly_2'] = np.random.uniform(0, 2 * np.pi, 1)[0]





    #################################################################################
    #################################################################################
    ###                                                                           ###
    ###         USER SHOULD NOT SET ANYTHING BELOW THIS LINE                      ###
    ###                                                                           ###
    #################################################################################
    #################################################################################





    def configure_code_run(self, batch):
        """
        This function tells stroopwafel what program to run, along with its arguments.
        IN:
            batch(dict): This is a dictionary which stores some information about one of the runs. It has an number key which stores the unique id of the run
                It also has a subprocess which will run under the key process. Rest, it depends on the user. User is free to store any information they might need later 
                for each batch run in this dictionary. For example, here I have stored the 'output_container' and 'grid_filename' so that I can read them during discovery of interesting systems below
        OUT:
            compas_args (list(String)) : This defines what will run. It should point to the executable file along with the arguments.
            Additionally one must also store the grid_filename in the batch so that the grid file is created
        """
        batch_num = batch['number']
        grid_filename = os.path.join(self.output_folder, 'grid_' + str(batch_num) + '.csv')
        output_container = 'batch_' + str(batch_num)
        random_seed = self.random_seed_base + batch_num*self.num_per_core  # ensure that random numbers are not reused across batches
        compas_args = [self.compas_executable, '--grid', grid_filename, '--output-container', output_container, '--random-seed' , random_seed]
        [compas_args.extend([key, val]) for key, val in self.commandOptions.items()] 
        for params in self.extra_params:
            compas_args.extend(params.split("="))
        batch['grid_filename'] = grid_filename
        batch['output_container'] = output_container
        return compas_args

    def interesting_systems(self, batch):
        """
        This is a mandatory function, it tells stroopwafel what an interesting system is. User is free to define whatever looks interesting to them.
        IN:
            batch (dict): As input you will be given the current batch which just finished its execution. You can take in all the keys you defined in the configure_code_run method above
        OUT:
            Number of interesting systems
            In the below example, I define all the NSs as interesting, so I read the files, get the SEED from the system_params file and define the key is_hit in the end for all interesting systems 
        """
        try:
            folder = os.path.join(self.output_folder, batch['output_container'])
            f = h5py.File(folder + '/batch_' + str(batch['number']) + '.h5','r') # Open the output h5 file
            system_parameters = f['BSE_System_Parameters']
            system_parameters_df = pd.DataFrame(columns=system_parameters.keys())
            for group in system_parameters_df.keys():
                system_parameters_df[group] = system_parameters[group]
            system_parameters_df.rename(columns = lambda x: x.strip(), inplace = True)
            # Get all seeds, label samples' seed, and set is_hit to 0
            seeds = system_parameters_df['SEED']
            for index, sample in enumerate(batch['samples']):
                seed = seeds[index]
                sample.properties['SEED'] = seed
                sample.properties['is_hit'] = 0
                sample.properties['batch'] = batch['number']
            dcos = f['BSE_Double_Compact_Objects']
            dco_df = pd.DataFrame(columns=dcos.keys())
            for group in dcos.keys():
                dco_df[group] = dcos[group]
            dco_df.rename(columns = lambda x: x.strip(), inplace = True)
            # Generally, this is the line you would want to change. This line dictates what a hit is
            dns = dco_df[np.logical_and(dco_df['Merges_Hubble_Time'] == 1, \
                np.logical_and(dco_df['Stellar_Type(1)'] == 14, dco_df['Stellar_Type(2)'] == 14))]
            interesting_systems_seeds = set(dns['SEED']) # Get the seeds of all the hits
            # Turn is_hit to 1 for all the hits
            for sample in batch['samples']: 
                if sample.properties['SEED'] in interesting_systems_seeds:
                    sample.properties['is_hit'] = 1
            return len(dns)
        except IOError as error:
            return 0

    def selection_effects(self, sw):
        """
        This is not a mandatory function, it was written to support selection effects
        Fills in selection effects for each of the distributions
        IN:
            sw (Stroopwafel) : Stroopwafel object
        """
        if hasattr(sw, 'adapted_distributions'):
            biased_masses = []
            rows = []
            for distribution in sw.adapted_distributions:
                folder = os.path.join(self.output_folder, 'batch_' + str(int(distribution.mean.properties['batch'])))
                f = h5py.File(folder + '/batch_' + str(batch['number']) + '.h5','r') # Open the output h5 file
                dcos = f['BSE_Double_Compact_Objects'] # Get the BSE objects
                dco_df = pd.DataFrame(columns=dcos.keys())
                for group in dcos.keys():
                    dco_df[group] = dcos[group]
                dco_df.rename(columns = lambda x: x.strip(), inplace = True)
                row = dco_df.loc[dco_df['SEED'] == distribution.mean.properties['SEED']]
                rows.append([row.iloc[0]['Mass_1'], row.iloc[0]['Mass_2']])
                biased_masses.append(np.power(max(rows[-1]), 2.2))
            # update the weights
            mean = np.mean(biased_masses)
            for index, distribution in enumerate(sw.adapted_distributions):
                distribution.biased_weight = np.power(max(rows[index]), 2.2) / mean

    def rejected_systems(self, locations, dimensions):
        """
        This method takes a list of locations and marks the systems which can be
        rejected by the prior distribution
        IN:
            locations (List(Location)): list of location to inspect and mark rejected
        OUT:
            num_rejected (int): number of systems which can be rejected
        """
        m1 = dimensions[0]
        q = dimensions[1]
        a = dimensions[2]
        mass_1 = [location.dimensions[m1] for location in locations]
        mass_2 = [location.properties['--initial-mass-2'] for location in locations]
        metallicity = [location.properties['--metallicity'] for location in locations]
        eccentricity = [location.properties['--eccentricity'] for location in locations]
        num_rejected = 0
        for index, location in enumerate(locations):
            radius_1 = utils.get_zams_radius(mass_1[index], metallicity[index])
            radius_2 = utils.get_zams_radius(mass_2[index], metallicity[index])
            star_to_roche_lobe_radius_ratio_1 = radius_1 / (location.dimensions[a] * (1 - eccentricity[index]) * utils.calculate_roche_lobe_radius(mass_1[index], mass_2[index]))
            star_to_roche_lobe_radius_ratio_2 = radius_2 / (location.dimensions[a] * (1 - eccentricity[index]) * utils.calculate_roche_lobe_radius(mass_2[index], mass_1[index]))
            location.properties['is_rejected'] = 0
            if (mass_2[index] < constants.MINIMUM_SECONDARY_MASS) or (location.dimensions[a] <= (radius_1 + radius_2)) \
            or star_to_roche_lobe_radius_ratio_1 > 1 or star_to_roche_lobe_radius_ratio_2 > 1:
                location.properties['is_rejected'] = 1
                num_rejected += 1
        return num_rejected

    def run_sw(self):
        # STEP 1 : Import and assign input parameters for stroopwafel 
        parser=argparse.ArgumentParser()
        parser.add_argument('--num_systems', help = 'Total number of systems', type = int, default = self.num_systems)  
        parser.add_argument('--num_cores', help = 'Number of cores to run in parallel', type = int, default = self.num_cores)
        parser.add_argument('--num_per_core', help = 'Number of systems to generate in one core', type = int, default = self.num_per_core)
        parser.add_argument('--debug', help = 'If debug of COMPAS is to be printed', type = bool, default = self.debug)
        parser.add_argument('--mc_only', help = 'If run in MC simulation mode only', type = bool, default = self.mc_only)
        parser.add_argument('--run_on_hpc', help = 'If we are running on a (slurm-based) HPC', type = bool, default = self.run_on_hpc)
        parser.add_argument('--output_filename', help = 'Output filename', default = self.output_filename)
        parser.add_argument('--output_folder', help = 'Output folder name', default = self.output_folder)
        namespace, self.extra_params = parser.parse_known_args()

        start_time = time.time()
        #Define the parameters to the constructor of stroopwafel
        TOTAL_NUM_SYSTEMS = namespace.num_systems #total number of systems you want in the end
        NUM_CPU_CORES = namespace.num_cores #Number of cpu cores you want to run in parellel
        NUM_SYSTEMS_PER_RUN = namespace.num_per_core #Number of systems generated by each of run on each cpu core
        debug = namespace.debug #If True, will print the logs given by the external program (like COMPAS)
        run_on_hpc = namespace.run_on_hpc #If True, it will run on a clustered system helios, rather than your pc
        mc_only = namespace.mc_only # If you dont want to do the refinement phase and just do random mc exploration
        output_filename = namespace.output_filename #The name of the output file
        output_folder = os.path.join(os.getcwd(), namespace.output_folder)

        # Set commandOptions defaults - these are Compas option arguments
        self.commandOptions = dict()
        self.commandOptions.update({'--output-path' : self.output_folder}) 
        # commandOptions.update({'--logfile-delimiter' : 'COMMA'})  # overriden if there is a runSubmit + compas ConfigDefault.yaml

        # Over-ride with runSubmit + compasConfigDefault.yaml parameters, if given config is not valid
        sys.path.append(os.path.dirname(self.COMPAS_ROOT_DIR + '/utils/preProcessing/'))
        try:
            from runSubmit import pythonProgramOptions
            programOptions = pythonProgramOptions(config_file=self.config_file)   # Call the programoption class from runSubmit
            pySubOptions   = programOptions.command   # Get the dict from pythonProgramOptions

            # Continue to work from the dict, by edditing SW related options
            # Remove extraneous options
            pySubOptions.pop('compas_executable', None)
            pySubOptions.pop('--grid', None)
            pySubOptions.pop('--output-container', None)
            pySubOptions.pop('--number-of-binaries', None)
            pySubOptions.pop('--output-path', None)
            pySubOptions.pop('--random-seed', None)

            self.commandOptions.update(pySubOptions)

        except:
            print("Invalid runSubmit + compas ConfigDefault.yaml file, using default stroopwafel options")        

        print("Output folder is: ", self.output_folder)
        if os.path.exists(self.output_folder):
            command = input ("The output folder already exists. If you continue, I will remove all its content. Press (Y/N)\n")
            if (command == 'Y'):
                shutil.rmtree(self.output_folder)
            else:
                exit()
        os.makedirs(self.output_folder)


        # STEP 2 : Create an instance of the Stroopwafel class
        sw_object = sw.Stroopwafel(TOTAL_NUM_SYSTEMS, NUM_CPU_CORES, NUM_SYSTEMS_PER_RUN, self.output_folder, self.output_filename, debug = self.debug, run_on_helios = self.run_on_hpc, mc_only = self.mc_only)


        # STEP 3: Initialize the stroopwafel object with the user defined functions and create dimensions and initial distribution
        dimensions = self.create_dimensions()
        sw_object.initialize(dimensions, self.interesting_systems, self.configure_code_run, self.rejected_systems, update_properties_method = self.update_properties)


        intial_pdf = distributions.InitialDistribution(dimensions)
        # STEP 4: Run the 4 phases of stroopwafel
        sw_object.explore(intial_pdf) #Pass in the initial distribution for exploration phase
        sw_object.adapt(n_dimensional_distribution_type = distributions.Gaussian) #Adaptaion phase, tell stroopwafel what kind of distribution you would like to create instrumental distributions
        ## Do selection effects
        self.selection_effects(sw)
        sw_object.refine() #Stroopwafel will draw samples from the adapted distributions
        sw_object.postprocess(distributions.Gaussian, only_hits = False) #Run it to create weights, if you want only hits in the output, then make only_hits = True

        end_time = time.time()
        print ("Total running time = %d seconds" %(end_time - start_time))

def main():
    # ce_alpha_interface = SwInterface('/Users/adamboesky/Research/PRISE/exploring_parameter_space/common_envelope_alpha_config_files/config_common_envelope_alpha_0.1.yaml', 'common_envelope_alpha_0.1')
    # ce_alpha_interface.run_sw()
    for filename in os.listdir('/Users/adamboesky/Research/PRISE/exploring_parameter_space/common_envelope_alpha_config_files'):
        val = filename[:-5][-4:].replace('_','')
        ce_alpha_interface = SwInterface('common_envelope_alpha_config_files/' + filename, 'common_envelope_alpha', val, num_systems=1000000, num_per_core=1000000)
        ce_alpha_interface.run_sw()

if __name__ == '__main__':
    sys.exit(main())