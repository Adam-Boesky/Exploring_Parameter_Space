# File to sort the 
import numpy as np 
import h5py as h5
from astropy.table import Table
import time

##################################################################
##    Should be Changed by user ##
Dir      = '../../output/'
SW_name  = 'samples.csv'
Raw_name = 'COMPAS_Output.h5'
New_name = 'COMPAS_Output_wWeights.h5'

##################################################################

##################################################################
def check_discrepancies(File, samples):
    """takes the COMPAS hdf5 output and stroopwafel samples.csv file
    Check if any of the SEEDs in the system mismatch, 
    if not return a bool to point from SystemParams to DCO
    """
    SYS_seeds    = File['BSE_System_Parameters']['SEED'][()]
    #Assuming your Random SEEDs are drawn in increasing order from some starting seed (min(SYS_seeds)),
    #we can use them as indices
    SYS_seeds_i  = SYS_seeds - min(SYS_seeds) 
    
    discrepancies = np.flatnonzero(SYS_seeds - samples['SEED'][SYS_seeds_i])
    if len(discrepancies) > 0:
        File.close()
        raise ValueError("sorted SEEDs dont match up!! discrepancy indices:", discrepancies)
    else:
        return 1



##################################################################
### 
### Main
###
##################################################################
def create_weighted_file(data_dir = None, Stroopwafel_name = None, Raw_COMPAS_name = None, Weights_COMPAS_name = None):
    """
    data_dir:           Where do you want to read and write data
    Stroopwafel_name:   Namve of stroopwafel output file that contains at least SEEDS and mixture_weights
    Raw_COMPAS_name:    Name of compas.h5 file, as output by postporcessing.py (should contain at least 'BSE_System_Parameters', 'BSE_Double_Compact_Objects'
    Weights_COMPAS_name Name of your new COMPAS file
    """
    start = time.time()

    #################################
    #Open the raw/original hdf5 file
    print(data_dir, Raw_COMPAS_name,)
    File      = h5.File(data_dir + Raw_COMPAS_name,'r')
    #Create new hdf5 file
    h_new     = h5.File(data_dir + Weights_COMPAS_name, 'w')

    start_copy = time.time()
    #################################
    # copy over the rest of the data 
    for group in list(File.keys()):#['BSE_System_Parameters', 'BSE_Double_Compact_Objects']:
        print('Copy sort and check ', group)
        #Lieke: this matches old group naming (saves me the trouble of renamnig stuff in my notebooks)
        group_name = group#[4:].replace('_', '')
        h_new.create_group(group_name) 

        #################################
        #Get data from the raw/original table 
        OldTable       = File[group]#
        

        # Run_Details does not contain SEEDs
        if group == 'Run_Details':
            indexing = ()#np.full(len(OldTable['SEED'][()]), True)
        else:
            #We would like our resulting data to be sorted by SEED (makes life easier
            try:
                indexing = np.argsort(OldTable['SEED'][()])
            except:
                indexing = np.argsort(OldTable['SEED>MT'][()])
        
        ################################
        #Create a sorted by SEED data set for each column in OldTable
        for column in OldTable.keys():
            unit       = OldTable[column].attrs['units']
            columnOld  = OldTable[column][()]
            data       = columnOld[indexing]
            dataNew    = h_new[group_name].create_dataset(column, data=data)
            dataNew.attrs['units'] = unit
            #add attribute comment
            
    #Always close your files again
    h_new.close()
    File.close()        
    end_copy = time.time()
    print('copy time: ', end_copy - start_copy, 's')
    
        
    read_sample_start = time.time()
    # Open the samples.csv file from stroopwafel
    samples = Table.read(data_dir+'/'+Stroopwafel_name, format = 'csv')
    read_sample_end = time.time()
    print('reading sample file time: ', read_sample_end -  read_sample_start , 's')

    #Open new hdf5 file
    h_new     = h5.File(data_dir +'/'+ Weights_COMPAS_name, 'r+')

    # Check for discrerpancies and return a pointer from SYS to DCO
    check_discrepancies(h_new, samples)


    mix_start = time.time()
    #################################
    # Add on the mixture weights 
    #(we only add these to'BSE_System_Parameters and BSE_Double_Compact_Objects)
    for group in ['BSE_System_Parameters', 'BSE_Double_Compact_Objects']:
        group_name = group#[4:].replace('_', '')
        UpdatedTable       = h_new[group_name]#
        print('Adding weights to ', group)
        #################################
        #Add mixture Weights ! 
        if group == 'BSE_System_Parameters':
            #Were going to get the weights from the samples file 
            # Sort 'samples' by SEED
            sample_indexing = np.argsort(samples['SEED'])
            # UpdatedTable was already sorted by SEED, so they should now be the same:
            sorted_sample_seeds = samples['SEED'][sample_indexing]
            discrepancies = np.flatnonzero(sorted_sample_seeds - UpdatedTable['SEED'][()])
            if len(discrepancies) > 0:
                print(discrepancies)
                h_new.close()
                File.close()
                raise ValueError("sorted SEEDs dont match up!! discrepancy indices:", discrepancies)
            data        = samples['mixture_weight'][sample_indexing] #get weights from samples ordered by SEED

        elif group == 'BSE_Double_Compact_Objects':
            #make sure to point from SYS to DCO
            #and take the data from your'BSE_System_Parameters (that you shouldve started with!)
            SYS_to_DCO = np.in1d(h_new['BSE_System_Parameters']['SEED'][()], h_new['BSE_Double_Compact_Objects']['SEED']) #Bool to point SYS to DCO
            data        = h_new['BSE_System_Parameters']['mixture_weight'][SYS_to_DCO]#re-order samples.cvs

        # add weights to new hdf5 file
        dataNew     = h_new[group_name].create_dataset('mixture_weight', data=data)

    #Always close your files again
    h_new.close()
    File.close()

    end = time.time()
    print('adding mix weight time: ', end - mix_start)

    print('Done :) your new files are here: ', data_dir +'/'+Weights_COMPAS_name )
    print('total time: ', end-start, 's' )