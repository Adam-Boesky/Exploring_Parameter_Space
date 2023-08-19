import h5py as h5
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# test_file = h5.File('test.h5', 'w')

# test1 = test_file.create_group('test1')
# test2 = test_file.create_group('test2')

# test1.create_dataset('Redshifts', data=np.array([1,2,3,3,3,5,5,67,7,776,6,5,5,4]))

# test2.create_dataset('Redshifts', (100, 50))

# test_file.close()

# test_file = h5.File('test.h5', 'a')
# redshifts = test_file['test2']['Redshifts']
# print(redshifts[...])
# redshifts[2] = np.zeros((50))
# test_file.close()

# # file = h5.File('/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/final_sigma_remnant_prescription/output_sigma_30_remnant_prescription_D/COMPAS_Output_Weighted.h5')

# # print(file['BSE_Double_Compact_Objects']['SEED'])

# print(np.cumsum([[1,2,3], [4,5,6]]))

# test_arr = np.array([[1,2,3], [4,5,6]])

# print(len(test_arr.flatten()))

# test_file = h5.File('/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/final_alpha_CE_beta/output_alpha_CE_10.0_beta_0.75/Pickled_Rates/MSSFR_111/BHNS/rates_and_bootstraps.h5')
# # /n/home04/aboesky/berger/Exploring_Parameter_Space/Configuration_Files/test_config_folder/config_common_envelope_alpha_10.0_mass_transfer_fa_0.75.yaml
# print(test_file['Bootstraps']['total_formation_rates'][...])
# plt.plot(test_file['Actual']['redshifts'][...], test_file['Bootstraps']['total_formation_rates'][0][...])

# alphas = ['0.1', '0.5', '2.0', '10.0']
# betas = ['0.25', '0.5', '0.75']
# total_formation_rates = {}

# for alpha in alphas:
#     total_formation_rates[alpha] = {}
#     for beta in betas:
#         with open('/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/final_alpha_CE_beta/output_alpha_CE_' + alpha + '_beta_' + beta + '/Pickled_Rates/MSSFR_111/BHNS/rates_and_bootstraps.h5', 'rb') as f:
#             h5_data = h5.File(f)
#             total_formation_rates[alpha][beta] = h5_data['Actual']['total_formation_rates'][...]
#         print(total_formation_rates[alpha][beta])

        # test_file = h5.File('/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/final_alpha_CE_beta/output_alpha_CE_' + alpha + '_beta_' + beta + '/Pickled_Rates/MSSFR_111/BBH/rates_and_bootstraps.h5')
        # print(test_file['Bootstraps']['total_formation_rates'])
        # print(test_file['Actual'].keys())

# sigmas = ['30', '265', '750']
# remnant_mass_prescriptions = ['M', 'R', 'D']

# for sigma in sigmas:
#     for rem in remnant_mass_prescriptions:
#         with open('/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/final_sigma_remnant_prescription/output_sigma_'+sigma+'_remnant_prescription_'+rem+'/Pickled_Rates/MSSFR_111/BHNS/rates_and_bootstraps.h5', 'rb') as f:
#             h5_data = h5.File(f)
#             print(h5_data['Bootstraps']['total_formation_rates'])

with open('/n/holystore01/LABS/berger_lab/Users/aboesky/two_parameters/final_alpha_CE_beta/output_alpha_CE_10.0_beta_0.25/Pickled_Rates/MSSFR_111/BBH/all_rates_weights_at_redshifts.h5', 'rb') as f:
        h5_data = h5.File(f)
        print(h5_data['actual'].keys())
        print(np.shape(h5_data['actual']['total_formation_rates'][...]))

        plt.plot(np.sum(h5_data['actual']['total_formation_rates'][...], axis=0))
        plt.show()

