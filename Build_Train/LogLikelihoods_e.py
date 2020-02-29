####################################################################################################################
####################################################################################################################
# Create summary table for likelihoods for all hyperparameter sets.
# Author: Haiying Kong
# Last Modified: 16 December 2019
####################################################################################################################
####################################################################################################################
import tensorflow as tf
import gc
import numpy as np
from collections import OrderedDict
import pickle
import os
import os.path
from os import path

####################################################################################################################
# Collect garbage.
gc.collect()

####################################################################################################################
####################################################################################################################
# Set parameters and folder names.
thresh = 3
feat = 'e'
dir_name = '/icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Lock/WGS/Kipoi/MMSplice/Aggregated_byGene/Max/byFeature/Survival/Tumor' + str(thresh) + '/'
data_dir = dir_name + 'pkl/'
data_file = data_dir + 'All_' + feat + '_0.pkl'
dir_name = dir_name + 'Features_' + feat + '/'
HParams_dir = dir_name + 'HParams/'
models_dir = dir_name + 'Models/'

####################################################################################################################
####################################################################################################################
# Load Params_0, and dataset.
####################################################################################################################
pkl_file = open(data_dir + 'Params_' + feat + '.pkl', 'rb')
Params_0 = pickle.load(pkl_file)
pkl_file.close()

# Define parameters.
Params = {'feature_names': Params_0['feature_names'],
          'n_features': Params_0['n_features'],
          'n_genes': Params_0['n_genes'],
	  'n_times': 60,
	  'time_stepsize': 2,
          'dense_n_nodes': [[128], [64], [32], [16]],
          'regularizations': ['l1', 'l2'],
          'reg_lambdas': [0.1, 0.3, 0.5],
          'dense_dropout_rates': [[0.1], [0.3], [0.5]],
          'lstm_dropout_rates': [0.1, 0.3, 0.5],
          'lstm_state_size': 1,
          'optimizers': ['Adam', 'Adagrad'],
          'learning_rates': [0.001, 0.01],
          'n_epochs': 100,
	  'data_file': data_file
         }

####################################################################################################################
# Create summary table for log-likelihood for all sets of hyperparameters.
apple = []
for dense_n_node in Params['dense_n_nodes']:
  for dense_dropout_rate in Params['dense_dropout_rates']:
    for regularization in Params['regularizations']:
      for reg_lambda in Params['reg_lambdas']:
        for lstm_dropout_rate in Params['lstm_dropout_rates']:
          for optimizer in Params['optimizers']:
            for learning_rate in Params['learning_rates']:
	      # Get params and save it in model directory.
              # Get the directory to save model and params.
              model_id = 'dense_' + str(dense_n_node[0]) + '_' + str(dense_dropout_rate[0]) + '_' + regularization + '_' + str(reg_lambda)    \
                         + '_lstm_' + str(lstm_dropout_rate) + '_' + optimizer + '_learning_' + str(learning_rate)
              model_dir = models_dir + model_id + '/'
              # Get the value for log-likelihood.
              if path.exists(model_dir + 'Log_Likelihoods.npy'):
                log_likelihood = np.load(model_dir + 'Log_Likelihoods.npy')[-1]
                values = [dense_n_node[0], dense_dropout_rate[0], regularization, reg_lambda,
		          Params['lstm_state_size'], lstm_dropout_rate, optimizer, learning_rate, Params['n_epochs'],
                          round(log_likelihood,3), model_id]
                values = np.array(values)
                apple.append(values)

apple = np.array(apple)
apple = apple[np.argsort(-apple[:,-2].astype('float32')), :]

header = ['dense_n_node', 'dense_dropout_rate', 'regularization', 'reg_lambda',
          'lstm_state_size', 'lstm_dropout_rate', 'optimizer', 'learning_rate', 'n_epochs',
          'log_likelihood', 'model_id']
header = '\t'.join([str(x) for x in header])
np.savetxt(HParams_dir + 'Log-Likelihoods.txt', apple, fmt='%s', delimiter='\t', header=header)


####################################################################################################################
####################################################################################################################
