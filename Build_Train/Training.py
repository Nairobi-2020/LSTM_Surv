####################################################################################################################
####################################################################################################################
# Build deep learning survival model, and train the model.
# Author: Haiying Kong
# Last Modified: 13 December 2019
####################################################################################################################
####################################################################################################################
import tensorflow as tf
import gc
import numpy as np
from collections import OrderedDict
import pickle
import os
import shutil
from bsub import bsub

####################################################################################################################
# Collect garbage.
gc.collect()

####################################################################################################################
####################################################################################################################
# Set parameters, create clean folders.
thresh = 3
feat = 'd'
dir_name = '/icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Lock/WGS/Kipoi/MMSplice/Aggregated_byGene/Max/byFeature/Survival/Tumor' + str(thresh) + '/'
data_dir = dir_name + 'pkl/'
data_file = data_dir + 'All_' + feat + '_0.pkl'

dir_name = dir_name + 'Features_' + feat + '/'
HParams_dir = dir_name + 'HParams/'
models_dir = dir_name + 'Models/'
if os.path.isdir(dir_name):
  shutil.rmtree(dir_name)

os.mkdir(dir_name)
os.mkdir(HParams_dir)
os.mkdir(models_dir)

err_dir = '/icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Err_Out/DeepLearning/Survival/LSTM/' + 'Features_' + feat + '/'
if os.path.isdir(err_dir):
  shutil.rmtree(err_dir)

os.mkdir(err_dir)

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
          'n_epochs': 50,
	  'data_file': data_file
         }

####################################################################################################################
# Submit jobs to train the model.
for dense_n_node in Params['dense_n_nodes']:
  for dense_dropout_rate in Params['dense_dropout_rates']:
    for regularization in Params['regularizations']:
      for reg_lambda in Params['reg_lambdas']:
        for lstm_dropout_rate in Params['lstm_dropout_rates']:
          for optimizer in Params['optimizers']:
            for learning_rate in Params['learning_rates']:    \

   	        # Get params and save it in model directory.
                keys = ['feature_names', 'n_features', 'n_genes', 'n_times', 'time_stepsize',
                        'dense_n_node', 'dense_dropout_rate', 'regularization', 'reg_lambda',
		        'lstm_state_size', 'lstm_dropout_rate', 'optimizer', 'learning_rate', 'n_epochs', 'data_file']
                values = [Params['feature_names'], Params['n_features'], Params['n_genes'], Params['n_times'], Params['time_stepsize'],
                          dense_n_node, dense_dropout_rate, regularization, reg_lambda,
		  	  Params['lstm_state_size'], lstm_dropout_rate, optimizer, learning_rate, Params['n_epochs'], Params['data_file']]
                params = dict(zip(keys, values))
                # Get the directory to save model and params.
                model_id = 'dense_' + str(dense_n_node[0]) + '_' + str(dense_dropout_rate[0]) + '_' + regularization + '_' + str(reg_lambda)    \
                           + '_lstm_' + str(lstm_dropout_rate) + '_' + optimizer + '_learning_' + str(learning_rate)
                model_dir = models_dir + model_id + '/'
                # Create new model_dir.
                if os.path.isdir(model_dir):
                  shutil.rmtree(model_dir)
                os.mkdir(model_dir)
                # Save the params for this set of hypyerparameters.
                output = open(model_dir+'params.pkl', 'wb')
                pickle.dump(params, output)
                output.close()    \

                # Submit job for model training with this hyperparameter set.
                job_name = err_dir + model_id
                job = bsub(job_name, W='50:00', M='20G', verbose=True)
                args = model_dir
                job("module load anaconda3/2019.07; source activate TensorFlow_CPU; python /icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Code/WGS/Kipoi/MMSplice/DeepLearning/Survival/LSTM/Training_bsub.py" + ' ' + args)


####################################################################################################################
####################################################################################################################
