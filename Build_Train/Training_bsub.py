####################################################################################################################
####################################################################################################################
# Build deep learning survival model, and train the model -- bsub.
# Author: Haiying Kong
# Last Modified: 10 December 2019
####################################################################################################################
####################################################################################################################
import tensorflow as tf
import gc
import numpy as np
from collections import OrderedDict
import pickle
import os

import sys
sys.path = ['/home/kong/.conda/envs/TensorFlow_CPU/lib/python37.zip', '/home/kong/.conda/envs/TensorFlow_CPU/lib/python3.7', '/home/kong/.conda/envs/TensorFlow_CPU/lib/python3.7/lib-dynload', '/home/kong/.conda/envs/TensorFlow_CPU/lib/python3.7/site-packages', '/home/kong/.conda/envs/kipoi-MMSplice/lib/python3.5/site-packages/MyPythonModules', '/icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Code/WGS/Kipoi/MMSplice/DeepLearning/Survival/LSTM']
import LSTM_Surv

####################################################################################################################
# Collect garbage.
gc.collect()

####################################################################################################################
####################################################################################################################
# Set parameters.


####################################################################################################################
####################################################################################################################
# Get argument (model_dir) passed from main code.
args = np.delete([sys.argv], [0]).tolist()
model_dir = str(args[0])

####################################################################################################################
# Load params from model_dir.
pkl_file = open(model_dir+'params.pkl', 'rb')
params = pickle.load(pkl_file)
pkl_file.close()

# Load data.
pkl_file = open(params['data_file'], 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

# Instantiate LSTM_Surv class.
model = LSTM_Surv.LSTM_Surv(params)
data['pheno']['SurvivalTime'] = data['pheno']['SurvivalTime']/model.time_stepsize
log_likelihoods = LSTM_Surv.LSTM_Surv.Train_Checkpoint(model, data, model_dir)

# Save the log_likelihoods for all epochs.
np.save(model_dir+'Log_Likelihoods.npy', log_likelihoods)


####################################################################################################################
####################################################################################################################
