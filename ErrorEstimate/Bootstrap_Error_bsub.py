####################################################################################################################
####################################################################################################################
# Estimate bootstrap error -- bsub.
# Author: Haiying Kong
# Last Modified: 17 December 2019
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
import Module_Error

####################################################################################################################
# Collect garbage.
gc.collect()

####################################################################################################################
####################################################################################################################
# Set parameters.


####################################################################################################################
####################################################################################################################
# Get argument passed from main code.
args_values = np.delete([sys.argv], [0])
args_keys = ['model_dir', 'boots_dir']
args_values = args_values.tolist()
args = dict(zip(args_keys, args_values))

####################################################################################################################
# Get error estimate for this bootstrap.
error = Module_Error.Bootstrap(args['model_dir'], args['boots_dir'])

# Save the error.
np.save(args['boots_dir']+'BootsError.npy', error)


####################################################################################################################
####################################################################################################################
