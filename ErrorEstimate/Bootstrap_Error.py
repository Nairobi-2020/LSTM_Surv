####################################################################################################################
####################################################################################################################
# Estimate bootstrap error.
# Author: Haiying Kong
# Last Modified: 16 December 2019
####################################################################################################################
####################################################################################################################
import gc
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
feat = 'e'
n_bootstrap = 50

dir_name = '/icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Lock/WGS/Kipoi/MMSplice/Aggregated_byGene/Max/byFeature/Survival/Tumor' + str(thresh) + '/'
data_dir = dir_name + 'pkl/'
data_file = data_dir + 'All_' + feat + '_0.pkl'

dir_name = dir_name + 'Features_' + feat + '/'
HParams_dir = dir_name + 'HParams/'
models_dir = dir_name + 'Models/'

bootstrap_dir = dir_name + 'Bootstrap/'
if os.path.isdir(bootstrap_dir):
  shutil.rmtree(bootstrap_dir)

os.mkdir(bootstrap_dir)

err_dir = '/icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Err_Out/DeepLearning/Survival/LSTM/' + 'Features_' + feat + '_Boots/'
if os.path.isdir(err_dir):
  shutil.rmtree(err_dir)

os.mkdir(err_dir)


####################################################################################################################
####################################################################################################################
# Get parameters for the model of choice.
model_id = 'dense_16_0.1_l1_0.5_lstm_0.3_Adam_learning_0.01'
model_dir = models_dir + model_id + '/'

####################################################################################################################
# Bootstrap.
for i_boots in range(n_bootstrap):
  boots_dir = bootstrap_dir + 'boots_' + str(i_boots) + '/'
  if os.path.isdir(boots_dir):
    shutil.rmtree(boots_dir)
  os.mkdir(boots_dir)    \

  # Submit one job to find error for one bootstrap.
  job_name = err_dir + 'boots_' + str(i_boots)
  job = bsub(job_name, W='50:00', M='10G', verbose=True)
  args = model_dir + ' ' + boots_dir
  job('module load anaconda3/2019.07; source activate TensorFlow_CPU; python /icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Code/WGS/Kipoi/MMSplice/DeepLearning/Survival/LSTM/Bootstrap_Error_bsub.py' + ' ' + args)


####################################################################################################################
####################################################################################################################
