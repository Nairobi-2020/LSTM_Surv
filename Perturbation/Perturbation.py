####################################################################################################################
####################################################################################################################
# Find impact scores for genes with perturbation at input.
# Author: Haiying Kong
# Last Modified: 15 December 2019
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
feat = 'd'

dir_name = '/icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Lock/WGS/Kipoi/MMSplice/Aggregated_byGene/Max/byFeature/Survival/Tumor' + str(thresh) + '/' + 'Features_' + feat + '/'
models_dir = dir_name + 'Models/'
Params_file = dir_name + '../pkl/Params_' + feat + '.pkl'
lock_dir = dir_name + 'Perturbation/'
if os.path.isdir(lock_dir):
  shutil.rmtree(lock_dir)

os.mkdir(lock_dir)

err_dir = '/icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Err_Out/DeepLearning/Survival/LSTM/' + 'Features_' + feat + '_Perturbation/'
if os.path.isdir(err_dir):
  shutil.rmtree(err_dir)

os.mkdir(err_dir)

####################################################################################################################
####################################################################################################################
# Get parameters for the model of choice.
model_id = 'dense_64_0.3_l2_0.1_lstm_0.5_Adam_learning_0.01'
model_dir = models_dir + model_id + '/'

####################################################################################################################
# Compute changes on risk score upon perturbation.
####################################################################################################################
# Get one score for each gene each sample.
for i_gene in range(conv_out.shape[1]):
  job_name = err_dir + 'Gene_' + i_gene
  job = bsub(job_name, W='10:00', M='10G', verbose=True)
  args = Params_file + ' ' + model_dir + ' ' + lock_dir + ' ' + str(i_gene)
  job("module load anaconda3/2019.07; source activate TensorFlow_CPU; python /icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Code/WGS/Kipoi/MMSplice/DeepLearning/Survival/LSTM/Perturbation_bsub.py" + ' ' + args)


####################################################################################################################
####################################################################################################################
