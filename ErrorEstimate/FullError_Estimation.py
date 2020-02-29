####################################################################################################################
####################################################################################################################
# Estimate error for LSTM_Surv model and Kaplan-Meier.
# Author: Haiying Kong
# Last Modified: 15 December 2019
####################################################################################################################
####################################################################################################################
import tensorflow as tf
import os
import gc
import numpy as np
from collections import OrderedDict
import pickle
import copy

import sys
sys.path = ['/home/kong/.conda/envs/TensorFlow_CPU/lib/python37.zip', '/home/kong/.conda/envs/TensorFlow_CPU/lib/python3.7', '/home/kong/.conda/envs/TensorFlow_CPU/lib/python3.7/lib-dynload', '/home/kong/.conda/envs/TensorFlow_CPU/lib/python3.7/site-packages', '/home/kong/.conda/envs/kipoi-MMSplice/lib/python3.5/site-packages/MyPythonModules', '/icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Code/WGS/Kipoi/MMSplice/DeepLearning/Survival/LSTM/']
import LSTM_Surv
import Module_Error

####################################################################################################################
# Collect garbage.
gc.collect()

####################################################################################################################
####################################################################################################################
# The chosen model id:
model_id = 'dense_64_0.3_l1_0.3_lstm_0.3_Adam_learning_0.01'

####################################################################################################################
# Set parameters.
thresh = 3
feat = 'd'
n_boots = 50
dir_name = '/icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Lock/WGS/Kipoi/MMSplice/Aggregated_byGene/Max/byFeature/Survival/Tumor' + str(thresh) + '/'
data_dir = dir_name + 'pkl/'
data_file = data_dir + 'All_' + feat + '_0.pkl'

dir_name = dir_name + 'Features_' + feat + '/'
model_dir = dir_name + 'Models/' + model_id + '/'
boots_dir = dir_name + 'Bootstrap/'
res_dir = '/icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Results/WGS/Kipoi/MMSplice/Tumor' + str(thresh) + '/Features_' + feat + '/'

####################################################################################################################
# Load dataset.
pkl_file = open(data_dir + 'All_' + feat + '_0.pkl', 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

####################################################################################################################
####################################################################################################################
# Predict marginal survival probability with prediction model.
####################################################################################################################
model = LSTM_Surv.LSTM_Surv.Prediction_Model(model_dir, data)
model.Breslow()
model.Hazards_Survivals()

marginal_survivals = []
for i_time in range(1, len(model.times)-1):
  marg_surv = 0
  for i_pid in range(model.Survivals.shape[0]):
    marg_surv = marg_surv + round(model.Survivals[i_pid, i_time].numpy(), 4)
  marg_surv = marg_surv/len(model.pheno)
  marginal_survivals.append([model.times[i_time+1], marg_surv])

marginal_survivals = np.array(marginal_survivals)
header = ['Time', 'Survival']
header = '\t'.join([str(x) for x in header])
np.savetxt(res_dir+'LSTM_Surv.txt', marginal_survivals, fmt='%5.4f', delimiter='\t', header=header)

####################################################################################################################
# Compute prediction errors for the model.
####################################################################################################################
# Apparent error:
app_error = Module_Error.Error(model.pheno, model.Survivals, model.n_times, model.times, model.time_stepsize)
# app_error = Error(model.pheno, model.Survivals, model.n_times, model.times, model.time_stepsize)

# Bootstrap error:
boots_error = []
for i_boots in range(n_boots):
  b_dir = boots_dir + 'boots_' + str(i_boots) + '/'
  b_error = np.load(b_dir + 'BootsError.npy')
  boots_error.append(b_error)

boots_error = np.array(boots_error)
idx = np.unique(np.where(boots_error>1)[0])
boots_error = np.delete(boots_error, list(idx), axis=0)
boots_error = np.mean(boots_error, axis=0)

# Find full error as weighted sum of apparent error and bootstrap error.
error = (1-0.632)*app_error + 0.632*boots_error

# Save the prediction error for LSTM_Surv model.
header = ['Error']
header = '\t'.join([str(x) for x in header])
np.savetxt(res_dir+'LSTM_Surv_Error.txt', error, fmt='%5.4f', delimiter='\t', header=header)

####################################################################################################################
####################################################################################################################
# Define function to get Kaplan-Meier's estimate for marginal survival probability.
def Kaplan_Meier(status, time):    \

  idx = np.argsort(time)
  status = status[idx]
  time = time[idx]
  event_idx = np.where(status==1)[0]    \

  kaplan_meier = []
  prob = 1
  for idx in event_idx:
    n_at_risk = len(np.where(time >= time[idx])[0])
    n_event = len(np.where((time == time[idx]) & (status == 1))[0])
    prob = prob * (1 - n_event/n_at_risk)
    kaplan_meier.append([time[idx], round(prob, 4)])
  kaplan_meier = np.array(kaplan_meier)    \

  return kaplan_meier

####################################################################################################################
# Define function to get Kaplan-Meier estimate at time points of interest.
def KM_survivals(kaplan_meier, times):    \

  km_survivals = []
  for time in times:
    idx_closest = np.argmin(abs(kaplan_meier[:,0] - time))
    if kaplan_meier[idx_closest,0] <= time:
      survival = kaplan_meier[idx_closest, 1]
    else:
      survival = kaplan_meier[idx_closest-1, 1]
    km_survivals.append([time, survival])
  km_survivals = np.array(km_survivals)[1:, ]    \

  return km_survivals

####################################################################################################################
# Compute prediction errors for Kaplan-Meier.
####################################################################################################################
# Apparent error:
status = copy.deepcopy(model.pheno['SurvivalStatus'])
time = copy.deepcopy(model.pheno['SurvivalTime'])
kaplan_meier = Kaplan_Meier(status, time)
km_survivals = KM_survivals(kaplan_meier, model.times)
header = ['Time', 'Survival']
header = '\t'.join([str(x) for x in header])
np.savetxt(res_dir+'Kaplan_Meier.txt', km_survivals, fmt='%5.4f', delimiter='\t', header=header)
km_survivals = np.array([km_survivals[:,1],] * model.Survivals.shape[0])
app_error = Module_Error.Error(model.pheno, km_survivals, model.n_times, model.times, model.time_stepsize)

# Bootstrap error:
boots_error = []
for i_boots in range(n_boots):
  n_in_sam = len(model.pheno)
  i_in_pids = np.random.choice(range(len(model.pheno)), n_in_sam, replace=True)
  i_out_pids = np.delete(np.array(range(len(model.pheno))), i_in_pids)
  train = copy.deepcopy(model.pheno[i_in_pids])
  test = copy.deepcopy(model.pheno[i_out_pids])    \

  status = train['SurvivalStatus']
  time = train['SurvivalTime']
  kaplan_meier = Kaplan_Meier(status, time)
  km_survivals = KM_survivals(kaplan_meier, model.times)
  km_survivals = np.array([km_survivals[:,1],] * test.shape[0])
  b_error = Module_Error.Error(test, km_survivals, model.n_times, model.times, model.time_stepsize)
  boots_error.append(b_error)

boots_error = np.array(boots_error)
boots_error = np.mean(boots_error, axis=0)

# Find full error as weighted sum of apparent error and bootstrap error.
error = (1-0.632)*app_error + 0.632*boots_error

# Save the prediction error for Kaplan-Meier model.
header = ['Error']
header = '\t'.join([str(x) for x in header])
np.savetxt(res_dir+'Kaplan_Meier_Error.txt', error, fmt='%5.4f', delimiter='\t', header=header)

####################################################################################################################
####################################################################################################################
