####################################################################################################################
####################################################################################################################
# Module for error estimate.
# Author: Haiying Kong
# Last Modified: 16 December 2019
####################################################################################################################
####################################################################################################################
import tensorflow as tf
import os
import gc
import numpy as np
import pickle
from collections import OrderedDict

import sys
sys.path = ['/home/kong/.conda/envs/TensorFlow_CPU/lib/python37.zip', '/home/kong/.conda/envs/TensorFlow_CPU/lib/python3.7', '/home/kong/.conda/envs/TensorFlow_CPU/lib/python3.7/lib-dynload', '/home/kong/.conda/envs/TensorFlow_CPU/lib/python3.7/site-packages', '/home/kong/.conda/envs/kipoi-MMSplice/lib/python3.5/site-packages/MyPythonModules', '/icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Code/WGS/Kipoi/MMSplice/DeepLearning/Survival/LSTM/']
import LSTM_Surv

####################################################################################################################
# Collect garbage.
gc.collect()

####################################################################################################################
####################################################################################################################
# Find array for statuses for all patients at all time points.
####################################################################################################################
def SurvivalStatus_Matrix(pheno, n_times, time_stepsize):    \

  statuses = []
  for i_pid in range(len(pheno)):
    time = pheno['SurvivalTime'][i_pid]
    status = pheno['SurvivalStatus'][i_pid]
    tim = int(time)
    if time > n_times:
      statuses.append(np.repeat(1, n_times-1))
    elif time < time_stepsize:
      if status == 1:
        statuses.append(np.repeat(0, n_times-1))
      elif status == 0:
        statuses.append(np.repeat(-1, n_times-1))
    else:
      if status == 1:
        one_status = np.repeat(0, n_times-1)
        if tim == 1:
          one_status[0] = 1
        else:
          one_status[0:(tim-1)] = 1
        statuses.append(one_status)
      if status == 0:
        one_status = np.repeat(-1, n_times-1)
        if tim == 1:
          one_status[0] = 1
        else:
          one_status[0:(tim-1)] = 1
        statuses.append(one_status)
  statuses = np.array(statuses)    \

  return statuses

####################################################################################################################
# Define censoring probability function.
####################################################################################################################
def G(pheno):    \

  censor_status = 1 - pheno['SurvivalStatus']
  censor_time = pheno['SurvivalTime']
  idx = np.argsort(censor_time)
  censor_status = censor_status[idx]
  censor_time = censor_time[idx]
  censor_idx = np.where(censor_status==1)[0]
  apple = [[0.0, 1.0]]
  prob = 1
  for idx in censor_idx:
    n_at_risk = len(np.where(censor_time >= censor_time[idx])[0])
    n_censored = len(np.where((censor_time == censor_time[idx]) & (censor_status==1))[0])
    prob = prob * (1 - n_censored/n_at_risk)
    apple.append([censor_time[idx], prob])
  apple = np.array(apple)
  apple[-1,1] = apple[-2,1]/5
  return apple

####################################################################################################################
# Define censoring probability function.
####################################################################################################################
def G_t(g, g_time):    \

  idx_closest = np.argmin(abs(g[:, 0] - g_time))
  if g[idx_closest,0] <= g_time:
    apple = g[idx_closest, 1]
  else:
    apple = g[idx_closest-1, 1]
  return apple

####################################################################################################################
# Define censoring probability function.
####################################################################################################################
def Weights_BrierScore(pheno, n_times, times, time_stepsize):    \

  statuses = SurvivalStatus_Matrix(pheno, n_times, time_stepsize)
  g = G(pheno)
  weights = []
  for i_pid in range(len(pheno)):
    weight = []
    for time in times[0:len(times)-1]:
      tim = time * time_stepsize
      wei = 0
      if pheno['SurvivalTime'][i_pid] > tim:
        wei = 1/G_t(g, tim)
      elif statuses[i_pid, time] == 1:
        wei = 1/G_t(g, pheno['SurvivalTime'][i_pid])
      weight.append(wei)
    weights.append(weight)
  weights = np.array(weights)    \

  return weights

####################################################################################################################
# Define error computation by comparing predicted survival and phenotype survival.
####################################################################################################################
def Error(pheno, survivals, n_times, times, time_stepsize):    \

  statuses = SurvivalStatus_Matrix(pheno, n_times, time_stepsize)
  weights = Weights_BrierScore(pheno, n_times, times, time_stepsize)
  n_pid = survivals.shape[0]
  errors = []
  for i_time in range(n_times-1):
    error = 0
    for i_pid in range(n_pid):
      error = error + (statuses[i_pid,i_time] - survivals[i_pid,i_time])**2 * weights[i_pid,i_time]
    errors.append(error/n_pid)
  errors = np.array(errors)
  return errors

####################################################################################################################
# Define one bootstrap.
####################################################################################################################
def Bootstrap(model_dir, boots_dir):    \

  # Load hyperparameters.
  pkl_file = open(model_dir + 'params.pkl', 'rb')
  params = pickle.load(pkl_file)
  pkl_file.close()
  params['n_epochs'] = 100    \

  # Load data.
  pkl_file = open(params['data_file'], 'rb')
  data = pickle.load(pkl_file)
  pkl_file.close()    \

  # Create train and test data by bootstrapping from data.
  n_in_sam = len(data['pheno'])
  i_in_pids = np.random.choice(range(len(data['pheno'])), n_in_sam, replace=True)
  i_out_pids = np.delete(np.array(range(len(data['pheno']))), i_in_pids)
  train = OrderedDict()
  test = OrderedDict()
  train['features'] = data['features'][i_in_pids, :, :]
  test['features'] = data['features'][i_out_pids, :, :]
  train['pheno'] = data['pheno'][i_in_pids]
  test['pheno'] = data['pheno'][i_out_pids]    \

  output = open(boots_dir + 'train.pkl', 'wb')
  pickle.dump(train, output)
  output.close()
  output = open(boots_dir + 'test.pkl', 'wb')
  pickle.dump(test, output)
  output.close()
  output = open(boots_dir + 'params.pkl', 'wb')
  pickle.dump(params, output)
  output.close()    \

  # Create model for this bootstrapping.
  boots_model = LSTM_Surv.LSTM_Surv(params)    \

  # Train the model with train data.
  log_likelihoods = LSTM_Surv.LSTM_Surv.Train_Checkpoint(boots_model, train, boots_dir)    \

  # Get sruvival estimate for test data with the trained model.
  boots_test = LSTM_Surv.LSTM_Surv.Prediction_Model(boots_dir, test)    \

  # Get error estimate for this boostrap.
  statuses = SurvivalStatus_Matrix(boots_test.pheno, boots_test.n_times, boots_test.time_stepsize)
  error = Error(boots_test.pheno, boots_test.Survivals, boots_test.n_times, boots_test.times, boots_test.time_stepsize)
  error = np.array(error)    \

  return error

####################################################################################################################
####################################################################################################################
