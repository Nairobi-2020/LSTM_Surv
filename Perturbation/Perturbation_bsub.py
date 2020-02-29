####################################################################################################################
####################################################################################################################
# Find impact scores for genes with perturbation at input -- bsub.
# Author: Haiying Kong
# Last Modified: 15 December 2019
####################################################################################################################
####################################################################################################################
import tensorflow as tf
import gc
import pickle
import copy
import numpy as np
from scipy.spatial import distance

import sys
sys.path = ['/home/kong/.conda/envs/TensorFlow_CPU/lib/python37.zip', '/home/kong/.conda/envs/TensorFlow_CPU/lib/python3.7', '/home/kong/.conda/envs/TensorFlow_CPU/lib/python3.7/lib-dynload', '/home/kong/.conda/envs/TensorFlow_CPU/lib/python3.7/site-packages', '/home/kong/.conda/envs/kipoi-MMSplice/lib/python3.5/site-packages/MyPythonModules', '/icgc/dkfzlsdf/analysis/B240/kong/Projects/PANCSTRAT/Code/WGS/Kipoi/MMSplice/DeepLearning/Survival/LSTM/']
import LSTM_Surv

####################################################################################################################
# Collect garbage.
gc.collect()

####################################################################################################################
####################################################################################################################
# Get argument passed from main code.
args_values = np.delete([sys.argv], [0])
args_keys = ['Params_file', 'model_dir', 'lock_dir', 'i_gene']
args_values = args_values.tolist()
args = dict(zip(args_keys, args_values))
args['i_gene'] = int(args['i_gene'])

args_keys = ['Params_file', 'model_dir', 'lock_dir', 'i_gene']
args_values = [Params_file, model_dir, lock_dir, 0]

####################################################################################################################
####################################################################################################################
# Define function to compute risk score with perturbed data.
####################################################################################################################
def Perturbed_RiskScore(model, dat):
  # Fully connected layer before LSTM.
  dense_1 = tf.matmul(dat, model.W_b['W_dense_1']) + model.W_b['b_dense_1']
  dense_1 = tf.nn.dropout(dense_1, rate=model.dense_dropout_rate[0])
  dense_1 =  dense_1 / model.dense_dropout_rate[0]
  dense_1 = tf.keras.utils.normalize(dense_1, axis=0)
  mean_variance = tf.nn.moments(dense_1, axes=[0])
  dense_1 = tf.nn.batch_normalization(
              x = dense_1,
              mean = mean_variance[0],
              variance = mean_variance[1],
              offset = np.repeat([1.5], dense_1.shape[1], axis=0),
              scale = np.repeat([1], dense_1.shape[1], axis=0),
              variance_epsilon = 1e-4)
  dense_1 = tf.nn.relu(dense_1)    \

  # Prepare inputs for LSTM layer.
  lstm_inputs = []
  unstacked_dense_1 = tf.unstack(dense_1, axis=0)    \

  for step_lstm in unstacked_dense_1:
    # Create LSTM input for one patient.
    lstm_input = []
    for time in model.times:
      one_time_point = tf.concat([tf.reshape(step_lstm,[1,len(step_lstm)]), tf.cast(tf.reshape(time, [1,1]), dtype=tf.float32)], axis=1)
      lstm_input.append(one_time_point)
    lstm_input = tf.concat(lstm_input, axis=0)
    lstm_inputs.append(lstm_input)    \

  # LSTM layer:
  lstm_inputs = tf.stack(lstm_inputs, axis=0)  # First dimension as sample.
  lstm_inputs = tf.transpose(lstm_inputs, [1, 0, 2])
  lstm_outputs, lstm_states = model.lstm_layer(inputs=lstm_inputs)
  lstm_outputs = tf.transpose(tf.squeeze(lstm_outputs))    \

  risk_scores = tf.matmul(lstm_outputs, model.W_b['W_cox_score']) + model.W_b['b_cox_score']
  return tf.squeeze(risk_scores)

####################################################################################################################
####################################################################################################################
# Load model.
####################################################################################################################
# Load model params.
pkl_file = open(args['model_dir'] + 'params.pkl', 'rb')
params = pickle.load(pkl_file)
pkl_file.close()

# Load data.
pkl_file = open(params['data_file'], 'rb')
data = pickle.load(pkl_file)
pkl_file.close()

# Get a model template with training data.
model = LSTM_Surv.LSTM_Surv(params)
data['pheno']['SurvivalTime'] = copy.deepcopy(data['pheno']['SurvivalTime']/model.time_stepsize)
loss, grads = model.Gradients(data)

# Create template for W_b.
tensor_names = np.loadtxt(args['model_dir'] + 'W_b_TensorNames.npy', dtype='U20')
tensor_names = list(tensor_names)
W_b = {}
for i in range(len(model.trainable_variables)):
  W_b[tensor_names[i]] = model.trainable_variables[i]

# Load W_b at the last checkpoint of the best model.
checkpoint = tf.train.Checkpoint()
manager = tf.train.CheckpointManager(checkpoint, args['model_dir'], max_to_keep=3)
checkpoint.W_b = W_b
checkpoint.restore(manager.latest_checkpoint)

# Update model.W_b with checkpoint.W_b
keys = list(W_b.keys())[0:6]
values = list(W_b.values())[0:6]
model.W_b = dict(zip(keys, values))

# Update model.lstm_layer weights.
lstm_weights = list(W_b.values())[6:9]
for i in range(len(lstm_weights)):
  lstm_weights[i] = lstm_weights[i].numpy()

model.lstm_layer.set_weights(lstm_weights)

####################################################################################################################
# Get gene names.
####################################################################################################################
pkl_file = open(Params_file, 'rb')
Params = pickle.load(pkl_file)
pkl_file.close()
genes = Params['genes']

####################################################################################################################
# Compute changes on risk score upon perturbation.
####################################################################################################################
# Get one score for each gene each sample.
conv_out = tf.nn.conv1d(input=model.features, filters=model.W_b['W_conv'], stride=model.features.shape[1], padding='VALID', data_format='NWC')
conv_out = tf.squeeze(conv_out)
mean_variance = tf.nn.moments(conv_out, axes=[0])
conv_out = tf.nn.batch_normalization(
             x = conv_out,
             mean = mean_variance[0],
             variance = mean_variance[1],
             offset = np.repeat([1.5], conv_out.shape[1], axis=0),
             scale = np.repeat([1], conv_out.shape[1], axis=0),
             variance_epsilon = 1e-4)
conv_out = tf.nn.relu(conv_out)

# Find min and max values for i_gene.
score_min = min(conv_out[:, args['i_gene']]).numpy()
score_max = max(conv_out[:, args['i_gene']]).numpy()
if abs(score_min) > abs(score_max):
  score_min = score_min * 5
elif abs(score_min) < abs(score_max):
  score_max = score_max * 5

risk_scores = []
for i_pid in range(len(model.pheno)):
 # Find risk score after setting i_pid, i_gene as min.
  aster = copy.deepcopy(conv_out).numpy()
  aster[i_pid, args['i_gene']] = score_min
  risk_min = Perturbed_RiskScore(model, aster)    \

  # Find risk score after setting i_pid, i_gene as min.
  aster = copy.deepcopy(conv_out).numpy()
  aster[i_pid, args['i_gene']] = score_max
  risk_max = Perturbed_RiskScore(model, aster)    \

  # Add risk_min and risk_max to the risk_scores list.
  risk_scores.append([risk_min, risk_max])

risk_scores = np.array(risk_scores)
#impact = distance.euclidean(risk_scores[:,0], risk_scores[:,1])

header = ['RiskScore_min', 'RiskScore_max']
header = '\t'.join([str(x) for x in header])
np.savetxt(res_dir + 'Gene_Impact.txt', apple, fmt='%s', delimiter='\t', header=header)

####################################################################################################################
####################################################################################################################
