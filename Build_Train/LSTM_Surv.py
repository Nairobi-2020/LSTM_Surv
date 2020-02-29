####################################################################################################################
####################################################################################################################
# Define LSTM_Surv class. 
# Author: Haiying Kong
# Last Modified: 15 December 2019
####################################################################################################################
####################################################################################################################
import tensorflow as tf
import gc
import numpy as np
import pickle
import copy

####################################################################################################################
###################################################################################################################
# Define the model.
class LSTM_Surv(tf.Module):    \

  ##################################################################################################################
  def __init__(self, params, name=None):
    super(LSTM_Surv, self).__init__(name=name)
    with self.name_scope:
      self.feature_names = params['feature_names']
      self.n_features = params['n_features']
      self.n_genes = params['n_genes']
      self.n_times = params['n_times']
      self.times = list(range(self.n_times))
      self.time_stepsize = params['time_stepsize']
      self.dense_n_node = params['dense_n_node']
      self.dense_dropout_rate = params['dense_dropout_rate']
      self.regularization = params['regularization']
      self.reg_lambda = params['reg_lambda']
      self.lstm_state_size = params['lstm_state_size']
      self.lstm_dropout_rate = params['lstm_dropout_rate']
      lstm_cell = tf.keras.layers.LSTMCell(units=self.lstm_state_size, dropout=self.lstm_dropout_rate)
      self.lstm_layer = tf.keras.layers.RNN([lstm_cell], time_major=True, return_sequences=True, return_state=True)
      self.optimizer = params['optimizer']
      self.learning_rate = params['learning_rate']
      self.n_epochs = params['n_epochs']
      weight_lstm_hazards = tf.random.uniform(shape=[], minval=0, maxval=1)
      weight_cox_hazards = 1 - weight_lstm_hazards
      weights_two_hazards = tf.reshape(tf.stack([weight_lstm_hazards, weight_cox_hazards]), [2,1])
      self.W_b = {
        'W_conv': tf.Variable(tf.random.truncated_normal([self.n_features, self.n_genes, self.n_genes]), dtype=tf.float32, name='W_conv'),
        'W_dense_1': tf.Variable(tf.random.truncated_normal([self.n_genes, self.dense_n_node[0]]), dtype=tf.float32, name='W_dense_1'),
	'W_cox_score': tf.Variable(tf.random.truncated_normal([self.n_times, 1]), dtype=tf.float32, name='W_cox_score'),
	'W_hazards': tf.Variable(weights_two_hazards, dtype=tf.float32, name='W_hazards'),
        'b_dense_1': tf.Variable(tf.zeros([self.dense_n_node[0]]), dtype=tf.float32, name='b_dense_1'),
	'b_cox_score': tf.Variable(tf.zeros([1]), dtype=tf.float32, name='b_cox_score')
        }

  ##################################################################################################################
  @tf.Module.with_name_scope
  def __call__(self, data):    \

    self.features = tf.dtypes.cast(data['features'], dtype=tf.float32)
    self.pheno = data['pheno']    \

    # Pre dense the features and create matrix for the scores by genes and by samples.
    conv_out = tf.nn.conv1d(input=self.features, filters=self.W_b['W_conv'], stride=self.features.shape[1], padding='VALID', data_format='NWC')
    conv_out = tf.squeeze(conv_out)
    mean_variance = tf.nn.moments(conv_out, axes=[0])
    conv_out = tf.nn.batch_normalization(
                 x = conv_out,
                 mean = mean_variance[0],
                 variance = mean_variance[1],
                 offset = np.repeat([1.5], conv_out.shape[1], axis=0),
                 scale = np.repeat([1], conv_out.shape[1], axis=0),
                 variance_epsilon = 1e-4)
    conv_out = tf.nn.relu(conv_out)    \
 
    # Fully connected layer before LSTM.
    dense_1 = tf.matmul(conv_out, self.W_b['W_dense_1']) + self.W_b['b_dense_1']
    dense_1 = tf.nn.dropout(dense_1, rate=self.dense_dropout_rate[0])
    dense_1 =  dense_1 / self.dense_dropout_rate[0]
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
      for time in self.times:
        one_time_point = tf.concat([tf.reshape(step_lstm,[1,len(step_lstm)]), tf.cast(tf.reshape(time, [1,1]), dtype=tf.float32)], axis=1)
        lstm_input.append(one_time_point)
      lstm_input = tf.concat(lstm_input, axis=0)
      lstm_inputs.append(lstm_input)    \
 
    # LSTM layer:
    lstm_inputs = tf.stack(lstm_inputs, axis=0)  # First dimension as sample.
    lstm_inputs = tf.transpose(lstm_inputs, [1, 0, 2])
    lstm_outputs, lstm_states = self.lstm_layer(inputs=lstm_inputs)
    lstm_outputs = tf.transpose(tf.squeeze(lstm_outputs))    \

    risk_scores = tf.matmul(lstm_outputs, self.W_b['W_cox_score']) + self.W_b['b_cox_score']
    self.risk_scores = tf.squeeze(risk_scores)
    self.lstm_hazards = tf.math.exp(lstm_outputs)
    self.lstm_hazards = tf.slice(self.lstm_hazards, [0,1], [self.lstm_hazards.shape[0], self.lstm_hazards.shape[1]-1])    \

    return self

  ##################################################################################################################
  # Breslow estimate on baseline hazards with the risk_scores
  def Breslow(self):
    unstacked_risk_scores = tf.unstack(self.risk_scores)
    # Find cummulative baseline hazards with Breslow estimator.
    cumm_breslow = []
    for time in self.times:
      cumm = 0
      i_happeneds = np.where((self.pheno['SurvivalTime'] <= time) & (self.pheno['SurvivalStatus'] == 1))[0]
      for i_happened in i_happeneds:
        i_risks = np.where(self.pheno['SurvivalTime'] >= self.pheno['SurvivalTime'][i_happened])[0]
        denominator = tf.constant(0, dtype=tf.float32)
        for i_risk in i_risks:
          denominator = denominator + tf.math.exp(self.risk_scores[i_risk])
        cumm = cumm + 1/denominator
      cumm_breslow.append(cumm)    \
 
    # Find baseline hazard probability.
    breslow = []
    for i in range(len(self.times)-1):
      breslow.append(cumm_breslow[i+1] - cumm_breslow[i])
    self.breslow = tf.stack(breslow)    \
 
    cumm_breslow.pop(0)
    self.cumm_breslow = tf.stack(cumm_breslow)    \
 
    # Find hazard for all riskscores. (check back if the first dimension is sample.)
    cox_hazards = []
    for i_pid in range(len(self.pheno)):
      exp_risk = tf.reshape(tf.exp(unstacked_risk_scores[i_pid]), [])
      cox_hazards.append(tf.scalar_mul(exp_risk, self.breslow))
    self.cox_hazards = tf.squeeze(tf.stack(cox_hazards))    \

    return self

  ##################################################################################################################
  # Find hazards and Hazards.
  def Hazards_Survivals(self):
    # Combine lstm hazards and cox hazards, find hazards and cummulative hazards.
    unstacked_lstm_hazards = tf.unstack(self.lstm_hazards)
    unstacked_cox_hazards = tf.unstack(self.cox_hazards)    \

    hazards = []
    Hazards = []
    for i in range(len(unstacked_cox_hazards)):
      # Find hazard as weighted sum of lstm_hazard and cox_hazard.
      hazard = tf.stack([unstacked_lstm_hazards[i], unstacked_cox_hazards[i]], axis=1)
      hazard = tf.matmul(hazard, self.W_b['W_hazards'])
      hazard = tf.squeeze(hazard)
      hazards.append(hazard)
      # Find cumulative hazard.
      hazard_0 = tf.slice(hazard, [0], [hazard.shape[0]-1])
      hazard_1 = tf.slice(hazard, [1], [hazard.shape[0]-1])
      block = tf.math.scalar_mul(0.5, tf.math.add(hazard_1, hazard_0))
      Hazard = tf.math.cumsum(block)
      Hazard = tf.concat([tf.slice(hazard, [0], [1]), Hazard], axis=0)
      Hazards.append(Hazard) 
    hazards = tf.stack(hazards)
    self.hazards = tf.squeeze(hazards)
    Hazards = tf.stack(Hazards)
    self.Hazards = tf.squeeze(Hazards)
    self.Survivals = tf.math.exp(-self.Hazards)    \
 
    return self

  ##################################################################################################################
  # Define log_likelihood function.
  def Log_Likelihood(self):
    log_likelihood = 0
    unstacked_hazards = tf.unstack(self.hazards)
    for i_pid in range(len(self.pheno)):
      time = self.pheno['SurvivalTime'][i_pid]
      status = self.pheno['SurvivalStatus'][i_pid]
      tim = int(time)
      if tim <= min(self.times):
        Hazard = self.hazards[i_pid, 0] * time
        if status == 0:
          log_likelihood = log_likelihood - Hazard
        if status == 1:
          log_likelihood = log_likelihood + tf.math.log(self.hazards[i_pid, 0])- Hazard
      elif tim >= max(self.times):
        Hazard = self.Hazards[i_pid, self.n_times-2] + self.hazards[i_pid, self.n_times-2] * (time-self.times[self.n_times-2])
        if status == 0:
          log_likelihood = log_likelihood - Hazard
        if status == 1:
          log_likelihood = log_likelihood + tf.math.log(self.hazards[i_pid, self.n_times-2])- Hazard
      else:
        Hazard = self.Hazards[i_pid, tim-1] + (self.hazards[i_pid, tim] - self.hazards[i_pid, tim-1]) * (time-tim)**2 / 2
        if status == 0:
          log_likelihood = log_likelihood - Hazard
        if status == 1:
          hazard = self.hazards[i_pid, tim-1] + (self.hazards[i_pid, tim] - self.hazards[i_pid, tim-1]) * (time-tim)
          log_likelihood = log_likelihood + tf.math.log(hazard)- Hazard    \
 
    self.log_likelihood = log_likelihood
    return self

  ##################################################################################################################
  # Define gradients.
  def Gradients(self, data):
    with tf.GradientTape() as tape:
      self(data)
      self.Breslow()
      self.Hazards_Survivals()
      self.Log_Likelihood()
      loss = -self.log_likelihood
      grads = tape.gradient(loss, self.trainable_variables)
    return loss, grads

  ##################################################################################################################
  # Define training the model with training data and save checkpoint and final model.
  def Train_Checkpoint(self, data, model_dir):    \
 
    if self.optimizer == 'Adam':
      optimizer = tf.keras.optimizers.Adam(learning_rate = self.learning_rate)
    if self.optimizer == 'Adagrad':
      optimizer = tf.keras.optimizers.Adagrad(learning_rate = self.learning_rate)    \

    log_likelihoods = []    \

    # The one epoch to get weight matrices for LSTM.
    epoch = 0
    loss, grads = self.Gradients(data)
    optimizer.apply_gradients(zip(grads, self.trainable_variables))    \

    # Define checkpoint.
    checkpoint = tf.train.Checkpoint(step = tf.Variable(0))
    checkpoint.W_b = {}
    for i in range(len(self.trainable_variables)):
      tensor_name = self.trainable_variables[i].name.split('/')
      tensor_name.pop(0)
      tensor_name = '_'.join(tensor_name)
      tensor_name = tensor_name.split(':')
      tensor_name.pop(-1)
      checkpoint.W_b[tensor_name[0]] = self.trainable_variables[i]    \

    # Save the names of tensors at checkpoint.
    tensor_names = np.array(list(checkpoint.W_b.keys()))
    np.savetxt(model_dir+'W_b_TensorNames.npy', tensor_names, fmt='%s', delimiter='\t')    \

    # Define checkpoint manager.
    manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=3)    \

    # Train the model.
    if manager.latest_checkpoint:
      print('Restored from {}'.format(manager.latest_checkpoint))
    else:
      print('Initializing from scratch.')    \

    checkpoint.step.assign_add(1)
    if int(checkpoint.step) % 5 == 0:
      manager.save()
      print('Saved checkpoint for step {}: {}'.format(int(checkpoint.step), model_dir))
      print('loss {:1.4f}'.format(loss.numpy()))    \

    # Add loss of the epoch to the list.
    log_likelihood = -loss.numpy().item()
    log_likelihoods.append(log_likelihood)    \

    for epoch in range(1, self.n_epochs):
      loss, grads = self.Gradients(data)
      optimizer.apply_gradients(zip(grads, self.trainable_variables))    \
 
      checkpoint.step.assign_add(1)
      if int(checkpoint.step) % 5 == 0:
        manager.save()
        print('Saved checkpoint for step {}: {}'.format(int(checkpoint.step), model_dir))
        print('loss {:1.4f}'.format(loss.numpy()))    \
 
      # Add loss of the epoch to the list.
      log_likelihood = -loss.numpy().item()
      log_likelihoods.append(log_likelihood)    \
 
    # tf.saved_model.save(self, model_dir)
    log_likelihoods = np.array(log_likelihoods)    \
 
    return log_likelihoods

  ##################################################################################################################
  # Define prediction model.
  def Prediction_Model(model_dir, data):    \

    # Load hyperparameters.
    pkl_file = open(model_dir + 'params.pkl', 'rb')
    params = pickle.load(pkl_file)
    pkl_file.close()    \
 
    # Get a model template with training data.
    model = LSTM_Surv(params)
    data['pheno']['SurvivalTime'] = copy.deepcopy(data['pheno']['SurvivalTime']/model.time_stepsize)
    loss, grads = model.Gradients(data)    \
 
    # Create template for W_b.
    tensor_names = np.loadtxt(model_dir + 'W_b_TensorNames.npy', dtype='U20')
    tensor_names = list(tensor_names)
    W_b = {}
    for i in range(len(model.trainable_variables)):
      W_b[tensor_names[i]] = model.trainable_variables[i]    \
 
    # Load W_b at the last checkpoint of the best model.
    checkpoint = tf.train.Checkpoint()
    manager = tf.train.CheckpointManager(checkpoint, model_dir, max_to_keep=3)
    checkpoint.W_b = W_b
    checkpoint.restore(manager.latest_checkpoint)    \
 
    # Update model.W_b with checkpoint.W_b
    keys = list(W_b.keys())[0:6]
    values = list(W_b.values())[0:6]
    model.W_b = dict(zip(keys, values))    \
 
    # Update model.lstm_layer weights.
    lstm_weights = list(W_b.values())[6:9]
    for i in range(len(lstm_weights)):
      lstm_weights[i] = lstm_weights[i].numpy()
    model.lstm_layer.set_weights(lstm_weights)    \

    # Compute all survival statistics with the model.
    model(data)
    model.Breslow()
    model.Hazards_Survivals()    \

    return model


####################################################################################################################
####################################################################################################################
