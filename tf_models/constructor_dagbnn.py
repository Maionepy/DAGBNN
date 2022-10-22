import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tf_models.fc_new import FC
from tf_models.dagbnn import DAGBNN
from tf_models.gl import GL
from tf_models.glKeras import ModelGL


def construct_model(obs_dim=11, act_dim=3, rew_dim=1, hidden_dim=200, num_networks=7, num_elites=5, session=None):
    print('[ DAGBNN ] Observation dim {} | Action dim: {} | Hidden dim: {}'.format(obs_dim, act_dim, hidden_dim))
    params = {'name': 'DAGBNN', 'num_networks': num_networks, 'num_elites': num_elites, 'sess': session}
    model = DAGBNN(params)

    model.add(FC(hidden_dim, input_dim=obs_dim + 2*act_dim, activation="swish", weight_decay=0.000025))
    model.add(FC(hidden_dim, activation="swish", weight_decay=0.00005))
    model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
    model.add(FC(hidden_dim, activation="swish", weight_decay=0.000075))
    model.add(GL(obs_dim + rew_dim, action_dim = act_dim, weight_decay=0.0001))
    model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
    return model

def construct_model_keras(obs_dim=11, act_dim=3, rew_dim=1, hidden_dim=200, num_networks=7, num_elites=5, session=None):
    model = ModelGL(input_dim=obs_dim + 2*act_dim, hidden_dim=hidden_dim, output_dim= obs_dim + rew_dim)
    model.compile(optimizer='Adam', loss='mse')
    return model