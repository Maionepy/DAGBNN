from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow.compat.v1 as tf

import pdb


class GL:
    """Represents a fully-connected Gbar layer in a network.
    """
    _activations = {
        None: tf.identity,
        "ReLU": tf.nn.relu,
        "tanh": tf.tanh,
        "sigmoid": tf.sigmoid,
        "softmax": tf.nn.softmax,
        "swish": lambda x: x * tf.sigmoid(x)
    }

    def __init__(self, output_dim, action_dim, input_dim=None,
                 activation=None, weight_decay=None, ensemble_size=1):
        """Initializes a fully connected layer.

        Arguments:
            output_dim: (int) The dimensionality of the output of this layer.
            input_dim: (int/None) The dimensionality of the input of this layer.
            activation: (str/None) The activation function applied on the outputs.
                                    See GL._activations to see the list of allowed strings.
                                    None applies the identity function.
            weight_decay: (float) The rate of weight decay applied to the weights of this layer.
            ensemble_size: (int) The number of networks in the ensemble within which this layer will be used.
        """
        # Set layer parameters
        self.input_dim, self.output_dim = input_dim, output_dim
        self.action_dim_multiplier = action_dim
        self.activation = activation
        self.weight_decay = weight_decay
        self.ensemble_size = ensemble_size
        self.is_last_layer = True

        # Initialize internal state
        self.variables_constructed = False
        self.weights, self.biases = None, None
        self.decays = None

    def __repr__(self):
        return "GL(output_dim={!r}, input_dim={!r}, action_dim={!r}, activation={!r}, weight_decay={!r}, ensemble_size={!r})"\
            .format(
                self.output_dim, self.input_dim, self.action_dim_multiplier, self.activation, self.weight_decay, self.ensemble_size
            )

    #### Extensions
    def get_model_vars(self, idx, sess):
        weights, biases = sess.run([self.weights, self.biases])
        weight = weights[idx].copy()
        bias = biases[idx].copy()
        return {'weights': weight, 'biases': bias}

    def set_model_vars(self, idx, sess, variables):
        for attr, var in variables.items():
            tensor = getattr(self, attr)
            op = tensor[idx].assign(var)
            sess.run(op)
            # print('assigned {}: {}'.format(attr, idx))

    def set_model_vars(self, variables):
        ops = [getattr(self, attr).assign(var) for attr, var in variables.items()]
        return ops
        # for attr, var in variables.items():
            # tensor = getattr(self, attr)
            # op = tensor.assign(var)


    def reset(self, sess):
        sess.run(self.weights.initializer)
        sess.run(self.biases.initializer)

    #######################
    # Basic Functionality #
    #######################

    def compute_output_tensor(self, input_tensor, inputsNN):
        """Returns the resulting tensor when all operations of this layer are applied to input_tensor.

        If input_tensor is 2D, this method returns a 3D tensor representing the output of each
        layer in the ensemble on the input_tensor. Otherwise, if the input_tensor is 3D, the output
        is also 3D, where output[i] = layer_ensemble[i](input[i]).

        Arguments:
            input_tensor: (tf.Tensor) The input to the layer.

        Returns: The output of the layer, as described above.
        """
        # Get raw layer outputs
        if len(input_tensor.shape) == 2:
            raw_output = tf.einsum("ij,ajk->aik", input_tensor, self.weights) + self.biases

        elif len(input_tensor.shape) == 3 and input_tensor.shape[0] == self.ensemble_size:

            if len(inputsNN.shape) == 3: #in trainning
                raw_output = tf.matmul(input_tensor, self.weights) + self.biases

                factor_aux = []
                for _ in range(self.ensemble_size):
                    factor_aux_aux = []
                    for __ in range(int(self.output_dim/2 - 1)):
                        factor_aux_aux.append(0.0)
                    for __ in range(self.action_dim_multiplier):
                        factor_aux_aux.append(1.0)
                        factor_aux_aux.append(-1.0)
                    factor_aux.append([factor_aux_aux])
                factor = tf.math.multiply(inputsNN, tf.constant(factor_aux))

                super_multiplier_aux = []
                for nn in range(self.ensemble_size):
                    multiplier_aux = []
                    for action in range(self.action_dim_multiplier):
                        multiplier_aux_aux = []
                        for __ in range(int(self.output_dim/2 - 1)):
                            multiplier_aux_aux.append(0.0)
                        for ______ in range(2):
                            for ________ in range(action):
                                multiplier_aux_aux.append(0.0)
                            multiplier_aux_aux.append(1.0)
                            for ___ in range(self.action_dim_multiplier - action - 1):
                                multiplier_aux_aux.append(0.0)
                        multiplier_aux.append(multiplier_aux_aux)
                    
                    # last_aux = []
                    # for _____ in range(inputsNN.shape[1]):
                    #    last_aux.append(multiplier_aux)
                    # super_multiplier_aux.append(last_aux)
                    # it seems that is working (this new logic)
                    # after discovered the tf.shape logic, we can come back with this one (check if is working first perhaps?)


                    super_multiplier_aux.append(multiplier_aux) ## 
                multiplier = tf.constant(super_multiplier_aux)
                multiplier = multiplier[:,None,:,:] ##
                result = tf.math.multiply(tf.expand_dims(factor, axis=2), multiplier)
                new_factor = tf.reduce_sum(result, axis=-1)
                real_output = []
                shapeNN = tf.shape(inputsNN)
                for _ in range(int(self.output_dim/2 - 1)):
                    multiplier2 = tf.concat([tf.zeros([self.ensemble_size, shapeNN[1], 1 + _], dtype=tf.float32), new_factor[:,:,:1]], axis=-1)
                    for __ in range(self.action_dim_multiplier - 1):
                        multiplier2 = tf.concat([multiplier2, tf.zeros([self.ensemble_size, shapeNN[1], int(self.output_dim/2 - 1) - 1], dtype=tf.float32)], axis=-1)
                        multiplier2 = tf.concat([multiplier2, new_factor[:,:,1 + __ : __ + 2]], axis=-1)
                    multiplier2 = tf.concat([multiplier2, tf.zeros([self.ensemble_size, shapeNN[1], 2*(int(self.output_dim/2 - 1)) - _])], axis=-1)
                    appendix_res = tf.math.multiply(raw_output, multiplier2)
                    real_output.append(tf.expand_dims(tf.reduce_sum(appendix_res, axis = -1),axis=2))
                    
                final = tf.concat(real_output, axis=-1)
                raw_output = tf.concat([raw_output[:,:,:1], final, raw_output[:,:,-(1 + int(self.output_dim/2 - 1)):]], axis = -1)
                


            else: #in prediction
                inputsNN_aux = tf.expand_dims(inputsNN, axis=0)
                inputsNN = tf.repeat(inputsNN_aux, 7, 0)
                raw_output = tf.matmul(input_tensor, self.weights) + self.biases

                factor_aux = []
                for _ in range(self.ensemble_size):
                    factor_aux_aux = []
                    for __ in range(int(self.output_dim/2 - 1)):
                        factor_aux_aux.append(0.0)
                    for __ in range(self.action_dim_multiplier):
                        factor_aux_aux.append(1.0)
                        factor_aux_aux.append(-1.0)
                    factor_aux.append([factor_aux_aux])
                factor = tf.math.multiply(inputsNN, tf.constant(factor_aux))

                super_multiplier_aux = []
                for nn in range(self.ensemble_size):
                    multiplier_aux = []
                    for action in range(self.action_dim_multiplier):
                        multiplier_aux_aux = []
                        for __ in range(int(self.output_dim/2 - 1)):
                            multiplier_aux_aux.append(0.0)
                        for ______ in range(2):
                            for ________ in range(action):
                                multiplier_aux_aux.append(0.0)
                            multiplier_aux_aux.append(1.0)
                            for ___ in range(self.action_dim_multiplier - action - 1):
                                multiplier_aux_aux.append(0.0)
                        multiplier_aux.append(multiplier_aux_aux)
                    
                    # last_aux = []
                    # for _____ in range(inputsNN.shape[1]):
                    #    last_aux.append(multiplier_aux)
                    # super_multiplier_aux.append(last_aux)
                    # it seems that is working (this new logic)
                    super_multiplier_aux.append(multiplier_aux) ## 
                multiplier = tf.constant(super_multiplier_aux)
                multiplier = multiplier[:,None,:,:] ##
                result = tf.math.multiply(tf.expand_dims(factor, axis=2), multiplier)
                new_factor = tf.reduce_sum(result, axis=-1)
                real_output = []
                shapeNN = tf.shape(inputsNN)
                for _ in range(int(self.output_dim/2 - 1)):
                    multiplier2 = tf.concat([tf.zeros([self.ensemble_size, shapeNN[1], 1 + _], dtype=tf.float32), new_factor[:,:,:1]], axis=-1)
                    for __ in range(self.action_dim_multiplier - 1):
                        multiplier2 = tf.concat([multiplier2, tf.zeros([self.ensemble_size, shapeNN[1], int(self.output_dim/2 - 1) - 1], dtype=tf.float32)], axis=-1)
                        multiplier2 = tf.concat([multiplier2, new_factor[:,:,1 + __ : __ + 2]], axis=-1)
                    multiplier2 = tf.concat([multiplier2, tf.zeros([self.ensemble_size, shapeNN[1], 2*(int(self.output_dim/2 - 1)) - _])], axis=-1)
                    appendix_res = tf.math.multiply(raw_output, multiplier2)
                    real_output.append(tf.expand_dims(tf.reduce_sum(appendix_res, axis = -1),axis=2))
                    
                final = tf.concat(real_output, axis=-1)
                raw_output = tf.concat([raw_output[:,:,:1], final, raw_output[:,:,-(1 + int(self.output_dim/2 - 1)):]], axis = -1)

                
        else:
            raise ValueError("Invalid input dimension.")

        # Apply activations if necessary
        return GL._activations[self.activation](raw_output)

    def get_decays(self):
        """Returns the list of losses corresponding to the weight decay imposed on each weight of the
        network.

        Returns: the list of weight decay losses.
        """
        return self.decays

    def copy(self, sess=None):
        """Returns a Layer object with the same parameters as this layer.

        Arguments:
            sess: (tf.Session/None) session containing the current values of the variables to be copied.
                  Must be passed in to copy values.
            copy_vals: (bool) Indicates whether variable values will be copied over.
                       Ignored if the variables of this layer has not yet been constructed.

        Returns: The copied layer.
        """
        new_layer = eval(repr(self))
        return new_layer

    #########################################################
    # Methods for controlling internal Tensorflow variables #
    #########################################################

    def construct_vars(self):
        """Constructs the variables of this fully-connected layer.

        Returns: None
        """
        if self.variables_constructed:  # Ignore calls to this function once variables are constructed.
            return
        if self.input_dim is None or self.output_dim is None:
            raise RuntimeError("Cannot construct variables without fully specifying input and output dimensions.")

        # Construct variables
        self.weights = tf.get_variable(
            "GL_weights",
            shape=[self.ensemble_size, self.input_dim, int(((self.output_dim/2 - 1)*self.action_dim_multiplier + 1) + self.output_dim/2)],
            initializer=tf.truncated_normal_initializer(stddev=1/(2*np.sqrt(self.input_dim)))
        )
        self.biases = tf.get_variable(
            "GL_biases",
            shape=[self.ensemble_size, 1, int(((self.output_dim/2 - 1)*self.action_dim_multiplier + 1) + self.output_dim/2)],
            initializer=tf.constant_initializer(0.0)
        )

        if self.weight_decay is not None:
            self.decays = [tf.multiply(self.weight_decay, tf.nn.l2_loss(self.weights), name="weight_decay")]
        self.variables_constructed = True

    def get_vars(self):
        """Returns the variables of this layer.
        """
        return [self.weights, self.biases]

    ########################################
    # Methods for setting layer parameters #
    ########################################

    def get_input_dim(self):
        """Returns the dimension of the input.

        Returns: The dimension of the input
        """
        return self.input_dim

    def set_input_dim(self, input_dim):
        """Sets the dimension of the input.

        Arguments:
            input_dim: (int) The dimension of the input.

        Returns: None
        """
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.input_dim = input_dim

    def get_output_dim(self):
        """Returns the dimension of the output.

        Returns: The dimension of the output.
        """
        return self.output_dim

    def set_output_dim(self, output_dim):
        """Sets the dimension of the output.

        Arguments:
            output_dim: (int) The dimension of the output.

        Returns: None.
        """
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.output_dim = output_dim

    def get_activation(self, as_func=True):
        """Returns the current activation function for this layer.

        Arguments:
            as_func: (bool) Determines whether the returned value is the string corresponding
                     to the activation function or the activation function itself.

        Returns: The activation function (string/function, see as_func argument for details).
        """
        if as_func:
            return GL._activations[self.activation]
        else:
            return self.activation

    def set_activation(self, activation):
        """Sets the activation function for this layer.

        Arguments:
            activation: (str) The activation function to be used.

        Returns: None.
        """
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.activation = activation

    def unset_activation(self):
        """Removes the currently set activation function for this layer.

        Returns: None
        """
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.set_activation(None)

    def get_weight_decay(self):
        """Returns the current rate of weight decay set for this layer.

        Returns: The weight decay rate.
        """
        return self.weight_decay

    def set_weight_decay(self, weight_decay):
        """Sets the current weight decay rate for this layer.

        Returns: None
        """
        self.weight_decay = weight_decay
        if self.variables_constructed:
            if self.weight_decay is not None:
                self.decays = [tf.multiply(self.weight_decay, tf.nn.l2_loss(self.weights), name="weight_decay")]

    def unset_weight_decay(self):
        """Removes weight decay from this layer.

        Returns: None
        """
        self.set_weight_decay(None)
        if self.variables_constructed:
            self.decays = []

    def set_ensemble_size(self, ensemble_size):
        if self.variables_constructed:
            raise RuntimeError("Variables already constructed.")
        self.ensemble_size = ensemble_size

    def get_ensemble_size(self):
        return self.ensemble_size
