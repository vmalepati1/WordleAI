from __future__ import division
import collections

from keras import backend as K
from keras import activations, initializations, regularizers, constraints
from keras.engine import InputSpec
from keras.engine.topology import Layer


class Tensordot(Layer):
    '''
    Implements:
    O = np.random.random((10, 11, 4, 5)) # Input
    w = np.random.random((9, 4, 5))      # Learning
    np.tensordot(w, O,  [[1, 2], [2, 3]])
    # Shape should be: (9, 10, 11)
    '''
    def __init__(self, extra_dims, axis, init='glorot_uniform', activation='linear',
                 weights=None, W_regularizer=None, b_regularizer=None,
                 activity_regularizer=None, W_constraint=None,
                 b_constraint=None, bias=True, input_shape=None, **kwargs):

        self.init = initializations.get(init)
        self.activation = activations.get(activation)

        if isinstance(extra_dims, collections.Iterable):
            self.extra_dims = extra_dims
        else:
            self.extra_dims = [extra_dims]
        axis = np.asarray(axis)
        axis[1] += 1
        self.axis = axis
        self.input_dim = input_shape
        self.output_dim = self.get_output_shape_for(input_shape)

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=5)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(Tensordot, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(dtype=K.floatx(), shape=input_shape)]

        tensor_shape = self.extra_dims + [None] * len(self.axis[0])
        for ax0, ax1 in zip(self.axis[0], self.axis[1]):
            tensor_shape[ax0] = input_shape[ax1]
        self.W = self.init(tensor_shape, name='{}_W'.format(self.name))

        if self.bias:
            self.b = K.zeros(self.get_output_shape_for(input_shape)[1:],
                             name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        else:
            self.trainable_weights = [self.W]

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)

        self.constraints = {}
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        output = K.T.tensordot(self.W, x, self.axis)
        output = K.T.swapaxes(output, 0, 1)
        if self.bias:
            output += self.b
        output = self.activation(output)
        return output

    def get_output_shape_for(self, input_shape):
        if input_shape:
            keep_dims = list(input_shape)[1:]
            for dim in self.axis[1]:
                keep_dims[dim - 1] = None
            keep_dims = filter(None, keep_dims)
            output_shape = self.extra_dims + keep_dims
            return tuple([input_shape[0]] + output_shape)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(Tensordot, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
