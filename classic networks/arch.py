import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.contrib.framework.python.ops import add_arg_scope


class Arch(object):
    def predict(self, imgs):
        pass

    def __init_input(self):
        pass

    def __init_para(self):
        pass

    def __load_para(self, path):
        pass

    def __inference_layer(self):
        pass

    @staticmethod
    def get_variable(shape, name, initializer=tf.truncated_normal_initializer, trainable=True):
        return tf.get_variable(shape=shape,
                               name=name,
                               initializer=initializer,
                               trainable=trainable)

    @staticmethod
    def get_weights(shape, name):
        return tf.get_variable(shape=shape,
                               name=name,
                               initializer=tf.truncated_normal_initializer)

    @staticmethod
    def get_bias(shape, name):
        return tf.get_variable(shape=shape,
                               name=name,
                               initializer=tf.truncated_normal_initializer)

    @staticmethod
    def get_conv(data, kernel, x_step=1, y_step=1, padding='SAME'):
        return tf.nn.conv2d(input=data,
                            filter=kernel,
                            strides=[1, x_step, y_step, 1],
                            padding=padding)

    @staticmethod
    def get_conv_relu(data, kernel, bias, x_step=1, y_step=1, padding='SAME'):
        return tf.nn.relu(Arch.get_conv(data, kernel, x_step, y_step, padding) + bias)

    @staticmethod
    def get_pool(data, x_size=2, y_size=2, x_step=2, y_step=2, padding='SAME'):
        return tf.nn.max_pool(value=data,
                              ksize=[1, x_size, y_size, 1],
                              strides=[1, x_step, y_step, 1],
                              padding=padding)

    @staticmethod
    def get_fc(data, weights, bias):
        return tf.matmul(data, weights) + bias

    @staticmethod
    def get_fc_dropout(data, weights, bias, prob):
        return tf.nn.dropout(tf.matmul(data, weights) + bias, prob)

    @staticmethod
    def get_lrn(data):
        return tf.nn.lrn(data, 4, 2, 0.0001 / 9, 0.75)

    @staticmethod
    @add_arg_scope
    def fc_layer(data, out_chs, relu_fn=tf.nn.relu, batch_size=1):
        data = tf.reshape(data, shape=[batch_size, -1])
        input_chs = data.get_shape()[-1].value
        weights = Arch.get_variable(name='weights', shape=[input_chs, out_chs])
        bias = Arch.get_weights(name='biases', shape=[out_chs])
        data = tf.matmul(data, weights)

        if relu_fn is None:
            return data + bias

        return relu_fn(data + bias)

    @staticmethod
    @add_arg_scope
    def conv2d_layer(data, out_chs, kernel_size, strides, padding, relu_fn=tf.nn.relu, normalizer_fn=None, bias=False):
        input_chs = data.get_shape()[-1].value

        weights = Arch.get_weights(name='weights', shape=kernel_size + [input_chs, out_chs])
        data = tf.nn.conv2d(data, weights, strides=[1, strides, strides, 1], padding=padding)

        if normalizer_fn is not None:
            moving_mean = Arch.get_variable(name='moving_mean', shape=[out_chs], trainable=False)
            moving_variance = Arch.get_variable(name='moving_variance', shape=[out_chs], trainable=False)
            mean, variance = tf.nn.moments(data, axes=range(len(data.get_shape()) - 1))
            moving_averages.assign_moving_average(moving_mean, mean, 0.9997, zero_debias=False)
            moving_averages.assign_moving_average(moving_variance, variance, 0.9997, zero_debias=False)

            beta = Arch.get_variable(name='beta', shape=[out_chs])
            gamma = Arch.get_variable(name='gamma', shape=[out_chs])

            data = normalizer_fn(data, mean, variance, beta, gamma, variance_epsilon=0.001)

        if relu_fn is None:
            return data

        if bias:
            bias = Arch.get_weights(name='biases', shape=[out_chs])
            relu = relu_fn(data + bias)
        else:
            relu = relu_fn(data)

        return relu
