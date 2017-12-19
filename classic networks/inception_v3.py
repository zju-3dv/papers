import tensorflow as tf
from arch import Arch
import tensorflow.contrib.slim as slim
from arch_aux import AuxTool


BATCH_SIZE = 1


class InceptionV3(Arch):
    def __init__(self):
        self.__init_input()
        self.__inference_layer()
        self.saver = AuxTool.create_saver()

    def __enter__(self):
        self.sess = tf.Session()
        AuxTool.restore_sess(self.saver, self.sess, self.global_step, './data/inception_v3')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def __init_input(self):
        self.imgs = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 299, 299, 3])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 1001])
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def __inference_base(self):
        data = self.imgs

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], kernel_size=[3, 3], padding='VALID'):
            conv1 = slim.conv2d(data, 32, stride=2, scope='conv0')
            conv2 = slim.conv2d(conv1, 32, stride=1, scope='conv1')
            conv3 = slim.conv2d(conv2, 64, stride=1, padding='SAME', scope='conv2')
            pool1 = slim.max_pool2d(conv3, stride=2, scope='pool1')
            conv4 = slim.conv2d(pool1, 80, kernel_size=[1, 1], stride=1, scope='conv3')
            conv5 = slim.conv2d(conv4, 192, stride=1, scope='conv4')
            self.base_layer = slim.max_pool2d(conv5, stride=2, scope='pool2')

    def __inference_inception_1(self):
        data = self.base_layer

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            # 35 x 35 x 256.
            with tf.variable_scope('mixed_35x35x256a'):
                with tf.variable_scope('branch1x1'):
                    branch1x1 = slim.conv2d(data, 64, [1, 1])
                with tf.variable_scope('branch5x5'):
                    branch5x5 = slim.conv2d(data, 48, [1, 1])
                    branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
                with tf.variable_scope('branch3x3dbl'):
                    branch3x3dbl = slim.conv2d(data, 64, [1, 1])
                    branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
                    branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
                with tf.variable_scope('branch_pool'):
                    branch_pool = slim.avg_pool2d(data, [3, 3])
                    branch_pool = slim.conv2d(branch_pool, 32, [1, 1])
                data = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])

            # 35 x 35 x 288.
            with tf.variable_scope('mixed_35x35x288a'):
                with tf.variable_scope('branch1x1'):
                    branch1x1 = slim.conv2d(data, 64, [1, 1])
                with tf.variable_scope('branch5x5'):
                    branch5x5 = slim.conv2d(data, 48, [1, 1])
                    branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
                with tf.variable_scope('branch3x3dbl'):
                    branch3x3dbl = slim.conv2d(data, 64, [1, 1])
                    branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
                    branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
                with tf.variable_scope('branch_pool'):
                    branch_pool = slim.avg_pool2d(data, [3, 3])
                    branch_pool = slim.conv2d(branch_pool, 64, [1, 1])
                data = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])

            # 35 x 35 x 288.
            with tf.variable_scope('mixed_35x35x288b'):
                with tf.variable_scope('branch1x1'):
                    branch1x1 = slim.conv2d(data, 64, [1, 1])
                with tf.variable_scope('branch5x5'):
                    branch5x5 = slim.conv2d(data, 48, [1, 1])
                    branch5x5 = slim.conv2d(branch5x5, 64, [5, 5])
                with tf.variable_scope('branch3x3dbl'):
                    branch3x3dbl = slim.conv2d(data, 64, [1, 1])
                    branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
                    branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
                with tf.variable_scope('branch_pool'):
                    branch_pool = slim.avg_pool2d(data, [3, 3])
                    branch_pool = slim.conv2d(branch_pool, 64, [1, 1])
                self.incep_layer1 = tf.concat(axis=3, values=[branch1x1, branch5x5, branch3x3dbl, branch_pool])

    def __inference_inception_2(self):
        data = self.incep_layer1

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            # 17 x 17 x 768.
            with tf.variable_scope('mixed_17x17x768a'):
                with tf.variable_scope('branch3x3'):
                    branch3x3 = slim.conv2d(data, 384, [3, 3], stride=2, padding='VALID')
                with tf.variable_scope('branch3x3dbl'):
                    branch3x3dbl = slim.conv2d(data, 64, [1, 1])
                    branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3])
                    branch3x3dbl = slim.conv2d(branch3x3dbl, 96, [3, 3],
                                               stride=2, padding='VALID')
                with tf.variable_scope('branch_pool'):
                    branch_pool = slim.max_pool2d(data, [3, 3], stride=2, padding='VALID')
                data = tf.concat(axis=3, values=[branch3x3, branch3x3dbl, branch_pool])

            # 17 x 17 x 768.
            with tf.variable_scope('mixed_17x17x768b'):
                with tf.variable_scope('branch1x1'):
                    branch1x1 = slim.conv2d(data, 192, [1, 1])
                with tf.variable_scope('branch7x7'):
                    branch7x7 = slim.conv2d(data, 128, [1, 1])
                    branch7x7 = slim.conv2d(branch7x7, 128, [1, 7])
                    branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
                with tf.variable_scope('branch7x7dbl'):
                    branch7x7dbl = slim.conv2d(data, 128, [1, 1])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [7, 1])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [1, 7])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 128, [7, 1])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
                with tf.variable_scope('branch_pool'):
                    branch_pool = slim.avg_pool2d(data, [3, 3])
                    branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
                data = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])

            # 17 x 17 x 768.
            with tf.variable_scope('mixed_17x17x768c'):
                with tf.variable_scope('branch1x1'):
                    branch1x1 = slim.conv2d(data, 192, [1, 1])
                with tf.variable_scope('branch7x7'):
                    branch7x7 = slim.conv2d(data, 160, [1, 1])
                    branch7x7 = slim.conv2d(branch7x7, 160, [1, 7])
                    branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
                with tf.variable_scope('branch7x7dbl'):
                    branch7x7dbl = slim.conv2d(data, 160, [1, 1])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [1, 7])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
                with tf.variable_scope('branch_pool'):
                    branch_pool = slim.avg_pool2d(data, [3, 3])
                    branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
                data = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])

            # 17 x 17 x 768.
            with tf.variable_scope('mixed_17x17x768d'):
                with tf.variable_scope('branch1x1'):
                    branch1x1 = slim.conv2d(data, 192, [1, 1])
                with tf.variable_scope('branch7x7'):
                    branch7x7 = slim.conv2d(data, 160, [1, 1])
                    branch7x7 = slim.conv2d(branch7x7, 160, [1, 7])
                    branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
                with tf.variable_scope('branch7x7dbl'):
                    branch7x7dbl = slim.conv2d(data, 160, [1, 1])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [1, 7])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 160, [7, 1])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
                with tf.variable_scope('branch_pool'):
                    branch_pool = slim.avg_pool2d(data, [3, 3])
                    branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
                data = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])

            # 17 x 17 x 768.
            with tf.variable_scope('mixed_17x17x768e'):
                with tf.variable_scope('branch1x1'):
                    branch1x1 = slim.conv2d(data, 192, [1, 1])
                with tf.variable_scope('branch7x7'):
                    branch7x7 = slim.conv2d(data, 192, [1, 1])
                    branch7x7 = slim.conv2d(branch7x7, 192, [1, 7])
                    branch7x7 = slim.conv2d(branch7x7, 192, [7, 1])
                with tf.variable_scope('branch7x7dbl'):
                    branch7x7dbl = slim.conv2d(data, 192, [1, 1])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [7, 1])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [7, 1])
                    branch7x7dbl = slim.conv2d(branch7x7dbl, 192, [1, 7])
                with tf.variable_scope('branch_pool'):
                    branch_pool = slim.avg_pool2d(data, [3, 3])
                    branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
                self.incep_layer2 = tf.concat(axis=3, values=[branch1x1, branch7x7, branch7x7dbl, branch_pool])

    def __auxiliary_logits(self):
        data = self.incep_layer2
        data = tf.identity(data)

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope('aux_logits'):
                aux_logits = slim.avg_pool2d(data, [5, 5], stride=3, padding='VALID')
                aux_logits = slim.conv2d(aux_logits, 128, [1, 1], scope='proj')
                # Shape of feature map before the final layer.
                shape = aux_logits.get_shape()
                aux_logits = slim.conv2d(aux_logits, 768, shape[1:3],
                                         weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         padding='VALID')
                aux_logits = slim.flatten(aux_logits)
                self.aux_logits = slim.fully_connected(
                    aux_logits, 1001, normalizer_fn=None,
                    weights_initializer=tf.truncated_normal_initializer(stddev=0.001),
                    scope='FC'
                )

    def __inference_inception_3(self):
        data = self.incep_layer2

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            # 8 x 8 x 1280.
            with tf.variable_scope('mixed_17x17x1280a'):
                with tf.variable_scope('branch3x3'):
                    branch3x3 = slim.conv2d(data, 192, [1, 1])
                    branch3x3 = slim.conv2d(branch3x3, 320, [3, 3], stride=2, padding='VALID')
                with tf.variable_scope('branch7x7x3'):
                    branch7x7x3 = slim.conv2d(data, 192, [1, 1])
                    branch7x7x3 = slim.conv2d(branch7x7x3, 192, [1, 7])
                    branch7x7x3 = slim.conv2d(branch7x7x3, 192, [7, 1])
                    branch7x7x3 = slim.conv2d(branch7x7x3, 192, [3, 3], stride=2, padding='VALID')
                with tf.variable_scope('branch_pool'):
                    branch_pool = slim.max_pool2d(data, [3, 3], stride=2, padding='VALID')
                data = tf.concat(axis=3, values=[branch3x3, branch7x7x3, branch_pool])

            # 8 x 8 x 2048.
            with tf.variable_scope('mixed_8x8x2048a'):
                with tf.variable_scope('branch1x1'):
                    branch1x1 = slim.conv2d(data, 320, [1, 1])
                with tf.variable_scope('branch3x3'):
                    branch3x3 = slim.conv2d(data, 384, [1, 1])
                    branch3x3 = tf.concat(axis=3, values=[slim.conv2d(branch3x3, 384, [1, 3]),
                                                          slim.conv2d(branch3x3, 384, [3, 1])])
                with tf.variable_scope('branch3x3dbl'):
                    branch3x3dbl = slim.conv2d(data, 448, [1, 1])
                    branch3x3dbl = slim.conv2d(branch3x3dbl, 384, [3, 3])
                    branch3x3dbl = tf.concat(axis=3, values=[slim.conv2d(branch3x3dbl, 384, [1, 3]),
                                                             slim.conv2d(branch3x3dbl, 384, [3, 1])])
                with tf.variable_scope('branch_pool'):
                    branch_pool = slim.avg_pool2d(data, [3, 3])
                    branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
                data = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])

            # 8 x 8 x 2048.
            with tf.variable_scope('mixed_8x8x2048b'):
                with tf.variable_scope('branch1x1'):
                    branch1x1 = slim.conv2d(data, 320, [1, 1])
                with tf.variable_scope('branch3x3'):
                    branch3x3 = slim.conv2d(data, 384, [1, 1])
                    branch3x3 = tf.concat(axis=3, values=[slim.conv2d(branch3x3, 384, [1, 3]),
                                                          slim.conv2d(branch3x3, 384, [3, 1])])
                with tf.variable_scope('branch3x3dbl'):
                    branch3x3dbl = slim.conv2d(data, 448, [1, 1])
                    branch3x3dbl = slim.conv2d(branch3x3dbl, 384, [3, 3])
                    branch3x3dbl = tf.concat(axis=3, values=[slim.conv2d(branch3x3dbl, 384, [1, 3]),
                                                             slim.conv2d(branch3x3dbl, 384, [3, 1])])
                with tf.variable_scope('branch_pool'):
                    branch_pool = slim.avg_pool2d(data, [3, 3])
                    branch_pool = slim.conv2d(branch_pool, 192, [1, 1])
                self.incep_layer3 = tf.concat(axis=3, values=[branch1x1, branch3x3, branch3x3dbl, branch_pool])

    def __predict(self):
        data = self.incep_layer3

        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
            with tf.variable_scope('logits'):
                shape = data.get_shape()
                data = slim.avg_pool2d(data, shape[1:3], padding='VALID', scope='pool')
                # 1 x 1 x 2048
                data = slim.dropout(data, 0.8, scope='dropout')
                data = slim.flatten(data, scope='flatten')
                # 1001
                self.logits = slim.fully_connected(data, 1001, activation_fn=None, scope='logits')
                self.predicts = tf.nn.softmax(self.logits, name='predictions')

    def __inference_layer(self):
        normalizer_params = {
            'decay': 0.9999,
            'epsilon': 0.001,
            'variables_collections': {
                'beta': None,
                'moving_mean': ['moving_vars'],
                'moving_variance': ['moving_vars']
            }
        }

        with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_regularizer=slim.l2_regularizer(0.0004)):
            with slim.arg_scope([slim.conv2d],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params=normalizer_params):
                self.__inference_base()
                self.__inference_inception_1()
                self.__inference_inception_2()
                self.__auxiliary_logits()
                self.__inference_inception_3()
                self.__predict()


def main():
    with InceptionV3() as inception:
        pass

if __name__ == '__main__':
    main()
