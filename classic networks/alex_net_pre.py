import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt
from arch_aux import AuxTool

sys.path.append('./data/alex_net')
from caffe_classes import class_names
import cv2

BATCH_SIZE = 1


class AlexNet(object):
    def __init__(self):
        self.__init_input()
        self.__init_para()
        self.__inference_layer()

    def __enter__(self):
        self.sess = tf.Session()
        self.__load_para(self.sess)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def predict(self, img):
        softmax = tf.nn.softmax(self.logits)
        predict = tf.argmax(softmax, 1)
        predict = self.sess.run(fetches=predict, feed_dict={self.imgs: img, self.prob: 1.0})
        print class_names[predict[0]]

    def __init_input(self):
        self.imgs = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 227, 227, 3])
        self.prob = tf.placeholder(dtype=tf.float32)

    def __init_para(self):
        self.weights = {
            'w1': AlexNet.get_weights(shape=[11, 11, 3, 96], name='wc1'),
            'w2': AlexNet.get_weights(shape=[5, 5, 48, 256], name='wc2'),
            'w3': AlexNet.get_weights(shape=[3, 3, 256, 384], name='wc3'),
            'w4': AlexNet.get_weights(shape=[3, 3, 192, 384], name='wc4'),
            'w5': AlexNet.get_weights(shape=[3, 3, 192, 256], name='wc5'),
            'w6': AlexNet.get_weights(shape=[9216, 4096], name='wf6'),
            'w7': AlexNet.get_weights(shape=[4096, 4096], name='wf7'),
            'w8': AlexNet.get_weights(shape=[4096, 1000], name='wf8')
        }

        self.bias = {
            'b1': AlexNet.get_bias(shape=[96], name='bc1'),
            'b2': AlexNet.get_bias(shape=[256], name='bc2'),
            'b3': AlexNet.get_bias(shape=[384], name='bc3'),
            'b4': AlexNet.get_bias(shape=[384], name='bc4'),
            'b5': AlexNet.get_bias(shape=[256], name='bc5'),
            'b6': AlexNet.get_bias(shape=[4096], name='bf6'),
            'b7': AlexNet.get_bias(shape=[4096], name='bf7'),
            'b8': AlexNet.get_bias(shape=[1000], name='bf8')
        }

    def __load_para(self, sess):
        paras = np.load('./data/alex_net/bvlc_alexnet.npy', encoding='bytes').item()

        for name in paras:
            suf = name[-1]
            sess.run(tf.assign(self.weights['w' + suf], paras[name][0]))
            sess.run(tf.assign(self.bias['b' + suf], paras[name][1]))

    # conv, relu, lrn, max pool
    def __layer1(self):
        with tf.name_scope('layer1'):
            data = self.imgs
            conv = AlexNet.get_conv(data, self.weights['w1'], x_step=4, y_step=4)
            relu = tf.nn.relu(conv + self.bias['b1'])
            relu = AlexNet.get_pool(relu)
            self.layer1 = AlexNet.get_lrn(relu)

    # conv, relu, lrn, max pool
    def __layer2(self):
        with tf.name_scope('layer2'):
            data = self.layer1
            conv = AlexNet.get_conv(data, self.weights['w2'], groups=2)
            relu = tf.nn.relu(conv + self.bias['b2'])
            relu = AlexNet.get_pool(relu)
            self.layer2 = AlexNet.get_lrn(relu)

    # conv, relu
    def __layer3(self):
        with tf.name_scope('layer3'):
            data = self.layer2
            conv = AlexNet.get_conv(data, self.weights['w3'])
            self.layer3 = tf.nn.relu(conv + self.bias['b3'])

    # conv, relu
    def __layer4(self):
        with tf.name_scope('layer4'):
            data = self.layer3
            conv = AlexNet.get_conv(data, self.weights['w4'], groups=2)
            self.layer4 = tf.nn.relu(conv + self.bias['b4'])

    # conv, relu, max pool
    def __layer5(self):
        with tf.name_scope('layer5'):
            data = self.layer4
            conv = AlexNet.get_conv(data, self.weights['w5'], groups=2)
            relu = tf.nn.relu(conv + self.bias['b5'])
            self.layer5 = AlexNet.get_pool(relu)

    # fc, dropout
    def __layer6(self):
        with tf.name_scope('layer6'):
            data = self.layer5
            data = tf.reshape(data, shape=[BATCH_SIZE, -1])
            fc = tf.matmul(data, self.weights['w6'])
            self.layer6 = tf.nn.dropout(fc + self.bias['b6'], keep_prob=self.prob)

    # fc, dropout
    def __layer7(self):
        with tf.name_scope('layer7'):
            data = self.layer6
            fc = tf.matmul(data, self.weights['w7'])
            self.layer7 = tf.nn.dropout(fc + self.bias['b7'], keep_prob=self.prob)

    # fc, logits
    def __layer8(self):
        with tf.name_scope('layer8'):
            data = self.layer7
            fc = tf.matmul(data, self.weights['w8'])
            self.logits = fc + self.bias['b8']

    # inference layer
    def __inference_layer(self):
        self.__layer1()
        self.__layer2()
        self.__layer3()
        self.__layer4()
        self.__layer5()
        self.__layer6()
        self.__layer7()
        self.__layer8()

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
    def get_conv(data, kernel, x_step=1, y_step=1, padding='SAME', groups=1):
        conv = lambda d, k: tf.nn.conv2d(input=d,
                                         filter=k,
                                         strides=[1, x_step, y_step, 1],
                                         padding=padding)
        if groups == 1:
            return conv(data, kernel)

        datas = tf.split(data, groups, axis=-1)
        kernels = tf.split(kernel, groups, axis=-1)
        rs = [conv(d, k) for (d, k) in zip(datas, kernels)]
        return tf.concat(rs, axis=-1)

    @staticmethod
    def get_pool(data, x_size=3, y_size=3, x_step=2, y_step=2, padding='VALID'):
        return tf.nn.max_pool(value=data,
                              ksize=[1, x_size, y_size, 1],
                              strides=[1, x_step, y_step, 1],
                              padding=padding)

    @staticmethod
    def get_lrn(data):
        return tf.nn.lrn(data, 2, 1.0, 2e-05, 0.75)


def main():
    img = AuxTool.read_img('./data/alex_net/quail227.JPEG')
    img = cv2.resize(img, (227, 227))
    img = np.reshape(img, [1, 227, 227, 3])
    with AlexNet() as alex_net:
        alex_net.predict(img)


if __name__ == '__main__':
    main()