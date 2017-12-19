import tensorflow as tf
from arch import Arch
import numpy as np
from arch_aux import AuxTool
import cv2
from scipy import misc

import sys
sys.path.append('./data/vgg_net/')
from imagenet_classes import class_names

BATCH_SIZE = 1


class VGGNet(Arch):
    def __init__(self):
        self.__init_input()
        self.__init_para()
        self.__inference_layer()

    def predict(self, imgs):
        predicts = tf.nn.softmax(self.logits)
        predict = tf.argmax(predicts, 1)

        for img in imgs:
            pred = self.sess.run(fetches=predict, feed_dict={self.imgs: img})
            print 'image class: {}'.format(class_names[pred[0]])
            AuxTool.show_img(np.reshape(img, [224, 224, 3]))

    def __enter__(self):
        self.sess = tf.Session()
        self.__load_para('./data/vgg_net/vgg16_weights.npz')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def __load_para(self, path):
        paras = np.load(path, encoding='bytes')

        for key in paras:
            if key[-1] == 'W':
                self.sess.run(tf.assign(self.weights[key], paras[key]))
            else:
                self.sess.run(tf.assign(self.bias[key], paras[key]))

    def __init_input(self):
        self.imgs = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 224, 224, 3])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 1000])
        self.prob = tf.placeholder(dtype=tf.float32)

    def __init_para(self):
        self.weights = {
            'conv1_1_W': VGGNet.get_weights(shape=[3, 3, 3, 64], name='w1'),
            'conv1_2_W': VGGNet.get_weights(shape=[3, 3, 64, 64], name='w2'),

            'conv2_1_W': VGGNet.get_weights(shape=[3, 3, 64, 128], name='w3'),
            'conv2_2_W': VGGNet.get_weights(shape=[3, 3, 128, 128], name='w4'),

            'conv3_1_W': VGGNet.get_weights(shape=[3, 3, 128, 256], name='w5'),
            'conv3_2_W': VGGNet.get_weights(shape=[3, 3, 256, 256], name='w6'),
            'conv3_3_W': VGGNet.get_weights(shape=[3, 3, 256, 256], name='w7'),

            'conv4_1_W': VGGNet.get_weights(shape=[3, 3, 256, 512], name='w8'),
            'conv4_2_W': VGGNet.get_weights(shape=[3, 3, 512, 512], name='w9'),
            'conv4_3_W': VGGNet.get_weights(shape=[3, 3, 512, 512], name='w10'),

            'conv5_1_W': VGGNet.get_weights(shape=[3, 3, 512, 512], name='w11'),
            'conv5_2_W': VGGNet.get_weights(shape=[3, 3, 512, 512], name='w12'),
            'conv5_3_W': VGGNet.get_weights(shape=[3, 3, 512, 512], name='w13'),

            'fc6_W': None,
            'fc7_W': VGGNet.get_weights(shape=[4096, 4096], name='w15'),
            'fc8_W': VGGNet.get_weights(shape=[4096, 1000], name='w16')
        }

        self.bias = {
            'conv1_1_b': VGGNet.get_bias(shape=[64], name='b1'),
            'conv1_2_b': VGGNet.get_bias(shape=[64], name='b2'),

            'conv2_1_b': VGGNet.get_bias(shape=[128], name='b3'),
            'conv2_2_b': VGGNet.get_bias(shape=[128], name='b4'),

            'conv3_1_b': VGGNet.get_bias(shape=[256], name='b5'),
            'conv3_2_b': VGGNet.get_bias(shape=[256], name='b6'),
            'conv3_3_b': VGGNet.get_bias(shape=[256], name='b7'),

            'conv4_1_b': VGGNet.get_bias(shape=[512], name='b8'),
            'conv4_2_b': VGGNet.get_bias(shape=[512], name='b9'),
            'conv4_3_b': VGGNet.get_bias(shape=[512], name='b10'),

            'conv5_1_b': VGGNet.get_bias(shape=[512], name='b11'),
            'conv5_2_b': VGGNet.get_bias(shape=[512], name='b12'),
            'conv5_3_b': VGGNet.get_bias(shape=[512], name='b13'),

            'fc6_b': VGGNet.get_bias(shape=[4096], name='b14'),
            'fc7_b': VGGNet.get_bias(shape=[4096], name='b15'),
            'fc8_b': VGGNet.get_bias(shape=[1000], name='b16')
        }

    # see the architecture in http://ww1.sinaimg.cn/large/89a72506gy1flkuvombu1j20iv0j7acp.jpg
    def __inference_layer(self):
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        data = self.imgs - mean

        layer1 = VGGNet.get_conv_relu(data, self.weights['conv1_1_W'], self.bias['conv1_1_b'])
        layer2 = VGGNet.get_conv_relu(layer1, self.weights['conv1_2_W'], self.bias['conv1_2_b'])
        pool1 = VGGNet.get_pool(layer2)

        layer3 = VGGNet.get_conv_relu(pool1, self.weights['conv2_1_W'], self.bias['conv2_1_b'])
        layer4 = VGGNet.get_conv_relu(layer3, self.weights['conv2_2_W'], self.bias['conv2_2_b'])
        pool2 = VGGNet.get_pool(layer4)

        layer5 = VGGNet.get_conv_relu(pool2, self.weights['conv3_1_W'], self.bias['conv3_1_b'])
        layer6 = VGGNet.get_conv_relu(layer5, self.weights['conv3_2_W'], self.bias['conv3_2_b'])
        layer7 = VGGNet.get_conv_relu(layer6, self.weights['conv3_3_W'], self.bias['conv3_3_b'])
        pool3 = VGGNet.get_pool(layer7)

        layer8 = VGGNet.get_conv_relu(pool3, self.weights['conv4_1_W'], self.bias['conv4_1_b'])
        layer9 = VGGNet.get_conv_relu(layer8, self.weights['conv4_2_W'], self.bias['conv4_2_b'])
        layer10 = VGGNet.get_conv_relu(layer9, self.weights['conv4_3_W'], self.bias['conv4_3_b'])
        pool4 = VGGNet.get_pool(layer10)

        layer11 = VGGNet.get_conv_relu(pool4, self.weights['conv5_1_W'], self.bias['conv5_1_b'])
        layer12 = VGGNet.get_conv_relu(layer11, self.weights['conv5_2_W'], self.bias['conv5_2_b'])
        layer13 = VGGNet.get_conv_relu(layer12, self.weights['conv5_3_W'], self.bias['conv5_3_b'])
        pool5 = VGGNet.get_pool(layer13)

        data = tf.reshape(pool5, shape=[BATCH_SIZE, -1])
        self.weights['fc6_W'] = VGGNet.get_weights(shape=[data.shape[1].value, 4096], name='w14')
        layer14 = VGGNet.get_fc(data, self.weights['fc6_W'], self.bias['fc6_b'])
        layer14 = tf.nn.relu(layer14)
        layer15 = VGGNet.get_fc(layer14, self.weights['fc7_W'], self.bias['fc7_b'])
        layer15 = tf.nn.relu(layer15)
        self.logits = VGGNet.get_fc(layer15, self.weights['fc8_W'], self.bias['fc8_b'])


def main():
    img = misc.imread('./data/alex_net/laska.png', mode='RGB')
    img = misc.imresize(img, (224, 224))

    with VGGNet() as vgg:
        vgg.predict([[img]])


if __name__ == '__main__':
    main()