import tensorflow as tf
from arch import Arch


BATCH_SIZE = 1


class VGGNet(Arch):
    def __init__(self):
        self.__init_input()
        self.__init_para()
        self.__inference_layer()
        self.__loss_layer()
        self.__optimizer_layer()

    def __init_input(self):
        self.imgs = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 224, 224, 3])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 1000])
        self.prob = tf.placeholder(dtype=tf.float32)

    def __init_para(self):
        self.weights = {
            '1': VGGNet.get_weights(shape=[3, 3, 3, 64], name='w1'),
            '2': VGGNet.get_weights(shape=[3, 3, 64, 64], name='w2'),

            '3': VGGNet.get_weights(shape=[3, 3, 64, 128], name='w3'),
            '4': VGGNet.get_weights(shape=[3, 3, 128, 128], name='w4'),

            '5': VGGNet.get_weights(shape=[3, 3, 128, 256], name='w5'),
            '6': VGGNet.get_weights(shape=[3, 3, 256, 256], name='w6'),
            '7': VGGNet.get_weights(shape=[3, 3, 256, 256], name='w7'),

            '8': VGGNet.get_weights(shape=[3, 3, 256, 512], name='w8'),
            '9': VGGNet.get_weights(shape=[3, 3, 512, 512], name='w9'),
            '10': VGGNet.get_weights(shape=[3, 3, 512, 512], name='w10'),

            '11': VGGNet.get_weights(shape=[3, 3, 512, 512], name='w11'),
            '12': VGGNet.get_weights(shape=[3, 3, 512, 512], name='w12'),
            '13': VGGNet.get_weights(shape=[3, 3, 512, 512], name='w13'),

            '14': VGGNet.get_weights(shape=[7 * 7 * 512, 4096], name='w14'),
            '15': VGGNet.get_weights(shape=[4096, 4096], name='w15'),
            '16': VGGNet.get_weights(shape=[4096, 1000], name='w16')
        }

        self.bias = {
            '1': VGGNet.get_bias(shape=[64], name='b1'),
            '2': VGGNet.get_bias(shape=[64], name='b2'),

            '3': VGGNet.get_bias(shape=[128], name='b3'),
            '4': VGGNet.get_bias(shape=[128], name='b4'),

            '5': VGGNet.get_bias(shape=[256], name='b5'),
            '6': VGGNet.get_bias(shape=[256], name='b6'),
            '7': VGGNet.get_bias(shape=[256], name='b7'),

            '8': VGGNet.get_bias(shape=[512], name='b8'),
            '9': VGGNet.get_bias(shape=[512], name='b9'),
            '10': VGGNet.get_bias(shape=[512], name='b10'),

            '11': VGGNet.get_bias(shape=[512], name='b11'),
            '12': VGGNet.get_bias(shape=[512], name='b12'),
            '13': VGGNet.get_bias(shape=[512], name='b13'),

            '14': VGGNet.get_bias(shape=[4096], name='b14'),
            '15': VGGNet.get_bias(shape=[4096], name='b15'),
            '16': VGGNet.get_bias(shape=[1000], name='b16')
        }

    # see the architecture in http://ww1.sinaimg.cn/large/89a72506gy1flkuvombu1j20iv0j7acp.jpg
    def __inference_layer(self):
        data = self.imgs

        layer1 = VGGNet.get_conv_relu(data, self.weights['1'], self.bias['1'])
        layer2 = VGGNet.get_conv_relu(layer1, self.weights['2'], self.bias['2'])
        pool1 = VGGNet.get_pool(layer2)

        layer3 = VGGNet.get_conv_relu(pool1, self.weights['3'], self.bias['3'])
        layer4 = VGGNet.get_conv_relu(layer3, self.weights['4'], self.bias['4'])
        pool2 = VGGNet.get_pool(layer4)

        layer5 = VGGNet.get_conv_relu(pool2, self.weights['5'], self.bias['5'])
        layer6 = VGGNet.get_conv_relu(layer5, self.weights['6'], self.bias['6'])
        layer7 = VGGNet.get_conv_relu(layer6, self.weights['7'], self.bias['7'])
        pool3 = VGGNet.get_pool(layer7)

        layer8 = VGGNet.get_conv_relu(pool3, self.weights['8'], self.bias['8'])
        layer9 = VGGNet.get_conv_relu(layer8, self.weights['9'], self.bias['9'])
        layer10 = VGGNet.get_conv_relu(layer9, self.weights['10'], self.bias['10'])
        pool4 = VGGNet.get_pool(layer10)

        layer11 = VGGNet.get_conv_relu(pool4, self.weights['11'], self.bias['11'])
        layer12 = VGGNet.get_conv_relu(layer11, self.weights['12'], self.bias['12'])
        layer13 = VGGNet.get_conv_relu(layer12, self.weights['13'], self.bias['13'])
        pool5 = VGGNet.get_pool(layer13)

        data = tf.reshape(pool5, shape=[BATCH_SIZE, -1])
        layer14 = VGGNet.get_fc_dropout(data, self.weights['14'], self.bias['14'], self.prob)
        layer15 = VGGNet.get_fc_dropout(layer14, self.weights['15'], self.bias['15'], self.prob)
        self.logits = VGGNet.get_fc(layer15, self.weights['16'], self.bias['16'])

    def __loss_layer(self):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
        self.loss = tf.reduce_mean(loss)

    def __optimizer_layer(self):
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)


def main():
    pass


if __name__ == '__main__':
    main()