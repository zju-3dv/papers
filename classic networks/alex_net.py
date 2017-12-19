import tensorflow as tf
from arch_aux import AuxTool

BATCH_SIZE = 128


class AlexNet(object):
    def __init__(self):
        self.__init_input()
        self.__init_weight()
        self.__inference_layer()
        self.__loss_layer()
        self.__optimizer_layer()
        self.summary = AuxTool.create_summary(self.loss)
        self.saver = AuxTool.create_saver()

    def __init_input(self):
        self.img = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 224, 224, 3])
        self.label = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 1000])
        self.prob = tf.placeholder(dtype=tf.float32)
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

    def __init_weight(self):
        self.weights = {
            'wc1': AlexNet.get_weights(shape=[11, 11, 3, 96], name='wconv1'),
            'wc2': AlexNet.get_weights(shape=[5, 5, 96, 256], name='wconv2'),
            'wc3': AlexNet.get_weights(shape=[3, 3, 256, 384], name='wconv3'),
            'wc4': AlexNet.get_weights(shape=[3, 3, 384, 384], name='wconv4'),
            'wc5': AlexNet.get_weights(shape=[3, 3, 384, 256], name='wconv5'),
            'wf6': AlexNet.get_weights(shape=[6 * 6 * 256, 4096], name='wfc6'),
            'wf7': AlexNet.get_weights(shape=[4096, 4096], name='wfc7'),
            'wf8': AlexNet.get_weights(shape=[4096, 1000], name='wfc8')
        }

        self.bias = {
            'bc1': AlexNet.get_bias(shape=[96], name='bconv1'),
            'bc2': AlexNet.get_bias(shape=[256], name='bconv2'),
            'bc3': AlexNet.get_bias(shape=[384], name='bconv3'),
            'bc4': AlexNet.get_bias(shape=[384], name='bconv4'),
            'bc5': AlexNet.get_bias(shape=[256], name='bconv5'),
            'bf6': AlexNet.get_bias(shape=[4096], name='bfc6'),
            'bf7': AlexNet.get_bias(shape=[4096], name='bfc7'),
            'bf8': AlexNet.get_bias(shape=[1000], name='bfc8')
        }

    # conv, relu, lrn, max pool
    def __layer1(self):
        with tf.name_scope('layer1'):
            data = self.img  # [batch_size, 224, 224, 3]
            conv = AlexNet.get_conv(data, self.weights['wc1'], x_step=4, y_step=4)
            relu = tf.nn.relu(conv + self.bias['bc1'])
            lrn = AlexNet.get_lrn(relu)
            self.layer1 = AlexNet.get_pool(lrn, x_size=3, y_size=3, padding='VALID')

    # conv, relu, lrn, max pool
    def __layer2(self):
        with tf.name_scope('layer2'):
            data = self.layer1  # [batch_size, 27, 27, 96]
            conv = AlexNet.get_conv(data, self.weights['wc2'])
            relu = tf.nn.relu(conv + self.bias['bc2'])
            lrn = AlexNet.get_lrn(relu)
            self.layer2 = AlexNet.get_pool(lrn, x_size=3, y_size=3, padding='VALID')

    # conv, relu
    def __layer3(self):
        with tf.name_scope('layer3'):
            data = self.layer2  # [batch_size, 13, 13, 256]
            conv = AlexNet.get_conv(data, self.weights['wc3'])
            self.layer3 = tf.nn.relu(conv + self.bias['bc3'])

    # conv, relu
    def __layer4(self):
        with tf.name_scope('layer4'):
            data = self.layer3  # [batch_size, 13, 13, 384]
            conv = AlexNet.get_conv(data, self.weights['wc4'])
            self.layer4 = tf.nn.relu(conv + self.bias['bc4'])

    # conv, relu, max pool
    def __layer5(self):
        with tf.name_scope('layer5'):
            data = self.layer4 # [batch_size, 13, 13, 384]
            conv = AlexNet.get_conv(data, self.weights['wc5'])
            relu = tf.nn.relu(conv + self.bias['bc5'])
            self.layer5 = AlexNet.get_pool(relu, x_size=3, y_size=3, padding='VALID')

    # fc, dropout
    def __layer6(self):
        with tf.name_scope('layer6'):
            data = self.layer5 # [batch_size, 6, 6, 256]
            data = tf.reshape(data, shape=[BATCH_SIZE, -1])
            fc = tf.matmul(data, self.weights['wf6'])
            self.layer6 = tf.nn.dropout(fc + self.bias['bf6'], self.prob)

    # fc, dropout
    def __layer7(self):
        with tf.name_scope('layer7'):
            data = self.layer6 # [batch_size, 4096]
            fc = tf.matmul(data, self.weights['wf7'])
            self.layer7 = tf.nn.dropout(fc + self.bias['bf7'], self.prob)

    # fc, logits
    def __layer8(self):
        with tf.name_scope('layer8'):
            data = self.layer7 # [batch_size, 4096]
            fc = tf.matmul(data, self.weights['wf8'])
            self.logits = fc + self.bias['bf8']

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

    # loss layer
    def __loss_layer(self):
        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label,
                                                           logits=self.logits)
            self.loss = tf.reduce_mean(loss)

    # optimizer layer
    def __optimizer_layer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=self.loss,
                                                                                   global_step=self.global_step)

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
    def get_pool(data, x_size=2, y_size=2, x_step=2, y_step=2, padding='SAME'):
        return tf.nn.max_pool(value=data,
                              ksize=[1, x_size, y_size, 1],
                              strides=[1, x_step, y_step, 1],
                              padding=padding)

    @staticmethod
    def get_lrn(data):
        return tf.nn.lrn(data, 4, 2, 0.0001/9, 0.75)


def main():
    alex_net = AlexNet()


if __name__ == '__main__':
    main()