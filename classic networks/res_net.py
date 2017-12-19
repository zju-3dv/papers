import tensorflow as tf
import tensorflow.contrib.slim as slim
from arch import Arch
from arch_aux import AuxTool
import sys
sys.path.append('./data/res_net')
from imagenet_classes import class_names

BATCH_SIZE = 1
NUM_CLASSES = 1000
IMAGENET_MEAN_BGR = [103.062623801, 115.902882574, 123.151630838]


class ResNet(Arch):
    def __init__(self):
        self.__init_input()
        self.__inference_layer()
        self.saver = AuxTool.create_saver()

    def __enter__(self):
        self.sess = tf.Session()
        AuxTool.restore_sess(self.saver, self.sess, None, './data/res_net/ResNet-L50')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.sess.close()

    def __init_input(self):
        self.imgs = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 224, 224, 3])
        self.labels = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 1000])

    def __scale1_layer(self):
        data = self.imgs
        with tf.variable_scope('scale1'):
            data = ResNet.conv2d_layer(data, 64, kernel_size=[7, 7], strides=2, bias=False)
            self.scale1 = slim.max_pool2d(data, kernel_size=[3, 3], stride=2, padding='SAME')

    def __scale2_layer(self):
        data = self.scale1
        with tf.variable_scope('scale2'):
            with tf.variable_scope('block1'):
                with tf.variable_scope('shortcut'):
                    bran1 = ResNet.conv2d_layer(data, 256, kernel_size=[1, 1], strides=1, relu_fn=None)
                bran2 = ResNet.res_3_layer(data)
                data = tf.nn.relu(bran1 + bran2)

            with tf.variable_scope('block2'):
                bran2 = ResNet.res_3_layer(data)
                data = tf.nn.relu(data + bran2)

            with tf.variable_scope('block3'):
                bran2 = ResNet.res_3_layer(data)
                self.scale2 = tf.nn.relu(data + bran2)

    def __scale3_layer(self):
        data = self.scale2
        with tf.variable_scope('scale3'):
            with tf.variable_scope('block1'):
                with tf.variable_scope('shortcut'):
                    bran1 = ResNet.conv2d_layer(data, 512, kernel_size=[1, 1], strides=2, relu_fn=None)
                bran2 = ResNet.res_3_layer(data, 128, 128, 512, stride1=2)
                data = tf.nn.relu(bran1 + bran2)

            with tf.variable_scope('block2'):
                bran2 = ResNet.res_3_layer(data, 128, 128, 512)
                data = tf.nn.relu(data + bran2)

            with tf.variable_scope('block3'):
                bran2 = ResNet.res_3_layer(data, 128, 128, 512)
                data = tf.nn.relu(data + bran2)

            with tf.variable_scope('block4'):
                bran2 = ResNet.res_3_layer(data, 128, 128, 512)
                self.scale3 = tf.nn.relu(data + bran2)

    def __scale4_layer(self):
        data = self.scale3
        with tf.variable_scope('scale4'):
            with tf.variable_scope('block1'):
                with tf.variable_scope('shortcut'):
                    bran1 = ResNet.conv2d_layer(data, 1024, kernel_size=[1, 1], strides=2, relu_fn=None)
                bran2 = ResNet.res_3_layer(data, 256, 256, 1024, stride1=2)
                data = tf.nn.relu(bran1 + bran2)

            with tf.variable_scope('block2'):
                bran2 = ResNet.res_3_layer(data, 256, 256, 1024)
                data = tf.nn.relu(data + bran2)

            with tf.variable_scope('block3'):
                bran2 = ResNet.res_3_layer(data, 256, 256, 1024)
                data = tf.nn.relu(data + bran2)

            with tf.variable_scope('block4'):
                bran2 = ResNet.res_3_layer(data, 256, 256, 1024)
                data = tf.nn.relu(data + bran2)

            with tf.variable_scope('block5'):
                bran2 = ResNet.res_3_layer(data, 256, 256, 1024)
                data = tf.nn.relu(data + bran2)

            with tf.variable_scope('block6'):
                bran2 = ResNet.res_3_layer(data, 256, 256, 1024)
                self.scale4 = tf.nn.relu(data + bran2)

    def __scale5_layer(self):
        data = self.scale4
        with tf.variable_scope('scale5'):
            with tf.variable_scope('block1'):
                with tf.variable_scope('shortcut'):
                    bran1 = ResNet.conv2d_layer(data, 2048, kernel_size=[1, 1], strides=2, relu_fn=None)
                bran2 = ResNet.res_3_layer(data, 512, 512, 2048, stride1=2)
                data = tf.nn.relu(bran1 + bran2)

            with tf.variable_scope('block2'):
                bran2 = ResNet.res_3_layer(data, 512, 512, 2048)
                data = tf.nn.relu(data + bran2)

            with tf.variable_scope('block3'):
                bran2 = ResNet.res_3_layer(data, 512, 512, 2048)
                self.scale5 = tf.nn.relu(data + bran2)

    def __logits_layer(self):
        data = slim.avg_pool2d(self.scale5, kernel_size=[7, 7], stride=1)
        with tf.variable_scope('fc'):
            self.logits = ResNet.fc_layer(data, NUM_CLASSES)

    def __inference_layer(self):
        with slim.arg_scope([ResNet.conv2d_layer],
                            normalizer_fn=tf.nn.batch_normalization,
                            padding='SAME'):
            self.__scale1_layer()
            self.__scale2_layer()
            self.__scale3_layer()
            self.__scale4_layer()
            self.__scale5_layer()
            self.__logits_layer()

    def predict(self, imgs):
        predicts = tf.nn.softmax(self.logits)
        predict = tf.argmax(predicts, 1)

        for img in imgs:
            pred = self.sess.run(predict, feed_dict={self.imgs: img})
            print class_names[pred[0]]

    @staticmethod
    def res_3_layer(data, out_chs1=64, out_chs2=64, out_chs3=256, stride1=1, stride2=1, stride3=1):
        with tf.variable_scope('a'):
            data = ResNet.conv2d_layer(data, out_chs1, kernel_size=[1, 1], strides=stride1, bias=False)
        with tf.variable_scope('b'):
            data = ResNet.conv2d_layer(data, out_chs2, kernel_size=[3, 3], strides=stride2, bias=False)
        with tf.variable_scope('c'):
            return ResNet.conv2d_layer(data, out_chs3, kernel_size=[1, 1], strides=stride3, relu_fn=None, bias=False)


def main():
    with ResNet() as res_net:
        pass

if __name__ == '__main__':
    main()
