import tensorflow as tf
import sys
sys.path.append('./models/tutorials/image/cifar10')
import cifar10

BATCH_SIZE = 128
TRAIN_OUT_PERIOD = 100
TEST_OUT_PERIOD = 10


class AdvancedArch(object):
    def __init__(self, data):
        self.data = data
        self.__init_input()
        self.__inference_layer()
        self.__loss_layer()
        self.__optimizer_layer()
        self.__summary()
        self.__saver()

    def train(self):
        train_data = self.data.train_data

        with tf.Session() as sess:
            init_step = self.__restore(sess)
            writer = tf.summary.FileWriter('./graph', sess.graph)
            tf.train.start_queue_runners()

            n_epochs = int(train_data.num_examples / BATCH_SIZE)
            total_loss = 0.0
            for i in range(init_step, n_epochs):
                img, label = train_data.next(BATCH_SIZE)
                _, loss, summary = sess.run(fetches=[self.optimizer, self.loss, self.summary],
                                            feed_dict={self.img: img, self.label: label})

                total_loss = total_loss + loss
                if (i + 1) % TRAIN_OUT_PERIOD == 0:
                    print "epoch: {0} loss: {1}".format(i, total_loss / TRAIN_OUT_PERIOD)
                    total_loss = 0.0
                    writer.add_summary(summary, global_step=i)
                    self.saver.save(sess, './model/advanced/', global_step=i)

    def test(self):
        test_data = self.data.test_data

        with tf.Session() as sess:
            self.__restore(sess)
            tf.train.start_queue_runners()

            predict = tf.argmax(tf.nn.softmax(self.logits), 1)
            label = tf.argmax(self.label, 1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, label), dtype=tf.float32))

            n_epochs = int(test_data.num_examples / BATCH_SIZE)
            total_acc = 0.0
            for i in range(0, n_epochs):
                img, label = test_data.next(BATCH_SIZE)
                acc = sess.run(fetches=accuracy,
                               feed_dict={self.img: img, self.label: label})

                total_acc = total_acc + acc
                if (i + 1) % TEST_OUT_PERIOD == 0:
                    print "epoch: {0} accuracy: {1}".format(i, total_acc / TEST_OUT_PERIOD)
                    total_acc = 0.0

    def __init_input(self):
        self.img = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 24, 24, 3])
        self.label = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 10])
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

    # conv, relu, max pool, lrn
    def __layer1(self):
        in_nodes = 3
        out_nodes = 64
        data = self.img

        with tf.name_scope('layer1'):
            weights = AdvancedArch.get_weights(shape=[5, 5, in_nodes, out_nodes],
                                               name='weights1')
            bias = AdvancedArch.get_bias(shape=[out_nodes],
                                         name='bias1')
            conv = AdvancedArch.get_conv2d(data, weights)
            relu = tf.nn.relu(conv + bias)
            pool = AdvancedArch.get_pool(relu)
            self.layer1_data = tf.nn.lrn(pool)

    # conv, relu, lrn, max pool
    def __layer2(self):
        data = self.layer1_data
        in_nodes = data.shape()[1].value
        out_nodes = 64

        with tf.name_scope('layer2'):
            weights = AdvancedArch.get_weights(shape=[5, 5, in_nodes, out_nodes],
                                               name='weights2')
            bias = AdvancedArch.get_bias(shape=[out_nodes],
                                         name='bias2')
            conv = AdvancedArch.get_conv2d(data, weights)
            relu = tf.nn.relu(conv + bias)
            lrn = tf.nn.lrn(relu)
            self.layer2_data = AdvancedArch.get_pool(lrn)

    # fc, relu
    def __layer3(self):
        data = tf.reshape(self.layer2_data, shape=[BATCH_SIZE, -1])
        in_nodes = data.shape[1].value
        out_nodes = 384

        with tf.name_scope('layer3'):
            weights = AdvancedArch.get_weights(shape=[in_nodes, out_nodes],
                                               name='weights3',
                                               l2_loss=0.004)
            bias = AdvancedArch.get_bias(shape=[out_nodes],
                                         name='bias3')
            fc = tf.matmul(data, weights)
            self.layer3_data = tf.nn.relu(fc + bias)

    # fc, relu
    def __layer4(self):
        data = self.layer3_data
        in_nodes = data.shape[1].value
        out_nodes = 192

        with tf.name_scope('layer4'):
            weights = AdvancedArch.get_weights(shape=[in_nodes, out_nodes],
                                               name='weight4',
                                               l2_loss=0.004)
            bias = AdvancedArch.get_bias(shape=[out_nodes],
                                         name='bias4')
            fc = tf.matmul(data, weights)
            self.layer4_data = tf.nn.relu(fc + bias)

    # fc
    def __logits_layer(self):
        data = self.layer4_data
        in_nodes = data.shape[1].value
        out_nodes = 10

        with tf.name_scope('logits'):
            weights = AdvancedArch.get_weights(shape=[in_nodes, out_nodes],
                                               name='weights5')
            bias = AdvancedArch.get_bias(shape=[out_nodes],
                                         name='bias5')
            self.logits = tf.matmul(data, weights) + bias

    # inference layer
    def __inference_layer(self):
        self.__layer1()
        self.__layer2()
        self.__layer3()
        self.__layer4()
        self.__logits_layer()

    # cross entropy, loss collection
    def __loss_layer(self):
        with tf.name_scope('loss'):
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            loss = tf.reduce_mean(loss)
            tf.add_to_collection('losses', loss)
            self.loss = tf.add_n(tf.get_collection(key='losses'))

    # optimizer
    def __optimizer_layer(self):
        with tf.name_scope('optimizer'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=self.loss,
                                                                                   global_step=self.global_step)

    # summary
    def __summary(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.histogram('histogram loss', self.loss)
        self.summary = tf.summary.merge_all()

    # saver
    def __saver(self):
        self.saver = tf.train.Saver()

    # restore
    def __restore(self, sess):
        ckpt = tf.train.get_checkpoint_state('./model/advanced/advanced_arch')

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            return 0

        return self.global_step

    @staticmethod
    def get_weights(shape, name, l2_loss=0.0):
        weights = tf.get_variable(name=name,
                                  shape=shape,
                                  initializer=tf.truncated_normal_initializer)
        if l2_loss != 0.0:
            loss = tf.nn.l2_loss(weights)
            tf.add_to_collection(name='losses', value=l2_loss*loss)

        return weights

    @staticmethod
    def get_bias(shape, name):
        return tf.get_variable(name=name,
                               shape=shape,
                               initializer=tf.truncated_normal_initializer)

    @staticmethod
    def get_conv2d(data, kernel, x_step=1, y_step=1):
        return tf.nn.conv2d(input=data,
                            filter=kernel,
                            strides=[1, x_step, y_step, 1],
                            padding='SAME')

    @staticmethod
    def get_pool(data, x_size=2, y_size=2, x_step=2, y_step=2):
        return tf.nn.max_pool(value=data,
                              ksize=[1, x_size, y_size, 1],
                              strides=[1, x_step, y_step, 1],
                              padding='SAME')


def main():
    cifar10.maybe_download_and_extract()


if __name__ == '__main__':
    main()