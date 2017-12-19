import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

BATCH_SIZE = 128
TRAIN_OUT_PERIOD = 100
TEST_OUT_PERIOD = 10


class SimpleArch(object):
    def __init__(self, data):
        self.data = data
        self.__init_input()
        self.__layer1()
        self.__layer2()
        self.__layer3()
        self.__layer4()
        self.__summary()
        self.__saver()

    def train(self):
        train_data = self.data.train

        with tf.Session() as sess:
            initial_step = self.__restore(sess)
            writer = tf.summary.FileWriter('./graph', sess.graph)

            n_epochs = int(train_data.num_examples / BATCH_SIZE)
            total_loss = 0.0
            for i in range(initial_step, n_epochs):
                img, label = train_data.next_batch(BATCH_SIZE)
                _, loss, summary = sess.run(fetches=[self.optimizer, self.loss, self.summary],
                                            feed_dict={self.img: img, self.label: label, self.prob: 0.75})

                total_loss = total_loss + loss
                if (i + 1) % TRAIN_OUT_PERIOD == 0:
                    writer.add_summary(summary, global_step=i)
                    self.saver.save(sess, './model/simple_arch', global_step=i)
                    print "epoch: {0} loss: {1}".format(i, total_loss / TRAIN_OUT_PERIOD)
                    total_loss = 0.0

    def test(self):
        test_data = self.data.test

        with tf.Session() as sess:
            self.__restore(sess)

            predict = tf.argmax(tf.nn.softmax(self.logits), 1)
            label = tf.argmax(self.label, 1)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, label), dtype=tf.float32))

            n_epochs = int(test_data.num_examples / BATCH_SIZE)
            total_acc = 0.0
            for i in range(0, n_epochs):
                img, label = test_data.next_batch(BATCH_SIZE)
                acc = sess.run(fetches=accuracy,
                               feed_dict={self.img: img, self.label: label, self.prob: 1.0})

                total_acc = total_acc + acc
                if (i + 1) % TEST_OUT_PERIOD == 0:
                    print "epoch: {0} accuracy: {1}".format(i, total_acc / TEST_OUT_PERIOD)
                    total_acc = 0.0

    def __restore(self, sess):
        ckpt = tf.train.get_checkpoint_state('./model')

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            return 0

        return sess.run(self.global_step)

    def __init_input(self):
        self.img = tf.placeholder(dtype=tf.float32, shape=[None, 28 * 28])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.prob = tf.placeholder(dtype=tf.float32)
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False)

    # conv, relu, max pool
    def __layer1(self):
        in_nodes = 1
        out_nodes = 32
        data = tf.reshape(self.img, shape=[-1, 28, 28, 1])

        with tf.name_scope("layer1"):
            weight = SimpleArch.get_weight([5, 5, in_nodes, out_nodes], 'weight1')
            bias = SimpleArch.get_bias([out_nodes], 'bias1')
            conv = SimpleArch.get_conv2d(data, weight)
            relu = tf.nn.relu(conv + bias)
            self.layer1_data = SimpleArch.get_pool2x2(relu)

    # conv, relu, max pool
    def __layer2(self):
        in_nodes = 32
        out_nodes = 64
        data = self.layer1_data


        with tf.name_scope("layer2"):
            weight = SimpleArch.get_weight([5, 5, in_nodes, out_nodes], 'weight2')
            bias = SimpleArch.get_bias([out_nodes], 'bias2')
            conv = SimpleArch.get_conv2d(data, weight)
            relu = tf.nn.relu(conv + bias)
            self.layer2_data = SimpleArch.get_pool2x2(relu)

    # fc, relu
    def __layer3(self):
        in_nodes = 7 * 7 * 64
        out_nodes = 1024
        data = tf.reshape(self.layer2_data, shape=[-1, in_nodes])

        with tf.name_scope("layer3"):
            weight = SimpleArch.get_weight([in_nodes, out_nodes], 'weight3')
            bias = SimpleArch.get_bias([out_nodes], 'bias3')
            fc = tf.matmul(data, weight)
            relu = tf.nn.relu(fc + bias)
            self.layer3_data = tf.nn.dropout(relu, self.prob)

    # fc, softmax, loss, optimizer
    def __layer4(self):
        in_nodes = 1024
        out_nodes = 10
        data = self.layer3_data

        with tf.name_scope("layer4"):
            weight = SimpleArch.get_weight([in_nodes, out_nodes], 'weight4')
            bias = SimpleArch.get_bias([out_nodes], 'bias4')
            fc = tf.matmul(data, weight)
            self.logits = fc + bias
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(loss)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss,
                                                                                  global_step=self.global_step)

    # summary
    def __summary(self):
        tf.summary.scalar("loss", self.loss)
        tf.summary.histogram("histogram loss", self.loss)
        self.summary = tf.summary.merge_all()

    # saver
    def __saver(self):
        self.saver = tf.train.Saver()

    @staticmethod
    def get_weight(shape, name):
        return tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer, name=name)

    @staticmethod
    def get_bias(shape, name):
        return tf.get_variable(shape=shape, initializer=tf.truncated_normal_initializer, name=name)

    @staticmethod
    def get_conv2d(data, kernel, x_step=1, y_step=1):
        return tf.nn.conv2d(input=data,
                            filter=kernel,
                            strides=[1, x_step, y_step, 1],
                            padding='SAME')

    @staticmethod
    def get_pool2x2(data):
        return tf.nn.max_pool(value=data,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')


def main():
    mnist_data = input_data.read_data_sets('./data/mnist', one_hot=True)
    simple_arch = SimpleArch(mnist_data)
    simple_arch.test()


if __name__ == '__main__':
    main()