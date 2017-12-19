import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt


class AuxTool(object):
    @staticmethod
    def create_summary(loss):
        tf.summary.scalar('loss', loss)
        tf.summary.histogram('histogram loss', loss)
        return tf.summary.merge_all()

    @staticmethod
    def create_saver():
        return tf.train.Saver()

    @staticmethod
    def restore_sess(saver, sess, global_step, path):
        ckpt = tf.train.get_checkpoint_state(path)

        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
            return 0

        if global_step is None:
            return 0

        return sess.run(global_step)

    @staticmethod
    def read_img(path):
        return cv2.imread(path)

    @staticmethod
    def show_img(img):
        plt.imshow(img)
        plt.show()

    @staticmethod
    def rgb2bgr(img):
        r, g, b = np.split(img, 3, axis=-1)
        return np.concatenate([b, g, r], axis=-1)
