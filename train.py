import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

import DataLoader
from maketable import *
from layer import *
from loadData import *

np.set_printoptions(threshold=0.1)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

SUBDIVSION = 3
NUM_CLASS = 10

class SpherePHD():

    def __init__(self, subdivision):
        self.conv_tables = []
        self.adj_tables = []
        for i in range(0, subdivision+1):
            self.conv_tables.append(make_conv_table(i))
            self.adj_tables.append(make_adjacency_table(i))
        self.pooling_tables = make_pooling_table(subdivision+1)


        self.num_epochs = 150
        self.num_steps = 1200
        self.batch_size = 50
        self.X = tf.placeholder(tf.float32, [None, 1, 20 * 4**subdivision, 1])
        self.Y = tf.placeholder(tf.float32, [None, NUM_CLASS])

        self.logits = conv_net(self.X, self.conv_tables, self.adj_tables, self.pooling_tables, subdivision)
        self.prediction = tf.nn.softmax(self.logits)

        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.Y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.00005)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # Evaluate model
        correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self.saver = tf.train.Saver()

        # Start training
        self.config = tf.ConfigProto()
        self.config.gpu_options.per_process_gpu_memory_fraction = 1
        self.config.gpu_options.allow_growth = True

    def train(self):

        # Training loop
        batch_size = self.batch_size
        save_file_path = './save/MNIST_Sphere_model_'
        sess = tf.Session(config=self.config)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        print("Training start")
        for epoch in range(self.num_epochs):
            print("epoch {}".format(epoch))
            for idx in range(1,2):
                train_images, train_labels = loadTrainData(idx)
                for step in range(1, self.num_steps + 1):
                    batch_x, batch_y = np.reshape(train_images[(step - 1) * batch_size:step * batch_size].astype(np.float32),
                                                  [batch_size, 1, 1280, 1]), train_labels[
                                                  (step - 1) * batch_size:step * batch_size].astype(np.float32)

                    sess.run(self.train_op, feed_dict={self.X: batch_x, self.Y: batch_y})

                    if step % 400 == 0:
                        loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={self.X: batch_x, self.Y: batch_y})
                        print("Step " + str((idx-1)*self.num_steps+step) + 
                              ", Minibatch Loss= " +"{:.4f}".format(loss) + 
                              ", Training Accuracy= " + "{:.3f}".format(acc)+'\r', end='')

            print()
            save_path = self.saver.save(sess, save_file_path+str(epoch)+'.ckpt')
        print("Model saved in path: %s" % save_path)
        print("Training finish")

    def test(self):

        batch_size = self.batch_size
        sess = tf.Session(config=self.config)
        sess = tf.Session()
        init = tf.global_variables_initializer()
        sess.run(init)
        self.saver.restore(sess, "./save/MNIST_Sphere_model_149.ckpt")
        print("Testing start")
        total_accuracy = 0
        for idx in range(1,2):
            test_images, test_labels = loadTestData(idx)
            for step in range(1, 201):
                batch_x, batch_y = np.reshape(test_images[(step - 1) * batch_size:step * batch_size].astype(np.float32),
                                              [batch_size, 1, 1280, 1]), test_labels[
                                              (step - 1) * batch_size:step * batch_size].astype(np.float32)
                acc = sess.run([self.accuracy], feed_dict={self.X: batch_x, self.Y: batch_y})
                total_accuracy = total_accuracy + acc[0]
        total_accuracy *= batch_size
        print("Total accuracy is {:.2f}%".format(total_accuracy/100))
        sess.close()

def main():
    MNIST = SpherePHD(SUBDIVSION)
    MNIST.train()
    MNIST.test()

if __name__ == '__main__':
    main()