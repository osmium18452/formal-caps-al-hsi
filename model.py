import tensorflow as tf
from tensorflow.contrib import slim
import capslayer as cl
import numpy as np


class model:
    def CapsNet(net, output):
        conv1 = tf.layers.conv2d(
            net,
            filters=64,
            kernel_size=3,
            strides=1,
            padding="same",
            activation=tf.nn.relu,
            name="convLayer"
        )

        convCaps, activation = cl.layers.primaryCaps(
            conv1,
            filters=64,
            kernel_size=3,
            strides=1,
            out_caps_dims=[8, 1],
            method="logistic"
        )

        n_input = np.prod(cl.shape(convCaps)[1:4])
        convCaps = tf.reshape(convCaps, shape=[-1, n_input, 8, 1])
        activation = tf.reshape(activation, shape=[-1, n_input])

        rt_poses, rt_probs = cl.layers.dense(
            convCaps,
            activation,
            num_outputs=output,
            out_caps_dims=[16, 1],
            routing_method="DynamicRouting"
        )
        return rt_probs

    def cnn(net, num_classes):
        print( num_classes)
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu):
            net = slim.conv2d(net, 300, 3, padding='VALID',
                              weights_initializer=tf.contrib.layers.xavier_initializer())
            net = slim.max_pool2d(net, 2, padding='SAME')
            net = slim.conv2d(net, 200, 3, padding='VALID',
                              weights_initializer=tf.contrib.layers.xavier_initializer())
            net = slim.max_pool2d(net, 2, padding='SAME')
            net = slim.flatten(net)
            net = slim.fully_connected(net, 200)
            net = slim.fully_connected(net, 100)
            logits = slim.fully_connected(net, num_classes, activation_fn=None)
        return logits
