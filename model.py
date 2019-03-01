import tensorflow as tf
import numpy as np

REGULARIZER_COF = 0.4

def _fc_variable( weight_shape,name="fc"):
    with tf.variable_scope(name):
        # check weight_shape
        input_channels  = int(weight_shape[0])
        output_channels = int(weight_shape[1])
        weight_shape    = (input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.contrib.layers.xavier_initializer(),
                                regularizer =regularizer)
        bias   = tf.get_variable("b", [weight_shape[1]],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv_variable( weight_shape,name="conv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        input_channels  = int(weight_shape[2])
        output_channels = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape     ,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                regularizer=regularizer)
        bias   = tf.get_variable("b", [output_channels],
                                initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv2d( x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def pixel_shuffle_layer(x, r, n_split, name="PS"):
    with tf.variable_scope(name):
        def PS(x, r):
            bs, a, b, c = x.get_shape().as_list()

            bs = tf.shape(x)[0]

            x = tf.reshape(x, (bs, a, b, r, r))
            x = tf.transpose(x, (0, 1, 2, 4, 3))
            x = tf.split(x, a, 1)
            x = tf.concat([tf.squeeze(x_,axis=1) for x_ in x], 2)
            x = tf.split(x, b, 1)
            x = tf.concat([tf.squeeze(x_,axis=1) for x_ in x], 2)
            return tf.reshape(x, (bs, a*r, b*r, 1))

        xc = tf.split(x, n_split, 3)
    return tf.concat([PS(x_, r) for x_ in xc], 3)

def buildSRGAN_g(x,reuse=False,isTraining=True,nBatch=64):

    with tf.variable_scope("SRGAN_g", reuse=reuse) as scope:
        if reuse: scope.reuse_variables()

        conv_w, conv_b = _conv_variable([3,3,3,64],name="conv4_g")
        h = _conv2d(x,conv_w,stride=1) + conv_b
        h = tf.nn.leaky_relu(h)

        tmp = h

        for i in range(16):
            conv_w, conv_b = _conv_variable([3,3,64,64],name="res%s-1" % i)
            nn = _conv2d(h,conv_w,stride=1) + conv_b
            nn = tf.contrib.layers.batch_norm(nn, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="Norm%s-1_g" %i)
            nn = tf.nn.leaky_relu(nn)
            conv_w, conv_b = _conv_variable([3,3,64,64],name="res%s-2" % i)
            nn = _conv2d(nn,conv_w,stride=1) + conv_b
            nn = tf.contrib.layers.batch_norm(nn, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="Norm%s-2_g" %i)

            nn = tf.math.add(h,nn, name="resadd%s" % i)
            h = nn

        conv_w, conv_b = _conv_variable([3,3,64,64],name="conv3_g" )
        h = _conv2d(h,conv_w,stride=1) + conv_b
        h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="Norm3_g")
        h = tf.nn.leaky_relu(h)
        h = tf.math.add(tmp,h, name="add")

        conv_w, conv_b = _conv_variable([3,3,64,256],name="conv2_g" )
        h = _conv2d(h,conv_w,stride=1) + conv_b
        h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="Norm2_g")
        h = tf.nn.leaky_relu(h)

        h = pixel_shuffle_layer(h, 2, 64)

        conv_w, conv_b = _conv_variable([3,3,64,256],name="conv1_g" )
        h = _conv2d(h,conv_w,stride=1) + conv_b
        h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="Norm1_g")
        h = tf.nn.leaky_relu(h)

        h = pixel_shuffle_layer(h, 2, 64)

        conv_w, conv_b = _conv_variable([3,3,64,64],name="conv0_g" )
        h = _conv2d(h,conv_w,stride=1) + conv_b
        h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="Norm0_g")
        h = tf.nn.leaky_relu(h)

        conv_w, conv_b = _conv_variable([3,3,64,3],name="convo_g" )
        h = _conv2d(h,conv_w,stride=1) + conv_b
        y = tf.nn.tanh(h)

    return y
