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
def _deconv_variable( weight_shape,name="conv"):
    with tf.variable_scope(name):
        # check weight_shape
        w = int(weight_shape[0])
        h = int(weight_shape[1])
        output_channels = int(weight_shape[2])
        input_channels  = int(weight_shape[3])
        weight_shape = (w,h,input_channels, output_channels)
        regularizer = tf.contrib.layers.l2_regularizer(scale=REGULARIZER_COF)
        # define variables
        weight = tf.get_variable("w", weight_shape    ,
                                initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                regularizer=regularizer)
        bias   = tf.get_variable("b", [input_channels], initializer=tf.constant_initializer(0.0))
    return weight, bias

def _conv2d( x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def _deconv2d( x, W, output_shape, stride=1):
    # x           : [nBatch, height, width, in_channels]
    # output_shape: [nBatch, height, width, out_channels]
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1,stride,stride,1], padding = "SAME",data_format="NHWC")

def _conv_layer(x, input_layer, output_layer, stride, filter_size=3, name="conv", isTraining=True):
    conv_w, conv_b = _conv_variable([filter_size,filter_size,input_layer,output_layer],name="conv"+name)
    h = _conv2d(x,conv_w,stride=stride) + conv_b
    h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNormc"+name)
    h = tf.nn.leaky_relu(h)
    return h

def _deconv_layer(x,input_layer, output_layer, stride=2, filter_size=3, name="deconv", isTraining=True):
    bs, h, w, c = x.get_shape().as_list()
    deconv_w, deconv_b = _deconv_variable([filter_size,filter_size,input_layer,output_layer],name="deconv"+name )
    h = _deconv2d(x,deconv_w, output_shape=[bs,h*2,w*2,output_layer], stride=stride) + deconv_b
    h = tf.contrib.layers.batch_norm(h, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, is_training=isTraining, scope="gNormd"+name)
    h = tf.nn.leaky_relu(h)
    return h

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

        h = _conv_layer(h,64,64,1,3,name="3_g", isTraining=isTraining)
        h = tf.math.add(tmp,h, name="add")

        h = _conv_layer(h, 64, 256, 1, 3, "2_g", isTraining=isTraining)

        h = _deconv_layer(h, 256, 128, 2, 3, "2_g", isTraining=isTraining)

        h = _conv_layer(h, 128, 128, 1, 3, "1_g", isTraining=isTraining)

        h = _deconv_layer(h, 128, 64, 2, 3, "1_g", isTraining=isTraining)

        h = _conv_layer(h, 64, 64, 1, 3, "0_g", isTraining=isTraining)

        conv_w, conv_b = _conv_variable([7,7,64,3],name="convo_g" )
        h = _conv2d(h,conv_w,stride=1) + conv_b
        y = tf.nn.tanh(h)

    return y
