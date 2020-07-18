import tensorflow as tf
import numpy as np


def conv2d(inputs, filters, kernel_size, strides, name, data_format,
         padding='SAME', groups=1):
    """2-D convolution from Caffe to TensorFlow."""
    # It includes the ops of conv and relu.
    if data_format == 'channels_first':  # NCHW
        # Get number of input channels
        input_channels = int(inputs.get_shape()[1])
        # Set params of tf.nn.conv2d
        strides1 = [1, 1, strides, strides]
        data_format1 = 'NCHW'
        # Set params of tf.split and tf.concat
        axis1 = 1

    elif data_format == 'channels_last':  # NHWC
        input_channels = int(inputs.get_shape()[-1])
        strides1 = [1, strides, strides, 1]
        data_format1 = 'NHWC'
        axis1 = 3

    # Create lambda function for the convolution
    convolve = lambda i, k: tf.nn.conv2d(i, k,
                                         strides=strides1,
                                         padding=padding,
                                         data_format=data_format1)
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights', shape=[kernel_size,
                                                    kernel_size,
                                                    input_channels / groups,
                                                    filters])
        biases = tf.get_variable('biases', shape=[filters])

    if groups == 1:
        conv = convolve(inputs, weights)
    # In the cases of multiple groups, split inputs & weights
    else:
        # Split input and weights and convolve them separately
        # Axis should refer to the axis of channels
        input_groups = tf.split(inputs, num_or_size_splits=groups, axis=axis1)
        weights_groups = tf.split(weights, num_or_size_splits=groups, axis=3)
        output_groups = [convolve(i, k) for i, k in zip(input_groups,
                                                        weights_groups)]

        # Concat the convolved output together again
        conv = tf.concat(output_groups, axis=axis1)

    # Add biases
    bias = tf.reshape(tf.nn.bias_add(conv, biases, data_format=data_format1), tf.shape(conv))

    # Apply relu function
    relu = tf.nn.relu(bias, name=scope.name)
    return relu


def max_pool(inputs, kernel_size, strides, name, data_format,
             padding='SAME'):
    """Create a max pooling layer."""
    if data_format == 'channels_first':
        data_format1 = 'NCHW'
        ksize = [1, 1, kernel_size, kernel_size]
        stride = [1, 1, strides, strides]
    elif data_format == 'channels_last':
        data_format1 = 'NHWC'
        ksize = [1, kernel_size, kernel_size, 1]
        stride = [1, strides, strides, 1]
    return tf.nn.max_pool(inputs, ksize=ksize, strides=stride,
                          padding=padding, data_format=data_format1,
                          name=name)


def lrn(inputs, radius, alpha, beta, name, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(inputs, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


def fc(inputs, num_in, num_out, name, relu=True):
    """Create a fully connected layer."""
    with tf.variable_scope(name) as scope:
        # Create tf variables for the weights and biases
        weights = tf.get_variable('weights', shape=[num_in, num_out],
                                  trainable=True)
        biases = tf.get_variable('biases', [num_out], trainable=True)

        # Matrix multiply weights and inputs and add bias
        act = tf.nn.xw_plus_b(inputs, weights, biases, name=scope.name)

    if relu:
        # Apply ReLu non linearity
        relu = tf.nn.relu(act)
        return relu
    else:
        return act


def dropout(inputs, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(inputs, keep_prob)


def alexnet_memory_generator(keep_prob, data_format=None):
    if data_format is None:
        data_format = (
            'channels_first' if tf.test.is_built_with_cuda() else 'channels_last')

    def model(inputs, is_training):
        """Constructs the AlexNet model given the inputs."""
        if data_format == 'channels_first':
            inputs = tf.transpose(inputs, [0, 3, 1, 2])

        # 1st Layer: Conv -> relu -> pool -> norm
        conv1 = conv2d(inputs=inputs, filters=96, kernel_size=11, strides=4,
                     padding='VALID', name='conv1', data_format=data_format)
        pool1 = max_pool(inputs=conv1, kernel_size=3, strides=2, name='pool1',
                         data_format=data_format, padding='VALID')
        norm1 = lrn(inputs=pool1, radius=2, alpha=2e-05, beta=0.75, name='norm1')

        # 2st Layer: Conv -> relu -> pool -> norm with 2 groups
        conv2 = conv2d(inputs=norm1, filters=256, kernel_size=5, strides=1,
                     name='conv2', groups=2, data_format=data_format)
        pool2 = max_pool(inputs=conv2, kernel_size=3, strides=2, name='pool2',
                         data_format=data_format, padding='VALID')
        norm2 = lrn(inputs=pool2, radius=2, alpha=2e-05, beta=0.75, name='norm2')

        # 3rd Layer: Conv -> relu
        conv3 = conv2d(inputs=norm2, filters=384, kernel_size=3, strides=1,
                     name='conv3', data_format=data_format)

        # 4th Layer: Conv -> relu with 2 groups
        conv4 = conv2d(inputs=conv3, filters=384, kernel_size=3, strides=1,
                     name='conv4', groups=2, data_format=data_format)

        # 5th Layer: Conv -> relu -> pool -> norm with 2 groups
        conv5 = conv2d(inputs=conv4, filters=256, kernel_size=3, strides=1,
                     name='conv5', groups=2, data_format=data_format)
        pool5 = max_pool(inputs=conv5, kernel_size=3, strides=2, name='pool5',
                         data_format=data_format, padding='VALID')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, 6 * 6 * 256, 4096, name='fc6')
        # dropout6 = dropout(fc6, keep_prob)
        if is_training:
            dropout6 = dropout(fc6, keep_prob)
        else:
            dropout6 = fc6

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, 4096, 4096, name='fc7')
        # dropout7 = dropout(fc7, keep_prob)
        # if is_training:
        #     dropout7 = dropout(fc7, keep_prob)
        # else:
        #     dropout7 = fc7

        # 8th Layer: FC and return unscaled activations
        # fc8 = fc(dropout7, 4096, 1, relu=False, name='fc8-euclidean')
        return conv5, fc7
    return model


def load_initial_weights(weights_path, skip_layer, session):
    """Load weights from file into network."""
    # Load the weights into memory
    weights_dict = np.load(weights_path, encoding='bytes').item()

    # Loop over all layer names stored in the weights dict
    for op_name in weights_dict:

        # Check if layer should be trained from scratch
        if op_name not in skip_layer:

            with tf.variable_scope(op_name, reuse=True):

                # Assign weights/biases to their corresponding tf variable
                for data in weights_dict[op_name]:

                    # Biases
                    if len(data.shape) == 1:
                        var = tf.get_variable('biases', trainable=False)
                        session.run(var.assign(data))

                    # Weights
                    else:
                        var = tf.get_variable('weights', trainable=False)
                        session.run(var.assign(data))
