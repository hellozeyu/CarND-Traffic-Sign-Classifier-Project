import tensorflow as tf
from tensorflow.contrib.layers import flatten

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def weight_variable(shape, name):
  return tf.Variable(tf.truncated_normal(shape, mean = 0, stddev=0.1), name=name)

def bias_variable(shape, name):
  return tf.Variable(tf.zeros(shape=shape), name)

def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

class LeNet(object):

    """
        For the SAME padding, the output height and width are computed as:

        out_height = ceil(float(in_height) / float(strides1))

        out_width = ceil(float(in_width) / float(strides[2]))

        For the VALID padding, the output height and width are computed as:

        out_height = ceil(float(in_height - filter_height + 1) / float(strides1))

        out_width = ceil(float(in_width - filter_width + 1) / float(strides[2]))
    """

    def __init__(self, num_classes):
        self.features = tf.placeholder(tf.float32, (None, 32, 32, 1), name='features')
        self.labels = tf.placeholder(tf.int32, (None, num_classes), name='labels')
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

        with tf.variable_scope('conv1'):
            # Convolutional. Input = 32x32x3. Output = 16x16x32.
            filter1 = [5, 5, 1, 32]
            W1 = weight_variable(filter1, "W1")
            b1 = bias_variable([filter1[-1]], name="b1")
            conv1 = conv2d(self.features, W1) + b1
            conv1 = tf.nn.relu(conv1)
            conv1 = maxpool2d(conv1)
        self.h_pool1 = conv1

        with tf.variable_scope('conv2'):
            # Convolutional. Input = 16x16x32. Output = 8x8x64.
            filter2 = [5, 5, 32, 64]
            W2 = weight_variable(filter2, "W2")
            b2 = bias_variable([filter2[-1]], name="b2")
            conv2 = conv2d(self.h_pool1, W2) + b2
            conv2 = tf.nn.relu(conv2)
            conv2 = maxpool2d(conv2)
        self.h_pool2 = conv2

        with tf.variable_scope('fc1'):
            # Fully Connected Input = 8x8x64. Output = 1024.
            W_fc1 = weight_variable([8 * 8 * 64, 1024], "W_fc1")
            b_fc1 = bias_variable([1024], 'b_fc1')
            h_pool2_flat = flatten(self.h_pool2)
            h_fc1 = tf.nn.xw_plus_b(h_pool2_flat, W_fc1, b_fc1)
        self.h_fc1 = h_fc1

        with tf.variable_scope('output'):
            # Fully Connected Input = 1024. Output = 43.
            h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
            W_fc2 = weight_variable([1024, 43], "W_fc2")
            b_fc2 = bias_variable([43], "b_fc2")
            self.scores = tf.nn.xw_plus_b(h_fc1_drop, W_fc2, b_fc2, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.labels)
            self.loss = tf.reduce_mean(losses)

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
