import tensorflow as tf

input = tf.placeholder(tf.float,(None,32,32,3))

filter_weights = tf.Variable(tf.truncated_normal((8,8,3,20)))
filter_bias = tf.Variable(tf.zeros(20))
strides = [1, 2, 2, 1]
padding = 'VALID'
conv = tf.nn.conv2d(input,filter_weights,strides,padding) + filter_bias