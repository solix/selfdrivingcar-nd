# Solution is available in the other "solution.py" tab
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)
output = None
entropy_cross = -tf.reduce_sum(tf.mul(one_hot,tf.log(softmax)))
# Print cross entropy from session
with tf.Session() as sess:
    output = sess.run(entropy_cross,feed_dict={one_hot:one_hot_data,softmax:softmax_data })

print(output)