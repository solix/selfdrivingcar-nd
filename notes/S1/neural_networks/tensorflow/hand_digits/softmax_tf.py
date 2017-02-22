# Solution is available in the other "solution.py" tab
import tensorflow as tf


def run():
    output = None
    logit_data = [2.0, 1.0, 0.1]
    logits = tf.placeholder(tf.float32)
    mul_logit = [i * 5 for i in logits]


    # Calculate the softmax of the logits
    softmax = tf.nn.softmax(logits)
    with tf.Session() as sess:
    # Feed in the logit data
     output = sess.run(softmax,feed_dict={logits:mul_logit})
     print(output)
    return output
