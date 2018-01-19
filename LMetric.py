import tensorflow as tf

#y_true should not be one hot encoded
def accuracy(y_pred,y_true):
  with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred,y_true) ,tf.float32))
    return accuracy