import tensorflow as tf
import numpy as np
#y_true should not be one hot encoded
def sparse_softmax_cross_entropy_with_logits(y_logits,y_true,class_num,loss_focus=None):
  with tf.name_scope('loss_sparse_softmax_cross_entropy_with_logits'):
    if loss_focus is not None:
      if loss_focus.shape[0]!=class_num:
        raise ValueError('loss focus weights are not compatible with class_num')
      loss_weights = loss_focus
    else:
      loss_weights = tf.constant(np.ones(class_num),dtype=tf.float32)
    
    if(loss_weights.shape[0]!=class_num):
      raise ValueError('Number of loss_weight for different class is not compatible')
    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    #labels=label_batch,logits=logits),name='loss')
    y_one_hot = tf.one_hot(y_true,depth=class_num,axis=-1,dtype=tf.float32)
    print('y_true shape:',y_true.shape.as_list())
    logsoftmax = -tf.log(tf.add(tf.nn.softmax(y_logits),1e-8))
    print('logsoftmax shape:',logsoftmax.shape.as_list())
    cross_entropy = tf.reduce_sum(tf.multiply(tf.multiply(y_one_hot,logsoftmax),loss_weights),axis=-1)
    print('cross_entropy shape:',cross_entropy.shape.as_list())
    loss =tf.reduce_mean(cross_entropy)
    return loss