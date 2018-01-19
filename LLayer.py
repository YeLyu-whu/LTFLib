import tensorflow as tf
import numpy as np

L_weight_collection = 'L_weight_collection'

def InitializerConstantType(val=0,dtype=tf.float32):
    return tf.constant_initializer(value=val,dtype=dtype)
def InitializerXavierType():
    return tf.contrib.layers.xavier_initializer(dtype=tf.float32,uniform=True)
#i_c :input channel number
#o_c:output channel number
def InitializerDeconvType(ksize,i_c,o_c):
    if o_c<i_c:
        raise ValueError('deconv filter weight error:outfilterNum is smaller than in channelNum')
    f = np.ceil(ksize/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([ksize, ksize])
    for x in range(ksize):
        for y in range(ksize):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    vals = np.zeros([ksize,ksize,o_c,i_c])
    for i in range(i_c):
        vals[:, :, i, i] = bilinear
    return vals

#o_c output channel number
def Conv2D(bottom,ksize,stride,o_c,use_relu=True,name=''):
  with tf.variable_scope(name):
    bshape = bottom.shape.as_list()
    kshape = [ksize,ksize]+[bshape[3]]+[o_c]
    #bshape = tf.shape(bottom)
    #kshape = tf.concat([[ksize,ksize],[bshape[3]],[o_c]],axis=0)
    w = TensorWeights(shape = kshape,initializer=InitializerXavierType())
    b = TensorBias(shape = [o_c],initializer=InitializerConstantType(0))
    conv = tf.nn.conv2d(bottom, w, strides =[1,stride,stride,1], padding='SAME',name='conv2d')
    if use_relu:
      top = Relu(tf.nn.bias_add(conv,b,name='bias_add'))
    else:
      top = tf.nn.bias_add(conv,b,name='bias_add')
    print(bottom.name,'->',top.name)
    return top

def MaxPooling2D(bottom,ksize,stride,name=''):
  with tf.variable_scope(name):
    top = tf.nn.max_pool(bottom,ksize=[1, stride, stride, 1],strides=[1, stride, stride, 1],padding='SAME', name=name)
    print(bottom.name,'->',top.name)
    return top

#top_sp: top tensor shape
def DeConv2D(bottom,ksize,stride,o_c,top_sp,usebias=False,name=''):
  with tf.variable_scope(name):
    f = np.ceil(ksize/2.0)
    if f!=stride:
      raise ValueError('ksize and stride are not compatible')
    bshape = bottom.shape.as_list()
    kshape = [ksize,ksize]+[o_c]+[bshape[3]]
    #bshape = tf.shape(bottom)
    #kshape = tf.concat([[ksize,ksize],[o_c],[bshape[3]]],axis=0)
    vals = InitializerDeconvType(ksize,bshape[3],o_c)
    w = TensorWeights(shape=kshape,initializer=InitializerConstantType(vals))
    strides = [1,stride,stride,1]
    deconv = tf.nn.conv2d_transpose(bottom, w, top_sp,strides=strides, padding='SAME',name='deconv')
    if usebias:
      b = TensorBias(shape=[o_c],initializer=InitializerConstantType(0))
      top = tf.nn.bias_add(deconv,b)
    else:
      top=deconv
    print(bottom.name,'->',top.name)
    return top

def Dropout(bottom,keep_prob=1.0,name=''):
  with tf.variable_scope(name):
    top = tf.nn.dropout(bottom,keep_prob=keep_prob,name=name)
    print(bottom.name,'->',top.name)
    return top

def Relu(bottom,name='relu'):
  with tf.variable_scope(name):
    top = tf.nn.relu(bottom,name=name)
    #print(bottom.name,'->',top.name)
    return top

def TensorWeights(shape,initializer,wd=1e-5,dtype = tf.float32):
  weight = tf.get_variable("weights",shape=shape,initializer=initializer,dtype=dtype)
  tf.add_to_collection('weight_collection', weight)
  return weight

def TensorBias(shape,initializer,dtype = tf.float32):
  bia = tf.get_variable("bias",shape=shape,initializer=initializer,dtype=dtype)
  return bia

def AddWeightDecay(var,wd=1e-5):
  with tf.variable_scope('weight_decay'):
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_decay_loss')
    tf.add_to_collection('weight_decay_collection', weight_decay)
    return weight_decay

def SummaryVar(var):
  if not tf.get_variable_scope().reuse:
    name = var.op.name
    logging.info("Creating Summary for: %s" % name)
    with tf.name_scope('summaries'):
      mean = tf.reduce_mean(var)
      tf.summary.scalar(name + '/mean', mean)
      with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
      tf.summary.scalar(name + '/sttdev', stddev)
      tf.summary.scalar(name + '/max', tf.reduce_max(var))
      tf.summary.scalar(name + '/min', tf.reduce_min(var))
      tf.summary.histogram(name, var)
  print(var.name)
  print(var.shape.as_list())
  return None