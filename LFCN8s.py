import tensorflow as tf
import numpy as np
from LLayer import *
from LLoss import *
from LMetric import *
from LNet import *
import LDataInput
import sys


if __name__=='__main__':
  try:
    H = 1080
    W = 2048
    ClassNum = 8#final output feature channels
    solver_opt = solverOptions()
    
    net = Net('FCN8s')
    with net.Graph.as_default():
      #begin
      Input_X = net.StartNetDef([None,H,W,3])
      #1
      X = Conv2D(Input_X,ksize=3,stride=1,o_c=64,use_relu=True,name='Conv1_1')
      X = Conv2D(X,ksize=3,stride=1,o_c=64,use_relu=True,name='Conv1_2')
      Pool1 = MaxPooling2D(X,ksize=2,stride=2,name='Pool1')
      #2
      X = Conv2D(Pool1,ksize=3,stride=1,o_c=128,use_relu=True,name='Conv2_1')
      X = Conv2D(X,ksize=3,stride=1,o_c=128,use_relu=True,name='Conv2_2')
      Pool2 = MaxPooling2D(X,ksize=2,stride=2,name='Pool2')
      #3
      X = Conv2D(Pool2,ksize=3,stride=1,o_c=256,use_relu=True,name='Conv3_1')
      X = Conv2D(X,ksize=3,stride=1,o_c=256,use_relu=True,name='Conv3_2')
      X = Conv2D(X,ksize=3,stride=1,o_c=256,use_relu=True,name='Conv3_3')
      Pool3 = MaxPooling2D(X,ksize=2,stride=2,name='Pool3')
      #4
      X = Conv2D(Pool3,ksize=3,stride=1,o_c=512,use_relu=True,name='Conv4_1')
      X = Conv2D(X,ksize=3,stride=1,o_c=512,use_relu=True,name='Conv4_2')
      X = Conv2D(X,ksize=3,stride=1,o_c=512,use_relu=True,name='Conv4_3')
      Pool4 = MaxPooling2D(X,ksize=2,stride=2,name='Pool4')
      #5
      X = Conv2D(Pool4,ksize=3,stride=1,o_c=512,use_relu=True,name='Conv5_1')
      X = Conv2D(X,ksize=3,stride=1,o_c=512,use_relu=True,name='Conv5_2')
      X = Conv2D(X,ksize=3,stride=1,o_c=512,use_relu=True,name='Conv5_3')
      Pool5 = MaxPooling2D(X,ksize=2,stride=2,name='Pool5')
      #6
      X = Conv2D(Pool5,ksize=7,stride=1,o_c=4096,use_relu=True,name='FC6')
      X = Dropout(X,keep_prob=0.5,name='Drop6')
      X = Conv2D(X,ksize=1,stride=1,o_c=4096,use_relu=True,name='FC7')
      X = Dropout(X,keep_prob=0.5,name='Drop7')
      score_fr = Conv2D(X,ksize=1,stride=1,o_c=ClassNum,use_relu=True,name='score_fr')
      #7
      X = DeConv2D(score_fr,ksize=4,stride=2,o_c=ClassNum,
        top_sp=tf.concat([tf.shape(Pool4)[:-1],[ClassNum]],axis=0),
        usebias=False,name='upscore2')
      score_pool4 = Conv2D(Pool4,ksize=1,stride=1,o_c=ClassNum,use_relu=False,name='score_pool4')
      fuse_pool4 = tf.add(X,score_pool4,name='fuse_pool4')
      #8
      X = DeConv2D(fuse_pool4,ksize=4,stride=2,o_c=ClassNum,
        top_sp=tf.concat([tf.shape(Pool3)[:-1],[ClassNum]],axis=0),
        usebias=False,name='upscore_pool4')
      score_pool3 = Conv2D(Pool3,ksize=1,stride=1,o_c=ClassNum,use_relu=False,name='score_pool3')
      fuse_pool3 = tf.add(X,score_pool3,name='fuse_pool3')
      #9
      X = DeConv2D(fuse_pool3,ksize=16,stride=8,o_c=ClassNum,
        top_sp=tf.concat([tf.shape(Input_X)[:-1],[ClassNum]],axis=0),
        usebias=False,name='upscore_32')

      logits = net.EndNetDef(logits=X)
      #end

      #define loss and metric
      loss = sparse_softmax_cross_entropy_with_logits(
        y_logits=logits,y_true=net.Input_Y,class_num=ClassNum,loss_focus=None)
      metric = accuracy(y_pred = net.Prediction(),y_true = net.Input_Y)

      net.Compile(loss,metric,solver_opt)

      filepath = '/home/yelyu/Work/MyDLSolutions/UAV/vid1_all_type/vid1_train_4d_150sp/ListFile.txt'
      fileList = LDataInput.getFileLists(filepath)
      gen = LDataInput.DataGenerator(fileList=fileList,output_channels=8,batch_size=1,epoch=1,shuffle = False)
      gen.set_label_encoder(LDataInput.UAVImageColorEncoder())
      print('begin gen')
      net.fit_generator(gen)
  except Exception as e:
    print('ExceptHook, terminate all child processes!')
    #for p in multiprocessing.active_children():
    #   p.terminate()