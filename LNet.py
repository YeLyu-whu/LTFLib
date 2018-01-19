import tensorflow as tf
import copy
import os
from LLayer import AddWeightDecay
from LDataInput import LOutOfRangeError

class solverOptions:
  def __init__(self):
    #phase: 'Train' or 'Test'
    self.phase='Train'
    self.test_interval=100000
    self.base_lr=0.0001
    #lr_policy: 'step' or 'fixed'
    self.lr_policy='step'
    #start_step: first input index
    self.start_step=0
    #stepsize: number of iterations
    self.stepsize=10
    self.gamma=0.96
    self.momentum=0.9
    self.weight_decay=0.005
    #interval of snapshot,batches per snapshot
    self.snapshot=20
    #path of params save
    self.snapshow_prefix='./snapshot'
    self.solver_type='Adam' 



class Net():
  def __init__(self,name):
    self.Graph = tf.Graph()
    self.Activate()

  def Activate(self):
    self.Graph.as_default()
    self.sess = tf.Session(graph = self.Graph)

  def Input(self):
    return self.Input

  def Input_Y(self):
    return self.Input_Y

  def Output(self):
    return self.Output

  def Model(self):
    return self.Model

  def Logits(self):
    return self.Logits

  def Prediction(self):
    return self.prediction

  def StartNetDef(self,shape_feature):
    print('Start Net Definition...')
    #shape_feature: [H,W,C]
    input_X = tf.placeholder(tf.float32,shape = shape_feature)
    input_Y = tf.placeholder(tf.int32,shape=shape_feature[:-1])
    self.Input = input_X
    self.Input_Y = input_Y
    return self.Input

  def EndNetDef(self,logits):
    self.Logits = logits
    self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    print('End Net Definition.')
    self.__get_Predict()
    return self.Logits

  def __parse_solver_opts(self,solverOptions):
    self.solverOptions = copy.copy(solverOptions)
    self.global_step = self.__get_global_step(self.solverOptions.start_step)
    self.lr = self.__get_learning_rate(self.solverOptions)
    self.saverFilePath = self.__get_saver_file_path(self.solverOptions)
    self.__get_solver(self.solverOptions,self.lr)
    self.__add_weight_decay(self.solverOptions.weight_decay)

  def print_solver(self):
    if self.solver is None:
      print('solver is not defined')
      self.solverOptions = None
      return False
    else:
      opts = self.solverOptions
      #TODO
      print('')
    return True

  def __get_saver_file_path(self,solverOptions):
    saverFilePath = os.path.join(solverOptions.snapshow_prefix,"model.ckpt")
    return saverFilePath

  def __get_solver(self,solverOptions,learning_rate):
    if self.solverOptions.solver_type=='Adam':
      self.optimizer =  tf.train.AdamOptimizer(learning_rate)
    else:
      print('Cannot find solver type '+self.solverOptions.solver_type)
      raise ValueError('solver type error')
    print('Use '+self.solverOptions.solver_type+' as solver')
    return self.optimizer

  def __get_global_step(self,start_step=0):
    global_step = tf.get_variable("global_step",initializer=tf.constant(start_step),dtype=tf.int32,trainable=False)
    #self.sess.run(global_step.initializer)
    return global_step

  def __get_learning_rate(self,solverOptions):
    starter_learning_rate = solverOptions.base_lr
    if solverOptions.lr_policy=='step':
      print('learning rate policy: step')
      learning_rate = tf.train.exponential_decay(
      starter_learning_rate, self.global_step,
      solverOptions.stepsize, solverOptions.gamma, 
      staircase=True)
      # Passing global_step to minimize() will increment it at each step.
    elif solverOptions.lr_policy=='fixed':
      learning_rate=starter_learning_rate
    else:
      raise ValueError('lr_policy is not correctly defined')
    return learning_rate

  def __add_weight_decay(self,wd):
    if wd==0:
      return
    else:
      weight_collection = tf.get_collection('weight_collection')
      for w in weight_collection:
        AddWeightDecay(w,wd)

  def Compile(self,loss,metrics,solverOptions,restore_params = False,restorePath=''):
    print('Compile Network...')
    self.__parse_solver_opts(solverOptions)
    if restore_params:
      if restorePath=='':
        saver.restore(self.sess, self.saverFilePath)
      else:
        saver.restore(self.sess, restorePath)
      print("Model restored...")
    else:
      init_global = tf.global_variables_initializer()
      self.sess.run(init_global)
      print('Model Initialized...')
    self.loss = loss
    self.metrics= metrics
    assert(self.optimizer is not None)
    assert(self.loss is not None)
    assert(self.metrics is not None)

    print('Initialize params in optimizer if there is any...')
    tempVariables = set(tf.global_variables())
    self.train_op = self.optimizer.minimize(self.loss,self.global_step)
    self.sess.run(tf.variables_initializer(set(tf.global_variables()) - tempVariables))
    print('Initialize params in optimizer done.')

  def fit(self,features,labels):
    init_local = tf.local_variable_initializer()
    self.sess.run(init_local)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=self.sess,coord = coord)
    try:
      while not coord.should_stop():
        batchNum = self.sess.run(self.global_step)
        rtvs = self.sess.run([self.loss,self.metrics,self.train_op],feed_dict={self.Input:features,self.Input_Y:labels})
        print("batch:%d, loss:%f,accuracy:%f"%(batchNum,rtvs[0],rtvs[1]))
        self.saver.save(self.sess,self.saverFilePath,global_step=batchNum)
    except tf.errors.OutOfRangeError:
      print('Done training -- batch limit reached')
    finally:
      coord.request_stop()
      coord.join(threads)
      self.saver.save(self.sess,self.saverFilePath,global_step=batchNum)

  def fit_generator_parallel_input(self,generator):
    pass

  def fit_generator(self,generator):
    while(True):
      try:
        for X,Y in generator.generate():
          print(X.shape)
          print(Y.shape)          
          batchNum = self.sess.run(self.global_step)
          rtvs = self.sess.run([self.loss,self.metrics,self.train_op],feed_dict={self.Input:X,self.Input_Y:Y})
          print("batch:%d, loss:%f,accuracy:%f"%(batchNum,rtvs[0],rtvs[1]))
          self.saver.save(self.sess,self.saverFilePath,global_step=batchNum)
      except LOutOfRangeError:
        print('Done training -- batch limit reached')
      except Exception as e:
        print("Graph Running Error!")
        print(e)
        raise
      finally:
        self.saver.save(self.sess,self.saverFilePath,global_step=batchNum)
        break


  def __get_Predict(self):
    if self.Logits is None:
      raise ValueError('Logits is not defined yet.')
    else:
      self.prediction = tf.argmax(self.Logits, axis=3,output_type=tf.int32, name="Prediction")