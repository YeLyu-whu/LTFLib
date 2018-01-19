import sys
import os 
import multiprocessing
import PIL
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as mp
import threading as thrd
from queue import Queue
#import keras
import copy
sys.path.append('/home/yelyu/Work/MyDLSolutions/UAV/util/')
from colorEncoder import UAVImageColorEncoder

class LOutOfRangeError(Exception):
  pass


class DataGenerator(object):
  def __init__(self,fileList,output_channels,batch_size,epoch,shuffle=True,threadBuf=2):
    print("Build Data Generator..")
    #self.dim_h = dim[0]
    #self.dim_w = dim[1]
    #self.dim_c = dim[2]
    self.fileList = fileList
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.temp_fileList = copy.copy(fileList)
    self.output_channels = output_channels
    self.label_encoder = None
    assert(epoch>0)
    self.epoch = epoch
    self.data_queue = Queue(threadBuf)
    self.FinishGen = False
    self.t = None

  def enQueueThread(self):
    count=0
    while count<self.epoch:
      count+=1
      fileList = self.fileList
      if self.shuffle:
        self.temp_fileList = copy.copy(fileList)
        np.random.shuffle(self.temp_fileList)

      num = len( self.temp_fileList)
      imax = int(num//self.batch_size)
      for i in range(imax):        
        batch_list = [it for it in  self.temp_fileList[i*self.batch_size:(i+1)*self.batch_size]] 
        X,y = self.__data_generation(batch_list)
        self.data_queue.put((X,y))
    self.FinishGen = True

  def generate(self):
    if self.t is None:
      self.t = thrd.Thread(target = self.enQueueThread)
      self.t.start()
      assert(self.t is not None)
      print('enQueueThread started.')
    while True:
      if self.FinishGen and self.data_queue.empty():
        raise LOutOfRangeError()
        break
      else:
        X,y = self.data_queue.get()
        self.data_queue.task_done()
        yield X,y
      
  #def generate(self):
  #  count=0
  #  while count<self.epoch:
  #    count+=1
  #    fileList = self.fileList
  #    if self.shuffle:
  #      self.temp_fileList = copy.copy(fileList)
  #      np.random.shuffle(self.temp_fileList)
#
  #    num = len( self.temp_fileList)
  #    imax = int(num//self.batch_size)
  #    for i in range(imax):        
  #      batch_list = [it for it in  self.temp_fileList[i*self.batch_size:(i+1)*self.batch_size] ] 
  #      X,y = self.__data_generation(batch_list)
  #      yield X,y
  #  raise LOutOfRangeError()

  def __data_generation(self,batch_list):
    X=[]
    y=[]
    for it in batch_list:
      paths = it.split()
      imgP = paths[0]
      lblP = paths[1]
      tx = np.array(PIL.Image.open(imgP))
      ty = np.array(PIL.Image.open(lblP))
      tx,ty = self.__preprocess(tx,ty)
      X.append(tx)
      y.append(ty)
    X = np.array(X)
    y = np.array(y)
    return X,y

  def __one_hot(self,label):
    o_c = self.output_channels
    outLabel = np.zeros(shape=(label.shape[0],label.shape[1],o_c))
    for i in range(o_c):
      mask = (label==i)
      outLabel[:,:,i][mask] = 1
    #print(outLabel.shape)
    #assert(outLabel.ndim==4)
    assert(outLabel.shape[-1]==o_c)
    return outLabel

  def set_label_encoder(self,encoder):
    self.label_encoder = encoder

  def __encode_label(self,label):
    return self.label_encoder.transform(label)

  def __from_one_hot(self,label):
    encoder = skl.preprocessing.OneHotEncoder(self.output_channels,dtype = np.int32)

  def __preprocess(self,img,lbl):
    img = img.astype(np.float32)/255.0-0.5
    lbl = self.__encode_label(lbl)
    #lbl = np.expand_dims(lbl,-1)
    #lbl = self.__one_hot(lbl)
    return img,lbl

def getFileLists(listFilePath):
  fileList=[]
  with open(listFilePath,'r') as f:
    print(listFilePath+'--file open success')
    lines = f.read().split()
  for idx in range(0,len(lines)//2):
    fileList.append(lines[idx*2]+' '+lines[idx*2+1])
  return fileList

if __name__=='__main__':
  filepath = '/home/yelyu/Work/MyDLSolutions/UAV/vid1_all_type/vid1_train_4d_150sp/ListFile.txt'
  #fileList = getFileLists(filepath)
  #gen = DataGenerator(fileList,8,2,False)
  #gen.set_label_encoder(UAVImageColorEncoder())
  #print('begin gen')
  #for X,y in gen.generate():
  #  print(X.shape)
   # print(y.shape)
    #print(y[0,:10,:10,:])
    #plt.subplot(1,2,1)
    #plt.imshow(X[0,:,:,:])
    #plt.subplot(1,2,2)
    #plt.imshow(y[0,:,:,:])
    #plt.draw()
    #plt.show()