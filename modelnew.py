# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 00:57:45 2018

@author: Akshay
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images # Returns np.array
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
dataset = pd.read_csv('train.csv')
X = dataset.iloc[:, 1:].values
X=np.array(X,dtype="float32")

t=pd.read_csv('test.csv')
test=t.iloc[:,:].values
test=np.array(test,dtype="float32")
y = dataset.iloc[:, 0].values

def model(features,labels,mode):
    regu=tf.contrib.layers.l2_regularizer(
    scale=0.01,
    scope=None)
    """labels=tf.contrib.layers.one_hot_encoding(labels,num_classes=10)"""
    conv1=tf.contrib.layers.conv2d(inputs=tf.reshape(features["x"],[-1,28,28,1]),num_outputs=32,kernel_size=5,stride=1,padding="SAME",activation_fn=tf.nn.relu,weights_regularizer=regu)
    
    maxpool2d1=tf.contrib.layers.max_pool2d(conv1,kernel_size=2,padding='SAME')
    
    conv2=tf.contrib.layers.conv2d(maxpool2d1,num_outputs=64,kernel_size=5,stride=1,padding="SAME")#,activation_fn=tf.nn.relu,weights_regularizer=regu)
    
    maxpool2d2=tf.contrib.layers.max_pool2d(conv2,kernel_size=2,padding='SAME')
    
    conv3=tf.contrib.layers.conv2d(maxpool2d2,num_outputs=64,kernel_size=5,stride=1,padding="SAME")#,activation_fn=tf.nn.relu,weights_regularizer=regu)
    
    maxpool2d3=tf.contrib.layers.max_pool2d(conv3,kernel_size=2,padding='SAME')
    flatten=tf.contrib.layers.flatten(maxpool2d3)
    
    
    dense=tf.contrib.layers.fully_connected(flatten,1024)#,weights_regularizer=regu)
    dropout=tf.nn.dropout(dense,keep_prob=0.8)
    logits=tf.contrib.layers.fully_connected(dropout,10)#,weights_regularizer=regu)
    predictions={"classes":tf.argmax(logits,1),"predictions":tf.nn.softmax(logits)}
    
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,predictions=predictions)
    
    cost=tf.losses.sparse_softmax_cross_entropy(labels=labels,logits=logits)
    
    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer=tf.train.AdamOptimizer()
        trainop=optimizer.minimize(cost,global_step=tf.train.get_global_step())
        return  tf.estimator.EstimatorSpec(mode=mode,loss=cost,train_op=trainop)
    
    eval_metric_ops={
            "accuracy":tf.metrics.accuracy(labels=labels,predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode,loss=cost,eval_metric_ops=eval_metric_ops)
    
def train_model(features,labels):
    classifier=tf.estimator.Estimator(model_fn=model,model_dir=r"modeldata4")
    inputs=tf.estimator.inputs.numpy_input_fn(x={"x":features},y=labels,batch_size=100,shuffle=True,num_epochs=50)
    with tf.device('/gpu:0'):
        a=classifier.train(input_fn=inputs)
    
    print(a)
  
def evaluate(features,labels):
    classifier=tf.estimator.Estimator(model_fn=model,model_dir=r"modeldata4")
    inputs=tf.estimator.inputs.numpy_input_fn(x={"x":features},y=labels,shuffle=False,batch_size=100,num_epochs=1)
    with tf.device('/gpu:0'):
        a=classifier.evaluate(input_fn=inputs)
        print(a)
def predict(features):
    classifier=tf.estimator.Estimator(model_fn=model,model_dir=r"modeldata4")
    inputs=tf.estimator.inputs.numpy_input_fn(x={"x":test},batch_size=1000,shuffle=False,num_epochs=1) 
    with tf.device('/gpu:0'):
        a=classifier.predict(input_fn=inputs,yield_single_examples=False)
        return a

#a=test.reshape([-1,28,28])
#b=a[0:25]
#fig=plt.figure()
#for i in enumerate(b):
#    
#    fig.add_subplot(5,5,i[0]+1)
#    plt.imshow(i[1])
#plt.show() 
#a=predict(test)   
#k=[]
#for i in a:
#    k=k+list(i["classes"])
#index=list(range(1,28001))
#alpha=np.array([index,k]).T
#
#np.savetxt('dataset6.csv', alpha.astype(np.int), fmt='%d', delimiter=',',header="ImageId,Label")    
#train_model(train_data,train_labels)
#evaluate(X,y)
#evaluate(eval_data,eval_labels)
        