import dataset
import tensorflow as tf
import time
import math
import random
import numpy as np
import pickle
from numpy.random import seed
seed(10)
from tensorflow import set_random_seed
set_random_seed(20)


batch_size = 32
classes = ['trash','other']
n_input = [128,128,3]
n_output=[len(classes)]
validation_size = 0.2 # 20% of the data will automatically be used for validation
train_path='training_data'

data = dataset.read_train_sets(train_path, n_input[0], classes, validation_size=validation_size)


print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

filter1=[3,3,3,32]
filter2=[3,3,32,32]
filter3=[3,3,32,64]
fc_size=1024

stddev=0.1
num_iteration=34
do_type=0 # 0:表示从0开始训练模型；1:表示用已有数据开始模型训练；
if do_type==0:
    weights={
        'wc1':tf.Variable(tf.random_normal(filter1,stddev=stddev)),
        'wc2':tf.Variable(tf.random_normal(filter2,stddev=stddev)),
        'wc3':tf.Variable(tf.random_normal(filter3,stddev=stddev)),
        'wcd1':tf.Variable(tf.random_normal([int(n_input[0]*n_input[1]/64*filter3[3]),fc_size],stddev=stddev)),
        'wcd2':tf.Variable(tf.random_normal([fc_size,n_output[0]],stddev=stddev))
    }
    biases={
        'wb1':tf.Variable(tf.constant(0.05,shape=[filter1[3]])),
        'wb2':tf.Variable(tf.constant(0.05,shape=[filter2[3]])),
        'wb3':tf.Variable(tf.constant(0.05,shape=[filter3[3]])),
        'wbd1':tf.Variable(tf.constant(0.05,shape=[fc_size])),
        'wbd2':tf.Variable(tf.constant(0.05,shape=n_output))
    }
else:
    fr=open('./model/trash.txt','rb')
    w,b=pickle.load(fr)
    fr.close()
    weights={
        'wc1':tf.Variable(w['wc1']),
        'wc2':tf.Variable(w['wc2']),
        'wc3':tf.Variable(w['wc3']),
        'wcd1':tf.Variable(w['wcd1']),
        'wcd2':tf.Variable(w['wcd2'])
    }
    biases={
        'wb1':tf.Variable(b['wb1']),
        'wb2':tf.Variable(b['wb2']),
        'wb3':tf.Variable(b['wb3']),
        'wbd1':tf.Variable(b['wbd1']),
        'wbd2':tf.Variable(b['wbd2'])
    }


def conv_calculate(x,w,b,keepratio):
    conv1=tf.nn.conv2d(x,w['wc1'],strides=[1,1,1,1],padding='SAME')
    conv1=tf.nn.relu(tf.nn.bias_add(conv1,b['wb1']))
    pool1=tf.nn.max_pool(conv1,strides=[1,2,2,1],ksize=[1,2,2,1],padding='SAME')
    dropout1=tf.nn.dropout(pool1,keepratio)

    conv2=tf.nn.conv2d(dropout1,w['wc2'],strides=[1,1,1,1],padding='SAME')
    conv2=tf.nn.relu(tf.nn.bias_add(conv2,b['wb2']))
    pool2=tf.nn.max_pool(conv2,strides=[1,2,2,1],ksize=[1,2,2,1],padding='SAME')
    dropout2=tf.nn.dropout(pool2,keepratio)

    conv3=tf.nn.conv2d(dropout2,w['wc3'],strides=[1,1,1,1],padding='SAME')
    conv3=tf.nn.relu(tf.nn.bias_add(conv3,b['wb3']))
    pool3=tf.nn.max_pool(conv3,strides=[1,2,2,1],ksize=[1,2,2,1],padding='SAME')
    dropout3=tf.nn.dropout(pool3,keepratio)

    dense=tf.reshape(dropout3,[-1,w['wcd1'].get_shape().as_list()[0]]) 
    fc1=tf.nn.relu(tf.add(tf.matmul(dense,w['wcd1']),b['wbd1']))
    fc1=tf.nn.dropout(fc1,keepratio)
    return tf.add(tf.matmul(fc1,w['wcd2']),b['wbd2'])    

x = tf.placeholder(tf.float32, shape=[None, n_input[0],n_input[1],n_input[2]], name='x')
y = tf.placeholder(tf.float32, shape=[None, n_output[0]], name='y')
keepratio=tf.placeholder(tf.float32)

score=conv_calculate(x,weights,biases,keepratio)
pred = tf.nn.softmax(score,name='pred')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=score,labels=y))
opti = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
corr=tf.equal(tf.argmax(score,1),tf.argmax(y,1))
accr = tf.reduce_mean(tf.cast(corr, tf.float32))

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss,i):
    tr_acc = session.run(accr, feed_dict=feed_dict_train)
    val_acc = session.run(accr, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0}--- iterations: {1}--- Training Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
    print(msg.format(epoch + 1,i, tr_acc, val_acc, val_loss))

total_iterations = 0
display_step=data.train.num_examples/batch_size # 1024/32=32

session = tf.Session()
session.run(tf.global_variables_initializer()) 
saver = tf.train.Saver()
def train():
    global total_iterations    
    for i in range(total_iterations,total_iterations + num_iteration):
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)        
        feed_dict_tr = {x: x_batch,y: y_true_batch,keepratio:0.7}        

        session.run(opti, feed_dict=feed_dict_tr)
        if i % int(display_step) == 0: 
            feed_dict_val = {x: x_valid_batch,y: y_valid_batch,keepratio:1.0}
            val_loss = session.run(loss, feed_dict=feed_dict_val)
            epoch = int(i / int(display_step))    
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss,i)
            saver.save(session, './model/trash.ckpt',global_step=i) 
            fw=open('./model/trash.txt','wb')
            pickle.dump((session.run(weights),session.run(biases)),fw)
            fw.close()
    total_iterations += num_iteration

train()

'''

'''