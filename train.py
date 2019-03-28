import dataset
import tensorflow as tf
import time
# from datetime import timedelta
import math
import random
import numpy as np
from numpy.random import seed
seed(10)
from tensorflow import set_random_seed
set_random_seed(20)


batch_size = 32
classes = ['trash','other']
num_classes = len(classes)
validation_size = 0.2 # 20% of the data will automatically be used for validation
img_size = 128
num_channels = 3
train_path='training_data'

data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)

print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size,img_size,num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


##Network graph params
filter_size_conv1 = 3 
num_filters_conv1 = 32

filter_size_conv2 = 3
num_filters_conv2 = 32

filter_size_conv3 = 3
num_filters_conv3 = 64
    
fc_layer_size = 1024

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))



def create_convolutional_layer(input,num_input_channels,conv_filter_size,num_filters):      
    ## We shall define the weights that will be trained using create_weights function. 3 3 3 32
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)

    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,filter=weights,strides=[1, 1, 1, 1], padding='SAME')
    layer += biases    
    layer = tf.nn.relu(layer)
    
    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
    return layer

def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def create_fc_layer(input,num_inputs,num_outputs,use_relu=True):    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    layer = tf.matmul(input, weights) + biases    
    layer=tf.nn.dropout(layer,keep_prob=0.7)    
    if use_relu:
        layer = tf.nn.relu(layer)        

    return layer


layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)
          
layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer()) 

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss,i):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0}--- iterations: {1}--- Training Accuracy: {2:>6.1%}, Validation Accuracy: {3:>6.1%},  Validation Loss: {4:.3f}"
    print(msg.format(epoch + 1,i, acc, val_acc, val_loss))

total_iterations = 0
display_step=data.train.num_examples/batch_size
saver = tf.train.Saver()
def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)        
        feed_dict_tr = {x: x_batch,y_true: y_true_batch}        

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(display_step) == 0: 
            feed_dict_val = {x: x_valid_batch,y_true: y_valid_batch}
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(display_step))    
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss,i)
            print(epoch, feed_dict_tr, feed_dict_val, val_loss,i)
            # saver.save(session, './trash_recognition_model/trash.ckpt',global_step=i) 


    total_iterations += num_iteration

train(num_iteration=1000)
'''
Training Epoch 1--- iterations: 0--- Training Accuracy:  40.6%, Validation Accuracy:  46.9%,  Validation Loss: 1.415
Training Epoch 2--- iterations: 49--- Training Accuracy:  62.5%, Validation Accuracy:  78.1%,  Validation Loss: 0.538
Training Epoch 3--- iterations: 98--- Training Accuracy:  78.1%, Validation Accuracy:  78.1%,  Validation Loss: 0.386
Training Epoch 4--- iterations: 147--- Training Accuracy:  81.2%, Validation Accuracy:  65.6%,  Validation Loss: 0.481
Training Epoch 5--- iterations: 196--- Training Accuracy:  75.0%, Validation Accuracy:  84.4%,  Validation Loss: 0.333
Training Epoch 6--- iterations: 245--- Training Accuracy:  78.1%, Validation Accuracy:  87.5%,  Validation Loss: 0.405
Training Epoch 7--- iterations: 294--- Training Accuracy:  87.5%, Validation Accuracy:  78.1%,  Validation Loss: 0.421
Training Epoch 8--- iterations: 343--- Training Accuracy:  78.1%, Validation Accuracy:  81.2%,  Validation Loss: 0.433
Training Epoch 9--- iterations: 392--- Training Accuracy:  90.6%, Validation Accuracy:  78.1%,  Validation Loss: 0.642
Training Epoch 10--- iterations: 441--- Training Accuracy:  84.4%, Validation Accuracy:  90.6%,  Validation Loss: 0.222
Training Epoch 11--- iterations: 490--- Training Accuracy:  81.2%, Validation Accuracy:  78.1%,  Validation Loss: 0.345
Training Epoch 12--- iterations: 539--- Training Accuracy:  84.4%, Validation Accuracy:  81.2%,  Validation Loss: 0.378
Training Epoch 13--- iterations: 588--- Training Accuracy:  87.5%, Validation Accuracy:  78.1%,  Validation Loss: 0.358
Training Epoch 14--- iterations: 637--- Training Accuracy:  87.5%, Validation Accuracy:  84.4%,  Validation Loss: 0.295
Training Epoch 15--- iterations: 686--- Training Accuracy: 100.0%, Validation Accuracy:  78.1%,  Validation Loss: 0.499
Training Epoch 16--- iterations: 735--- Training Accuracy:  87.5%, Validation Accuracy:  78.1%,  Validation Loss: 0.509
Training Epoch 17--- iterations: 784--- Training Accuracy:  87.5%, Validation Accuracy:  90.6%,  Validation Loss: 0.219


Number of files in Validation-set:      273
Training Epoch 1--- iterations: 0--- Training Accuracy:  62.5%, Validation Accuracy:  56.2%,  Validation Loss: 1.343
Training Epoch 2--- iterations: 33--- Training Accuracy:  50.0%, Validation Accuracy:  56.2%,  Validation Loss: 0.800
Training Epoch 3--- iterations: 66--- Training Accuracy:  59.4%, Validation Accuracy:  59.4%,  Validation Loss: 0.637
Training Epoch 4--- iterations: 99--- Training Accuracy:  46.9%, Validation Accuracy:  59.4%,  Validation Loss: 0.670
Training Epoch 5--- iterations: 132--- Training Accuracy:  56.2%, Validation Accuracy:  68.8%,  Validation Loss: 0.626
Training Epoch 6--- iterations: 165--- Training Accuracy:  65.6%, Validation Accuracy:  78.1%,  Validation Loss: 0.581
Training Epoch 7--- iterations: 198--- Training Accuracy:  68.8%, Validation Accuracy:  62.5%,  Validation Loss: 0.661
Training Epoch 8--- iterations: 231--- Training Accuracy:  71.9%, Validation Accuracy:  78.1%,  Validation Loss: 0.543
Training Epoch 9--- iterations: 264--- Training Accuracy:  62.5%, Validation Accuracy:  62.5%,  Validation Loss: 0.553
Training Epoch 10--- iterations: 297--- Training Accuracy:  71.9%, Validation Accuracy:  78.1%,  Validation Loss: 0.558
Training Epoch 11--- iterations: 330--- Training Accuracy:  56.2%, Validation Accuracy:  65.6%,  Validation Loss: 0.624
Training Epoch 12--- iterations: 363--- Training Accuracy:  53.1%, Validation Accuracy:  59.4%,  Validation Loss: 0.574
Training Epoch 13--- iterations: 396--- Training Accuracy:  75.0%, Validation Accuracy:  65.6%,  Validation Loss: 0.600
Training Epoch 14--- iterations: 429--- Training Accuracy:  71.9%, Validation Accuracy:  71.9%,  Validation Loss: 0.572
Training Epoch 15--- iterations: 462--- Training Accuracy:  71.9%, Validation Accuracy:  59.4%,  Validation Loss: 0.635
Training Epoch 16--- iterations: 495--- Training Accuracy:  75.0%, Validation Accuracy:  84.4%,  Validation Loss: 0.545
Training Epoch 17--- iterations: 528--- Training Accuracy:  71.9%, Validation Accuracy:  81.2%,  Validation Loss: 0.474
Training Epoch 18--- iterations: 561--- Training Accuracy:  68.8%, Validation Accuracy:  75.0%,  Validation Loss: 0.547
Training Epoch 19--- iterations: 594--- Training Accuracy:  71.9%, Validation Accuracy:  75.0%,  Validation Loss: 0.661
'''