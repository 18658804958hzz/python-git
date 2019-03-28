import tensorflow as tf
import numpy as np
import os,glob,cv2
import matplotlib.pyplot as plt
# import sys,argparse



image_size=128
num_channels=3
images = []
imgs_before=[]
fileNames=[]
# path =r'D:\python\pythonDemo\cifarTest\tf\work\cnn4_1\testing_data2\111\\' #'testing_data2\\'
path ='D:/python/pythonDemo/cifarTest/tf/work/tf_record/test/' #'testing_data2\\'

classify_label = ['trash','other']
meta_graph_path='trash_recognition_model/trash.ckpt-784.meta'
model_data_path='trash_recognition_model/trash.ckpt-784'
# meta_graph_path='trash_recognition_model/trash.ckpt-594.meta'
# model_data_path='trash_recognition_model/trash.ckpt-594'
files=os.listdir(path)
for fileName in files:    
    image = cv2.imread(path+fileName)
    imgs_before.append(image)
    fileNames.append(fileName)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)# inter_linear
    images.append(image)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
images = np.multiply(images, 1.0/255.0) 
#The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
x_batch = images.reshape(-1, image_size,image_size,num_channels)

## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph(meta_graph_path)
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, model_data_path)

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0") 
y_test_images = np.zeros((len(x_batch), 2)) 

feed_dict_testing = {x: x_batch, y_true: y_test_images}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
for k,v in enumerate(result):
    # print(k)
    print(fileNames[k],' predicted val is '+classify_label[v.argmax()])
    plt.title(fileNames[k]+' predicted val is '+classify_label[v.argmax()])
    # print(imgs_before[k].shape)# bgr
    plt.imshow(imgs_before[k][:,:,[2,1,0]])
    plt.show()
    # plt.imshow(imgs_before[k][:,:,[1,2,0]])
    # plt.show()
    # plt.imshow(imgs_before[k][:,:,[0,1,2]])
    # plt.show()
    # assert 0
    # result is of this format [probabiliy_of_rose probability_of_sunflower]
    # trash [1 0]
