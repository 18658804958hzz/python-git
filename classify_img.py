import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
import cv2
import os

classify_before=r'D:\python\pythonDemo\cifarTest\tf\work\cnn4_1\img_source\\'
# classify_before=r'D:\python\pythonDemo\cifarTest\tf\work\cnn4_1\training_data\trash\\'
classify_after=r'D:\python\pythonDemo\cifarTest\tf\work\cnn4_1\training_data\\'
meta_graph_path='trash_recognition_model/trash.ckpt-495.meta'
model_data_path='trash_recognition_model/trash.ckpt-495'
trash_count=0
other_count=0
total=0
batch_size=200
img_before_resize=[]
img_size=128,128,3

all_imgs_path=os.listdir(classify_before)
sess=tf.Session()
saver=tf.train.import_meta_graph(meta_graph_path)
saver.restore(sess,model_data_path)
graph=tf.get_default_graph()
x=graph.get_tensor_by_name('x:0')
y_pred=graph.get_tensor_by_name('y_pred:0')
y_true=graph.get_tensor_by_name('y_true:0')

def load_imgs():
    global img_before_resize
    batch_path=all_imgs_path[total:total+batch_size]
    img_after_resize=[]
    img_before_resize=[]
    for path in batch_path:
        img=cv2.imread(classify_before+path)
        img_before_resize.append(img)
        img=cv2.resize(img,img_size[0:2])
        img_after_resize.append(img)
    img_after_resize = np.array(img_after_resize, dtype=np.uint8)
    img_after_resize = img_after_resize.astype('float32')
    img_after_resize = np.multiply(img_after_resize, 1.0/255.0) 
    return img_after_resize.reshape(batch_size, img_size[0],img_size[1],img_size[2])

def copy_write_imgs(imgs):
    global trash_count,other_count
    feed_dict={x:imgs,y_true:np.zeros((batch_size,2))}
    result=sess.run(y_pred,feed_dict=feed_dict)
    true_flag=np.argmax(result,1)
    for k,v in enumerate(true_flag):
        # plt.title(k)
        # plt.imshow(img_before_resize[k][:,:,[2,1,0]])
        # plt.show()
        if v==0:
            trash_count+=1
            cv2.imwrite(classify_after+'trash2\\'+str(trash_count)+'.jpg',img_before_resize[k])
        else:
            other_count+=1
            cv2.imwrite(classify_after+'other2\\'+str(other_count)+'.jpg',img_before_resize[k])   

total_batch=int(np.floor(len(all_imgs_path)/batch_size))
for i in range(total_batch):
    imgs=load_imgs()
    copy_write_imgs(imgs)
    total+=batch_size
    print(total,'张图片写入成功')
    num=input()
    if num=='0':
        assert 0    


