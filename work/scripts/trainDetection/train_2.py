import tensorflow as tf
import numpy as np
import cv2,os,pickle

do_action=0  # 0:表示从0开始训练模型；1:表示用已有数据开始模型训练；
batch_size = 16  # 训练batch大小
train_steps = 2000  # 训练step数
img_width = 270
img_height = 480
num_channels = 3
num_objects = 16
conv_size = 3  # 卷积尺寸
regularizer_lambda = 0.0001  # L2正则化权重
obj_detection_threshold = 0.5  # 目标识别概率超过该门限表示发现目标
path='D:/python/pythonDemo/cifarTest/tf\work/trash_detection/imgs/tfRecords/rubbish_trash/'

# 建立权重参数：
stddev=0.1
if do_action==0:
    weight={
        'wc1':tf.Variable(tf.random_normal([conv_size, conv_size, num_channels, 64],stddev=stddev)),
        'wc2':tf.Variable(tf.random_normal([conv_size, conv_size, 64, 128],stddev=stddev)),
        'wc3':tf.Variable(tf.random_normal([conv_size, conv_size, 128, 256],stddev=stddev)),
        'wc4':tf.Variable(tf.random_normal([conv_size, conv_size, 256, 512],stddev=stddev)),
        'wc5':tf.Variable(tf.random_normal([1, 1, 512, 64],stddev=stddev)),
        'wc6':tf.Variable(tf.random_normal([1, 1, 64, 8],stddev=stddev)),
        'wc7':tf.Variable(tf.random_normal([1, 1, 8, 1],stddev=stddev))
    }
    biases={
        'bc1':tf.Variable(tf.random_normal([64],stddev=stddev)),
        'bc2':tf.Variable(tf.random_normal([128],stddev=stddev)),
        'bc3':tf.Variable(tf.random_normal([256],stddev=stddev)),
        'bc4':tf.Variable(tf.random_normal([512],stddev=stddev)),
        'bc5':tf.Variable(tf.random_normal([64],stddev=stddev)),
        'bc6':tf.Variable(tf.random_normal([8],stddev=stddev)),
        'bc7':tf.Variable(tf.random_normal([1],stddev=stddev))
    }
elif do_action==1:
    fr=open(path+'weight_biases.txt','rb')
    w,b=pickle.load(fr)
    fr.close()
    weight={
        'wc1':tf.Variable(w['wc1']),
        'wc2':tf.Variable(w['wc2']),
        'wc3':tf.Variable(w['wc3']),
        'wc4':tf.Variable(w['wc4']),
        'wc5':tf.Variable(w['wc5']),
        'wc6':tf.Variable(w['wc6']),
        'wc7':tf.Variable(w['wc7'])
    }
    biases={
        'bc1':tf.Variable(b['bc1']),        
        'bc2':tf.Variable(b['bc2']),
        'bc3':tf.Variable(b['bc3']),
        'bc4':tf.Variable(b['bc4']),
        'bc5':tf.Variable(b['bc5']),
        'bc6':tf.Variable(b['bc6']),
        'bc7':tf.Variable(b['bc7'])
    }
# 前向传播 建立graph
def multiLayer_perceptron(x,w,b):
    conv1=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w['wc1'], strides=[1, 1, 1, 1], padding="VALID"), b['bc1']))
    conv1_maxpool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv2=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1_maxpool, w['wc2'], strides=[1, 1, 1, 1], padding="VALID"), b['bc2']))
    conv2_maxpool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv3=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2_maxpool, w['wc3'], strides=[1, 1, 1, 1], padding="VALID"), b['bc3']))
    conv3_maxpool = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv4=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3_maxpool, w['wc4'], strides=[1, 1, 1, 1], padding="VALID"), b['bc4']))
    conv4_maxpool = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    conv5=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4, w['wc5'], strides=[1, 1, 1, 1], padding="VALID"), b['bc5']))

    conv6=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv5, w['wc6'], strides=[1, 1, 1, 1], padding="VALID"), b['bc6']))

    conv7=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv6, w['wc7'], strides=[1, 1, 1, 1], padding="VALID"), b['bc7']))
    conv7_shape = conv7.get_shape().as_list()  # 取得tensor的shape，并且转为list
    conv7_shape = conv7_shape[1]*conv7_shape[2]*conv7_shape[3]
    # 为下层全连接计算，转换tensor的shape为[batch_size, nodes]，batch_size即为conv7_shape[0]
    conv7 = tf.reshape(conv7, shape=[batch_size, conv7_shape])

    # 全连接层一
    wf1=tf.Variable(tf.random_normal([conv7_shape,256],stddev=stddev))
    bf1=tf.Variable(tf.random_normal([256],stddev=stddev))
    fc1=tf.nn.relu(tf.matmul(conv7, wf1)+bf1)
    # 全连接层权重正则化
    tf.add_to_collection("loss", tf.contrib.layers.l2_regularizer(regularizer_lambda)(wf1))

    # 全连接层二
    wf2=tf.Variable(tf.random_normal([256,num_objects * 3],stddev=stddev))
    bf1=tf.Variable(tf.random_normal([num_objects * 3],stddev=stddev))
    fc2=tf.nn.relu(tf.matmul(fc1, wf2)+bf1)
    return fc2,wf2

# 建立样本tensor
x = tf.placeholder(dtype=tf.float32, shape=[batch_size, img_height, img_width, num_channels], name="x_input")
y_ = tf.placeholder(dtype=tf.float32, shape=[batch_size, num_objects, 3], name='y_input')
fc2,wf2=multiLayer_perceptron(x,weight,biases)
# 为了下层计算损失函数，转换tensor的shape为[batch_size, num_objects, 3]，batch_size即为fc2[0]
y = tf.reshape(fc2, shape=[batch_size, num_objects, 3])
# 全连接层权重正则化
tf.add_to_collection("loss", tf.contrib.layers.l2_regularizer(regularizer_lambda)(wf2))

# 计算样本损失
loss_obj = tf.reduce_mean(
    tf.square(y[:, :, 0]-y_[:, :, 0]) +  # 概率损失
    0.5 * (tf.square(y[:, :, 1] - y_[:, :, 1]) + tf.square(y[:, :, 2] - y_[:, :, 2])))  # 坐标损失
tf.add_to_collection("loss", loss_obj)

# 将样本损失和全连接层权重正则损失相加
loss = tf.add_n(tf.get_collection("loss"))

# 定义学习率，随loss的范围而逐步减小，数值选择为试验所得
rate = tf.Variable(0.001, trainable=False)
tf.assign(rate, tf.where(tf.greater(loss, 0.02), 0.001,
                            (tf.where(tf.greater(loss, 0.01), 0.0005,
                                    tf.where(tf.greater(loss, 0.005), 0.00001,
                                            tf.where(tf.greater(loss, 0.001), 0.000005, 0.000001))))))

# 定义训练step
train_step = tf.train.AdamOptimizer(rate).minimize(loss)

###############################################################
# 读取全部的TFRecord文件

# 获取文件列表
print(os.path.join(path, "*.tfrecords"),'kkkkkkkkkkkkkkkkkkk') # D:/python/pythonDemo/cifarTest/tf\work/tf_record2/youshang/*.tfrecords kkkkkkkkkkkkkkkkkkk
files = tf.train.match_filenames_once(os.path.join(path, "*.tfrecords"))
# # 创建输入队列
queue = tf.train.string_input_producer(files, shuffle=True)

# 建立TFRecordReader
reader = tf.TFRecordReader()
_, serialized_example = reader.read(queue)
# serialized_example = tf.data.TFRecordDataset(files)
print(serialized_example)# Tensor("ReaderReadV2:1", shape=(), dtype=string)
rec_features = tf.parse_single_example(  # 返回字典，字典key值即features参数中的key值
    serialized_example,
    features={
        # 写入时shape固定的数值用FixedLenFeature
        "filename": tf.FixedLenFeature(shape=[], dtype=tf.string),  # 由于只有1个值也可以用shape[1]，返回list
        "width": tf.FixedLenFeature(shape=[], dtype=tf.int64),
        "height": tf.FixedLenFeature(shape=[], dtype=tf.int64),
        "data": tf.FixedLenFeature(shape=[], dtype=tf.string),
        # 写入时shape不固定的数值，读出时用VarLenFeature，读出为SparseTensorValue类对象
        "object/label": tf.VarLenFeature(dtype=tf.string),
        "object/bbox/xmin": tf.VarLenFeature(dtype=tf.int64),
        "object/bbox/xmax": tf.VarLenFeature(dtype=tf.int64),
        "object/bbox/ymin": tf.VarLenFeature(dtype=tf.int64),
        "object/bbox/ymax": tf.VarLenFeature(dtype=tf.int64),
    }
)

# 模型保存
saver = tf.train.Saver()
# 保存最小的目标损失值，不包含正则损失
loss_min = 0.01

# 运行Session
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())  # 初始化局部变量，用于取得文件列表
    print(sess.run(files))  # 打印文件列表
    sess.run(tf.global_variables_initializer())  # 初始化全局变量

    # 用子线程启动TFRecord的输入队列
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # 训练循环
    for i in range(train_steps):
        # 此处定义，用于在此层使用
        rec = None

        # 训练数据初值
        x_input = np.zeros(shape=(batch_size, img_height, img_width, num_channels), dtype=float)
        y_input = np.zeros(shape=(batch_size, num_objects, 3), dtype=float)

        # 建立训练样本batch
        for j in range(batch_size):
            # 读取一个TFRecord文件，直到该TFRecord内部包含pig标签
            label_list = []

            # TFRecord中返回bytes类型，需要用str.encode转成bytes用于比较
            while "trashCan".encode("utf-8") not in label_list:
                rec = sess.run(rec_features)
                label_list = rec["object/label"].values
            while "rubbishDump".encode("utf-8") not in label_list:
                rec = sess.run(rec_features)
                label_list = rec["object/label"].values
            # 取得图像数据
            if i<2:
                print(label_list)
            img = np.fromstring(rec["data"], np.uint8)
            img = np.reshape(img, (img_height, img_width, num_channels))

            # 至此已取得包含pig标签的标注数据
            # 训练数据组装

            # 取得图像数据
            x_input[j, :, :, :] = img/255.0  # /255.0进行uint8归一化

            # 取得样本结果
            obj_ind = 0  # y_input中的目标索引
            for index, label in enumerate(rec["object/label"].values):  # 用enumerate建立列表的遍历对象
                # TFRecord中返回bytes类型，需要用str.encode转成bytes用于比较
                if label in "trashCan-rubbishDump".encode("utf-8"):  # 发现目标label
                    # 目标概率
                    y_input[j, obj_ind, 0] = 1.0
                    # x轴中心点
                    y_input[j, obj_ind, 1] = (rec["object/bbox/xmin"].values[index] + rec["object/bbox/xmax"].values[index]) / 2.0 / img_width
                    # y轴中心点
                    y_input[j, obj_ind, 2] = (rec["object/bbox/ymin"].values[index] + rec["object/bbox/ymax"].values[index]) / 2.0 / img_height
                    # 目标索引递增
                    obj_ind += 1

        # 运行当前batch训练
        sess.run(train_step, feed_dict={x: x_input, y_: y_input})
        # print("train_step: {0}".format(i))

        # 在训练样本中检查训练成果
        if i % 10 == 0:
            loss_output = sess.run(loss_obj, feed_dict={x: x_input, y_: y_input})
            print("Step {1} train_loss: {0}".format(loss_output, i))

            # 保存损失值最小的模型
            if loss_output < loss_min:
                loss_min = loss_output
                saver.save(sess, os.path.join(path, "model_loss_min.ckpt"))
                fw=open(path+'weight_biases'+str(i)+'.txt','wb')
                pickle.dump(sess.run([weight,biases]),fw)
                fw.close()
                print("LOSS MIN model saved: {0}".format(loss_min))
            else:
                print("LOSS MIN: {0}".format(loss_min))

            # 用OpenCV显示处理图像
            y_output = sess.run(y, feed_dict={x: x_input, y_: y_input})

            # 显示batch中的0和1索引的图像
            img0 = x_input[0, :, :, :]
            img0 = img0*255.0
            img0 = img0.astype(np.uint8)

            # 绘制输入中心点
            for obj, p in enumerate(y_input[0, :, 0]):
                if p > 0.5:
                    img0 = cv2.circle(img0, (
                    int(y_input[0, obj, 1] * img_width), int(y_input[0, obj, 2] * img_height)),
                                        radius=7, color=(0,0,255), thickness=1)

            # 遍历图像0中的全部目标
            print(y_output[0, :, :])
            cnt = 0
            for obj, p in enumerate(y_output[0, :, 0]):
                if p > obj_detection_threshold:  # 目标概率大于门限，则认为发现目标
                    img0 = cv2.circle(img0, (int(y_output[0, obj, 1]*img_width), int(y_output[0, obj, 2]*img_height)), radius=20, color=(255, 0, 0), thickness=2)# thickness=cv2.FILLED
                    cnt += 1

            img0 = cv2.putText(img0,
                                "{0}".format(cnt),
                                (5, img_height-5),  # 左下角显示目标数目
                                cv2.FONT_HERSHEY_PLAIN,
                                1,
                                (0, 255, 0))

            # 显示图像
            cv2.imshow("image", img0[:,:,[2,1,0]])
            cv2.waitKey(10)

    coord.request_stop()  # 结束线程
    coord.join(threads)  # 等待线程结束
