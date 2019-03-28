import tensorflow as tf  # 导入TensorFlow
import cv2  # 导入OpenCV
import os  # 用于文件操作
import glob  # 用于遍历文件夹内的xml文件
import xml.etree.ElementTree as ET  # 用于解析xml文件

# xml文件所在的路劲
sourcePath=r'D:\python\pythonDemo\cifarTest\tf\work\trash_detection\imgs\tfRecords\trashCan\\'
# tfRecords文件保存路劲
savePath=r'D:\python\pythonDemo\cifarTest\tf\work\trash_detection\imgs\tfRecords\trashCan\\'

# 将LabelImg标注的图像文件和标注信息保存为TFRecord
def xml2tfRecord(sourcePath,savePath):
    """
    :param path: LabelImg标识文件的路径，及生成的TFRecord文件路径
    """
    # 遍历文件夹内的全部xml文件，1个xml文件描述1个图像文件的标注信息
    for f in glob.glob(sourcePath + "/*.xml"):
        # 解析xml文件
        try:
            tree = ET.parse(f)
        except FileNotFoundError:
            print("无法找到xml文件: "+f)
            return False
        except ET.ParseError:
            print("无法解析xml文件: "+f)
            return False
        else:  # ET.parse()正确运行
            # 取得xml根节点
            root = tree.getroot()

            # 取得图像路径和文件名
            img_name = root.find("filename").text
            img_path = root.find("path").text

            # 取得图像宽高
            img_width = int(root.find("size")[0].text)
            img_height = int(root.find("size")[1].text)

            # 取得所有标注object的信息
            label = []  # 类别名称
            xmin = []
            xmax = []
            ymin = []
            ymax = []

            # 查找根节点下全部名为object的节点
            for m in root.findall("object"):
                xmin.append(int(m[4][0].text))
                xmax.append(int(m[4][2].text))
                ymin.append(int(m[4][1].text))
                ymax.append(int(m[4][3].text))
                # 用encode将str类型转为bytes类型，相应的用decode由bytes转回str类型
                label.append(m[0].text.encode("utf-8"))

            # 至少有1个标注目标
            if len(label) > 0:
                # 用OpenCV读出图像原始数据，未压缩数据
                data = cv2.imread(img_path, cv2.IMREAD_COLOR)

                # 将OpenCV的BGR格式转为RGB格式
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

                # 建立Example
                example = tf.train.Example(features=tf.train.Features(feature={
                    # 用encode将str类型转为bytes类型
                    # 以下各feature的shape固定，读出时必须使用tf.FixedLenFeature
                    "filename": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name.encode("utf-8")])),
                    "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[img_width])),
                    "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[img_height])),
                    "data": tf.train.Feature(bytes_list=tf.train.BytesList(value=[data.tostring()])),  # 图像数据ndarray转化成bytes类型
                    # 以下各feature的shape不固定，，读出时必须使用tf.VarLenFeature
                    "object/label": tf.train.Feature(bytes_list=tf.train.BytesList(value=label)),
                    "object/bbox/xmin": tf.train.Feature(int64_list=tf.train.Int64List(value=xmin)),
                    "object/bbox/xmax": tf.train.Feature(int64_list=tf.train.Int64List(value=xmax)),
                    "object/bbox/ymin": tf.train.Feature(int64_list=tf.train.Int64List(value=ymin)),
                    "object/bbox/ymax": tf.train.Feature(int64_list=tf.train.Int64List(value=ymax))
                }))

                # 建立TFRecord的写对象
                # img_name.split('.')[0]用于去掉扩展名，只保留文件名
                with tf.python_io.TFRecordWriter(os.path.join(savePath, img_name.split('.')[0]+".tfrecords")) as writer:
                    # 数据写入TFRecord文件
                    writer.write(example.SerializeToString())

                    # 结束
                    print("生成TFRecord文件: " + os.path.join(savePath, img_name.split('.')[0]+".tfrecords"))
            else:
                print("xml文件{0}无标注目标".format(f))
                return False

    print("完成全部xml标注文件的保存")
    return True

if __name__ == "__main__":
    xml2tfRecord(sourcePath,savePath)