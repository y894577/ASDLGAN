# from PIL import Image
# im1 = Image.open("F:\\testdata\\BOSSbase\\"+"1.pgm")
# im2 = Image.open("F:\\python\\DCGAN-for-Steganography-master\\stego\\"+"3.pgm")
# im3 = Image.open("F:\\python\\DCGAN-for-Steganography-master\\covers\\"+"1.pgm")    # 读取文件
# imbpp1 = Image.open("F:\\python\\DCGAN-for-Steganography-master\\bpp\\"+"3.pgm")
# imbpp2 = Image.open("F:\\python\\DCGAN-for-Steganography-master\\bpp\\"+"3.pgm")
# im5 = Image.open("F:\\python\\DCGAN-for-Steganography-master\\pros\\"+"1.pgm")
# im1.show()
# im2.show()
# im3.show()
# imbpp1.show()
# imbpp2.show()
# im5.show()
'''
img_names=[(i) for i in range(0, 10000, 3)]
print(img_names)
'''
#
# import torch
# print(torch.__version__)
#
# print(torch.version.cuda)
# print(torch.cuda.is_available())
#
# import os
# import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#
#
# print(tf.test.gpu_device_name())
# print(tf.test.is_gpu_available())
#
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#
# print(sess.list_devices())

import tensorflow as tf

# with tf.device('/cpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
with tf.device('/CPU:0'):
    a = tf.constant([1.0, 2.0, 3.0], shape=[3], name='a')
    b = tf.constant([1.0, 2.0, 3.0], shape=[3], name='b')
    c = a + b

# 注意：allow_soft_placement=True表明：计算设备可自行选择，如果没有这个参数，会报错。
# 因为不是所有的操作都可以被放在GPU上，如果强行将无法放在GPU上的操作指定到GPU上，将会报错。
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(tf.global_variables_initializer())
print(sess.run(c))

print(tf.test.is_gpu_available())

