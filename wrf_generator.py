"""
This code implement a generator which can produce embedding probility map
这个文件是生成器，可以生成嵌入率图
"""

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.training import moving_averages
import torch
import six
from SRMfilters import *
from ops import *


class WrfGenerator(object):
    def __init__(self, hps, mode='train'):
        self.hps = hps
        self.images = tf.placeholder(tf.float32, shape=[hps['batch_size'], hps['image_size'], hps['image_size'], 1])
        # dtype:数据类型 shape 数据形状 name 名称
        self.mode = mode

        self.step = tf.contrib.framework.get_or_create_global_step()

        self._set_tes_weights()
        self._build_model()
        self._init = tf.global_variables_initializer()

    def _build_model(self):
        x = self.images
        image_size = self.hps['image_size']
        stride = self._stride_array(1)
        conv_units_num = 6
        atro2_units_num = 2
        atro4_units_num = 0
        # atro8_units_num = 2
        # atro16_units_num = 2

        # 2D卷积，卷积核3，通道30
        with tf.variable_scope('g_init'):
            # 5x5x1 -> 5x5x30 kernel去噪
            x = self._res_unit(x, 1, 30, stride)
        # x=self._batch_norm('g_init_bn', x)
        # x=self._relu('g_init_relu', x)

        # 残差网络训练六次
        # residual block
        for i in six.moves.range(1, conv_units_num + 1):
            with tf.variable_scope('g_res%d' % i):
                x = self._res_unit(x, 30, 30, stride)

        # 空洞卷积
        # Atrous residual block
        for i in six.moves.range(1, atro2_units_num + 1):
            with tf.variable_scope('g_atro_2_%d' % i):
                x = self._res_unit_atrous(x, 30, 30, 2, stride)
        for i in six.moves.range(1, atro4_units_num + 1):
            with tf.variable_scope('g_atro_4_%d' % i):
                x = self._res_unit_atrous(x, 30, 30, 4, stride)

        # 2D卷积
        # channels 30 -> 1
        x = self._conv('g_final', x, 1, 30, 1, stride)
        self.first = x
        x = tf.nn.sigmoid(x) - 0.5
        x = self._relu('final', x)

        # 获取容量 用于train
        self.pro = x
        self.capacity = self._get_cap(self.pro)

        # 重新整理x （512x512,1)
        x = tf.reshape(x, [-1, 1])

        self.rand_shape = [int(x.shape[0]), 1]
        self.rand = tf.placeholder(tf.float32, shape=self.rand_shape)  # , minval=0,maxval=1, dtype=tf.float32)

        x = tf.concat([x, self.rand], 1)
        # 拼接x和rand
        # t1 = [[1, 2, 3], [4, 5, 6]]
        # t2 = [[7, 8, 9], [10, 11, 12]]
        # tf.concat([t1, t2], 1)   [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

        # 输入tes
        x = self._tes(x)

        # 改变图M
        x = tf.reshape(x, [-1, image_size, image_size, 1])
        # bpp是modification map，用于将cover隐写成stego
        self.bpp = x
        # x与原图相加，得到嵌入后图像
        self.stego = tf.add(x, self.images)

    def _gen_initconv(self, name, x):
        with tf.variable_scope(name):
            n = 5 * 5 * 30
            SRMfilters = SRM()
            kernel = tf.get_variable('DW', initializer=SRMfilters.get_filters(), trainable=False)
            return tf.nn.conv2d(x, kernel, self._stride_array(1), padding='SAME')

    def _get_cap(self, pro):
        pro_1 = pro / 2.0
        # print('pro_1',pro_1)
        pro_0 = 1.0 - pro
        # print('pro_0', pro_0)
        pos1 = -1.0 * pro_1 * tf.log(pro_1 + 1e-20) / tf.log(2.0)
        neg1 = -1.0 * pro_1 * tf.log(pro_1 + 1e-20) / tf.log(2.0)
        zero = -1.0 * pro_0 * tf.log(pro_0) / tf.log(2.0)
        cap = pos1 + neg1 + zero
        cap = tf.reduce_sum(cap, axis=[1, 2, 3])
        return cap

    def _res_unit_atrous(self, x, in_filter, out_filter, dilation, stride):
        bn1 = batch_norm(name='bn1')
        bn2 = batch_norm(name='bn2')
        orig_x = x
        with tf.variable_scope('sub1'):
            x = self._atrous_conv('atro', x, 3, in_filter, out_filter, dilation)
            x = bn1(x, train=self.hps['mode'])
            x = self._relu('relu', x)
        with tf.variable_scope('sub2'):
            x = self._atrous_conv('atro', x, 3, out_filter, out_filter, dilation)
            x = bn2(x, train=self.hps['mode'])

            x += orig_x
            x = self._relu('relu', x)
        return x

        # in_filter 进入的深度
        # out_filter 输出的深度

    def _res_unit(self, x, in_filter, out_filter, stride):
        # 归一化处理
        bn1 = batch_norm(name='bn1')
        bn2 = batch_norm(name='bn2')

        orig_x = x
        # 如果进入和输出的filter深度不一样，先用1x1卷积核进行卷积，得到深度为out_filter的输出层
        if (in_filter != out_filter):
            orig_x = self._conv('orig_conv', orig_x, 1, in_filter, out_filter, self._stride_array(1))

        # 2层卷积
        with tf.variable_scope('sub1'):
            x = self._conv('conv', x, 5, in_filter, out_filter, stride)
            x = bn1(x, train=self.hps['mode'])
            x = self._relu('relu', x)

        with tf.variable_scope('sub2'):
            x = self._conv('conv', x, 5, out_filter, out_filter, stride)
            x = bn2(x, train=self.hps['mode'])
            x += orig_x
            x = self._relu('relu', x)

        return x

    def _stride_array(self, stride):
        return [1, stride, stride, 1]

    def _relu(slef, name, x):
        with tf.variable_scope(name) as scope:
            return tf.nn.relu(x)

    def _atrous_conv(self, name, x, filter_size, in_filters, out_filters, dilation, pad='SAME'):
        '''2D膨胀卷积'''
        # 基于不同的输入和卷积核，定义膨胀卷积函数
        convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=pad)
        with tf.variable_scope(name) as scope:
            n = filter_size * filter_size * out_filters
            # 获取或新建卷积核，正态随机初始化
            kernel = tf.get_variable(
                'DW',
                [filter_size, filter_size, in_filters, out_filters],
                tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.01))  # np.sqrt(2.0/n)))

            output = convolve(x, kernel)

            return output

        # 2D卷积

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            # 获取或新建卷积核，正态随机初始化
            kernel = tf.get_variable(
                'DW',
                [filter_size, filter_size, in_filters, out_filters],
                tf.float32,
                initializer=tf.random_normal_initializer(stddev=0.01))  # np.sqrt(2.0/n)))
            # 计算卷积
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _set_tes_weights(self):
        # 加载TES
        f = open('tes_ckpt/weights.txt', 'r')
        a = f.read()
        dict = eval(a)
        f.close()
        with tf.variable_scope('tes'):
            for i in range(1, 3):
                for j in range(1, 4):
                    with tf.variable_scope('n%d_layer%d' % (i, j)):
                        weights = tf.get_variable(name='weight', initializer=dict['n%d_layer%d' % (i, j)]['w'],
                                                  trainable=False)
                        bias = tf.get_variable(name='bias', initializer=dict['n%d_layer%d' % (i, j)]['b'],
                                               trainable=False)

    def _tes(self, x):
        with tf.variable_scope('tes', reuse=True):
            n1 = x
            n2 = x
            for j in range(1, 4):
                with tf.compat.v1.variable_scope('n1_layer%d' % j, reuse=True):
                    weights = tf.compat.v1.get_variable(name='weight')
                    bias = tf.get_variable(name='bias')
                    n1 = tf.nn.sigmoid(tf.matmul(n1, weights) + bias)
                    if j < 3:
                        n1 -= 0.5
            for j in range(1, 4):
                with tf.compat.v1.variable_scope('n2_layer%d' % j, reuse=True):
                    weights = tf.get_variable(name='weight')
                    bias = tf.get_variable(name='bias')
                    n2 = tf.nn.sigmoid(tf.matmul(n2, weights) + bias)
                    if j < 3:
                        n2 -= 0.5
                    else:
                        n2 -= 1
            return n1 + n2
