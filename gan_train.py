# 0:2
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.python.training import moving_averages
import six
from reader import *
from ops import *
from tlu_discriminator import *
from wrf_generator import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.allow_soft_placement = True
sess = tf.Session(config=config)

"""
Set hyperparameters
"""
hps = {}
hps['image_size'] = 512
hps['g_alph'] = 10 ** 7
hps['g_beta'] = 1
hps['capacity'] = 0.4
hps['g_lrn_rate'] = 2e-3
hps['d_lrn_rate'] = 2e-3
hps['step_num'] = 5000  # 5000
hps['batch_size'] = 1
hps['dis_extra_step'] = 0
hps['data_path'] = 'E:\\BaiduNetdiskDownload\\adb\\BossBase-1.01-cover\\train.tfrecords'
# hps['mode'] = False  # true for training
hps['mode'] = True
hps['outer_iter'] = 10

data_path = 'E:\\BaiduNetdiskDownload\\adb\\BossBase-1.01-cover\\train.tfrecords'
"""
read tfrecord file
"""
raw_image = read_and_decode(data_path)
# img_batch = tf.train.shuffle_batch([raw_image], batch_size=hps['batch_size'], capacity=10000, min_after_dequeue=10)
img_batch = tf.train.shuffle_batch([raw_image], batch_size=hps['batch_size'], capacity=100, min_after_dequeue=10)
# img_batch = tf.train.batch([raw_image],batc yt 6v57h_size=hps['batch_size'], capacity=10000)
coord = tf.train.Coordinator()
img_batch = tf.cast(img_batch, dtype=tf.float32)
"""
Create discriminator/generator object
Generator's model is built in its constructor
"""
Dis = TluDiscriminator(hps)
Gen = WrfGenerator(hps)

"""
Start to build discriminator model
"""
real_logits, real_pred = Dis._build_model(Gen.images, Gen.pro)
real_label = tf.placeholder(tf.float32, shape=[hps['batch_size'], 1])

fake_logits, fake_pred = Dis._build_model(Gen.stego, Gen.pro, True)
fake_label = tf.placeholder(tf.float32, shape=[hps['batch_size'], 1])

"""
Build loss terms
"""
dis_loss_real = tf.reduce_mean(cross_entropy(real_logits, real_label))
dis_loss_fake = tf.reduce_mean(cross_entropy(fake_logits, fake_label))
dis_loss = dis_loss_real + dis_loss_fake

gen_loss_fake = tf.reduce_mean(cross_entropy(fake_logits, real_label))  # real/fake loss
gen_loss_p2 = tf.reduce_mean(
    tf.square(Gen.capacity - tf.constant(hps['image_size'] * hps['image_size'] * hps['capacity'])))  # capacity loss

gen_loss_p1 = hps['g_alph'] * gen_loss_fake + gen_loss_p2  # +1e6*gen_loss_p3

t_vars = tf.trainable_variables()

d_vars = [var for var in t_vars if 'd_' in var.name]
g_vars = [var for var in t_vars if 'g_' in var.name]

"""
Build optimizers
"""
dis_optimizer = tf.train.AdamOptimizer(hps['d_lrn_rate'], beta1=0.5)
min_dis_loss = dis_optimizer.minimize(dis_loss, var_list=d_vars)
dis_train_op = [min_dis_loss]

gen_optimizer_1 = tf.train.AdamOptimizer(hps['g_lrn_rate'], beta1=0.5)
min_gen_loss_1 = gen_optimizer_1.minimize(gen_loss_p1, var_list=g_vars, global_step=Gen.step)

gen_train_op = [min_gen_loss_1]
saver = tf.train.Saver()
init = tf.global_variables_initializer()
try:
    os.mkdir(os.getcwd() + '\\result')
except:
    useless = 0

"""
Control the usage of memory
"""
sess_config = tf.ConfigProto()
# sess_config.gpu_options.per_process_gpu_memory_fraction=0.45
sess_config.allow_soft_placement = True
sess_config.gpu_options.allow_growth = True
"""
Ready to run tensors!
"""

with tf.Session(config=sess_config) as sess:
    if (hps['mode'] == True):
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    sess.run([init, Dis._init, Gen._init])
    try:
        saver.restore(sess, os.getcwd() + '\\gan_ckpt\\gan.ckpt')
        print("read success")
    except:
        useless = 0
    # batch_size
    real_truth = get_label(True)
    fake_truth = get_label(False)
    if hps['mode']:
        """
        Training process
        """
        count = 216

        for i in range(hps['outer_iter']):
            for j in range(hps['step_num']):
                step = sess.run(Gen.step)
                data = sess.run(img_batch)
                rand = np.random.uniform(0.0, 1.0, Gen.rand_shape)
                for i1 in range(hps['dis_extra_step']):
                    sess.run(dis_train_op, feed_dict={Gen.images: data, Gen.rand: rand, real_label: real_truth,
                                                      fake_label: fake_truth})
                # run discriminator train_op
                fake, d_loss, _ = sess.run([fake_pred, dis_loss, dis_train_op],
                                           feed_dict={Gen.images: data, Gen.rand: rand, real_label: real_truth,
                                                      fake_label: fake_truth})

                if step != 0 and step % 1000 == 0:
                    # save ckpt
                    bpp, stegos, cap, l1, l2, _ = sess.run(
                        [Gen.bpp, Gen.stego, Gen.capacity, gen_loss_fake, gen_loss_p2, gen_train_op],
                        feed_dict={Gen.images: data, Gen.rand: rand, real_label: real_truth})
                    cap = 1.0 * cap / hps['image_size'] / hps['image_size']
                    cap = np.mean(cap)
                    for j1 in range(1):
                        bppim = np.reshape(bpp[j1], [hps['image_size'], hps['image_size']])
                        ste = np.reshape(stegos[j1], [hps['image_size'], hps['image_size']])
                        saveImg(bppim, 'E:\\why_homework\\DCGAN-for-Steganography-master\\bpp\\%d' % count)
                        saveImg(ste, 'E:\\why_homework\\DCGAN-for-Steganography-master\\stego\\%d' % count)
                        count = count + 1
                    print('%d step:' % step, cap)
                    save_path = saver.save(sess, os.getcwd() + '\\gan_ckpt\\gan.ckpt')
                    saver.save(sess, os.getcwd() + '\\gan_ckpt_all\\gan.ckpt%d' % step)
                else:
                    # run generator train_op
                    cap, l1, l2, _ = sess.run([Gen.capacity, gen_loss_fake, gen_loss_p2, gen_train_op],
                                              feed_dict={Gen.images: data, Gen.rand: rand, real_label: real_truth})
                    cap = 1.0 * cap / hps['image_size'] / hps['image_size']
                    cap = np.mean(cap)
                    if step != 0 and step % 300 == 0:
                        writer = tf.summary.FileWriter("log/", sess.graph)
                    # for j1 in range(pros):
                    # a = np.reshape(pros[j1], [hps['image_size'], hps['image_size']])
                    print('%d step: ' % step, d_loss, ' ', l1, l2, ' ', cap)

    else:
        """
        Testing process
        """
        test_data = get_test_data('E:\\BaiduNetdiskDownload\\adb\\BossBase-1.01-cover', batch_size=hps['batch_size'],
                                  totol_size=2)
        batch_size = hps['batch_size']
        test_data = np.reshape(test_data, [int(test_data.shape[0]), 512, 512, 1])
        count = 0
        final_cap = 0.0
        rand = np.random.uniform(0.0, 1.0, Gen.rand_shape)
        images, pros, cap, stegos, bpp = sess.run([Gen.images, Gen.pro, Gen.capacity, Gen.stego, Gen.bpp],
                                                  feed_dict={Gen.images: test_data, Gen.rand: rand})  # data[3*i:3*i+3]
        a = np.reshape(pros[0], [hps['image_size'], hps['image_size']])
        b = np.reshape(images[0], [hps['image_size'], hps['image_size']])
        c = np.reshape(stegos[0], [hps['image_size'], hps['image_size']])
        bppim = np.reshape(bpp[0], [hps['image_size'], hps['image_size']])
        saveImg(b, 'E:\\why_homework\\DCGAN-for-Steganography-master\\cover')
        saveImg(c, 'E:\\why_homework\\DCGAN-for-Steganography-master\\stego')
        saveImg(bppim, 'E:\\why_homework\\DCGAN-for-Steganography-master\\bpp')
        cap = 1.0 * cap / hps['image_size'] / hps['image_size']
        cap = np.mean(cap)
        print(cap)
    # for i in range(200):
    # 	rand=np.random.uniform(0.0, 1.0, Gen.rand_shape)
    #
    # #512*512 #print("rand_shape ",Gen.rand_shape) images, pros, cap, stegos, bpp= sess.run([Gen.images, Gen.pro,
    # Gen.capacity, Gen.stego, Gen.bpp], feed_dict={Gen.images: test_data[batch_size * i:batch_size * (i + 1)],
    # Gen.rand:rand})#data[3*i:3*i+3] #print(type(stego),type(images),type(pros),type(cap)) #print(cap) #batch_size
    # cap=1.0*cap/hps['image_size']/hps['image_size'] cap=np.mean(cap) final_cap+=cap for j in range(hps[
    # 'batch_size']): a = np.reshape(pros[j], [hps['image_size'],hps['image_size']]) b = np.reshape(images[j],
    # [hps['image_size'],hps['image_size']]) c = np.reshape(stegos[j], [hps['image_size'], hps['image_size']]) bppim =
    # np.reshape(bpp[j],[hps['image_size'], hps['image_size']]) # for i1 in range(0,512): # 	for j1 in range(0,
    # 512): # 		if(bppim[i1][j1]>0): # 			bppim[i1][j1]=1 # 		else: # 			bppim[i1][j1]=0
    #
    # 		np.save('F:\\python\\DCGAN-for-Steganography-master\\test\\pros'+'\\'+'%d.npy'%count, a)
    # 		print(a)
    # 		#saveImg(a, 'F:\\python\\DCGAN-for-Steganography-master\\pros\\%d' % count)
    # 		saveImg(bppim,'F:\\python\\DCGAN-for-Steganography-master\\test\\bpp' + '\\' + '%d' % count)
    # 		#saveImg(a,'F:\\python\\DCGAN-for-Steganography-master\\pros\\%d'%count) #原本是这里被注释
    # 		saveImg(b,'F:\\python\\DCGAN-for-Steganography-master\\test\\covers'+'\\%d'%count)
    # 		saveImg(c, 'F:\\python\\DCGAN-for-Steganography-master\\test\\stego' + '\\%d' % count)
    # 		count+=1
    # 	print('process %d cap:%r'%(i,cap))
    # print('final_cap:%r'%(final_cap/200))
    if (hps['mode'] == 'train'):
        coord.request_stop()
        coord.join(threads)
