import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from time import time
from PreWork import get_train_file, get_test_file, get_batch
from CNNModel import CNN, losses, trainning, evaluation


N_CLASSES = 21
WIDTH = 256
HEIGHT = 256
BATCH_SIZE = 21
CAPACITY = 200 
MAX_STEP = 30000 
learning_rate = 0.0001
startTime = time()
loss = []
acc = []

train_dir = '/Users/ku/Desktop/workdic/Python/SGCN/benchmark/dataset'
logs_train_dir = '/Users/ku/Desktop/workdic/Python/SGCN/benchmark/CNN'
train, train_label = get_train_file(train_dir)
train_batch, train_label_batch = get_batch(train, train_label, WIDTH, HEIGHT, BATCH_SIZE, CAPACITY)


train_logits = CNN(train_batch, BATCH_SIZE, N_CLASSES)
train_loss = losses(train_logits, train_label_batch)
train_op = trainning(train_loss, learning_rate)
train_acc = evaluation(train_logits, train_label_batch)


summary_op = tf.summary.merge_all()
sess = tf.Session()
train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


epoch_start_time = time()
try:
	# 执行MAX_STEP步的训练，一步一个batch
	for step in np.arange(MAX_STEP):
		if coord.should_stop():
			break
		# 启动以下操作节点，有个疑问，为什么train_logits在这里没有开启？
		_, tmp_loss, tmp_acc = sess.run([train_op, train_loss, train_acc])
 
		# 每隔50步打印一次当前的loss以及acc，同时记录log，写入writer
		if step % 100 == 0:
			print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tmp_loss, tmp_acc * 100.0))
			loss.append(tmp_loss)
			loss_CNN = np.array(loss)
			acc.append(tmp_acc)
			acc_CNN = np.array(acc)
			print(acc_CNN)
			print(loss_CNN)
			np.save('/Users/ku/Desktop/SGCN/benchmark/acc_CNN.npy', acc_CNN)
			np.save('/Users/ku/Desktop/SGCN/benchmark/loss_CNN.npy',loss_CNN)
			summary_str = sess.run(summary_op)
			train_writer.add_summary(summary_str, step)
			
		if step % 1000 == 0:
			print('start evaluation!!!!!')
			tmp_acc = sess.run(train_acc)
			epoch_end_time = time()
			print('takes time:',(epoch_end_time-epoch_start_time),'test accuracy:', tmp_acc)
			epoch_start_time = epoch_end_time
			checkpoint_path = os.path.join(logs_train_dir, 'CNN.ckpt')
			saver.save(sess, checkpoint_path)
		
except tf.errors.OutOfRangeError:
	print('Done training -- epoch limit reached')
 
finally:
	coord.request_stop()
coord.join(threads)
sess.close()
np.save('/Users/ku/Desktop/SGCN/benchmark/acc_CNN.npy', acc_CNN)
np.save('/Users/ku/Desktop/SGCN/benchmark/loss_CNN.npy',loss_CNN)