import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from time import time
from PreWork import get_train_file, get_test_file, get_batch
from LSTMModel import weight_variable, bias_variable, RNN, losses, trainning, evaluation


N_CLASSES = 21
BATCH_SIZE = 21
WIDTH = 256
HEIGHT = 256
CAPACITY = 256
MAX_STEP = 30000
learning_rate = 0.001
startTime = time()
loss = []
acc = []

train_dir = '/Users/ku/Desktop/SGCN/benchmark/dataset'
logs_train_dir = '/Users/ku/Desktop/SGCN/benchmark/RNN'
train, train_label = get_train_file(train_dir)
train_batch, train_label_batch = get_batch(train, train_label, WIDTH, HEIGHT, BATCH_SIZE, CAPACITY)


train_logits = RNN(train_batch, N_CLASSES, WIDTH, HEIGHT)
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
  for step in np.arange(MAX_STEP):
    
    if coord.should_stop():
      break
    _, tmp_loss, tmp_acc = sess.run([train_op, train_loss, train_acc])
    
    if step % 100 == 0:
      print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (step, tmp_loss, tmp_acc * 100.0))
      loss.append(tmp_loss)
      loss_RNN = np.array(loss)
      acc.append(tmp_acc)
      acc_RNN = np.array(acc)
      print(acc_RNN)
      print(loss_RNN)
      np.save('/Users/ku/Desktop/SGCN/benchmark/acc_RNN.npy',acc_RNN)
      np.save('/Users/ku/Desktop/SGCN/benchmark/loss_RNN.npy',loss_RNN)
      summary_str = sess.run(summary_op)
      train_writer.add_summary(summary_str, step)
    
    if step % 1000 == 0:
      print('start evaluation!!!!!')
      tmp_acc = sess.run(train_acc)
      epoch_end_time = time()
      print('takes time:',(epoch_end_time-epoch_start_time),'test accuracy:', tmp_acc)
      epoch_start_time = epoch_end_time
      checkpoint_path = os.path.join(logs_train_dir, 'RNN.ckpt')
      saver.save(sess, checkpoint_path)
    
except tf.errors.OutOfRangeError:
  print('Done training -- epoch limit reached')
  
finally:
  coord.request_stop()
coord.join(threads)
sess.close()
np.save('/Users/ku/Desktop/SGCN/benchmark/acc_RNN.npy',acc_RNN)
np.save('/Users/ku/Desktop/SGCN/benchmark/loss_RNN.npy',loss_RNN)