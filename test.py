
import os
import tensorflow as tf
from time import time
import numpy as np
from LSTM.setting import batch_size, width, height, rnn_size, out_size, channel, learning_rate, num_epoch
 

 
def weight_variable(shape, w_alpha=0.01):
		initial = w_alpha * tf.random_normal(shape)
		return tf.Variable(initial)
def bias_variable(shape, b_alpha=0.1):
		initial = b_alpha * tf.random_normal(shape)
		return tf.Variable(initial)
def rnn_graph(x, rnn_size, out_size, width, height, channel):
		w = weight_variable([rnn_size, out_size])
		b = bias_variable([out_size])
		lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
		x = tf.transpose(x, [1,0,2,3])
		x = tf.reshape(x, [-1, channel*width])
		x = tf.split(x, height)
		outputs, status = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)
		y_conv = tf.add(tf.matmul(outputs[-1], w), b)
		return y_conv
 
def accuracy_graph(y, y_conv):
		correct = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
		return accuracy
 
def get_batch(image_list,label_list,img_width,img_height,batch_size,capacity,channel):
		image = tf.cast(image_list,tf.string)
		label = tf.cast(label_list,tf.int32)
		input_queue = tf.train.slice_input_producer([image,label],shuffle=True)
		label = input_queue[1]
		image_contents = tf.read_file(input_queue[0])
 
		image = tf.image.decode_jpeg(image_contents,channels=channel)
		image = tf.cast(image,tf.float32)
		if channel==3:
				image -= [42.79902,42.79902,42.79902] # 减均值
		elif channel == 1:
				image -= 42.79902  # 减均值
		image.set_shape((img_height,img_width,channel))
		image_batch,label_batch = tf.train.batch([image,label],batch_size=batch_size,num_threads=64,capacity=capacity)
		label_batch = tf.reshape(label_batch,[batch_size])
 
		return image_batch,label_batch
 
def get_file(file_dir):
		images = []
		for root,sub_folders,files in os.walk(file_dir):
				for name in files:
						images.append(os.path.join(root,name))
		labels = []
		for label_name in images:
				letter = label_name.split("\\")[-2]
				if letter =="lh1":labels.append(0)
				elif letter =="lh2":labels.append(1)
				elif letter == "lh3":labels.append(2)
				elif letter == "lh4":labels.append(3)
				elif letter == "lh5":labels.append(4)
				elif letter == "lh6":labels.append(5)
				elif letter == "lh7":
						labels.append(6)
 
		print("check for get_file:",images[0],"label is ",labels[0])
		#shuffle
		temp = np.array([images,labels])
		temp = temp.transpose()
		np.random.shuffle(temp)
		image_list = list(temp[:,0])
		label_list = list(temp[:,1])
		label_list = [int(float(i)) for i in label_list]
		return image_list,label_list

def onehot(labels):
		n_sample = len(labels)
		n_class = 7  # max(labels) + 1
		onehot_labels = np.zeros((n_sample,n_class))
		onehot_labels[np.arange(n_sample),labels] = 1
		return onehot_labels
 
if __name__ == '__main__':
		startTime = time()
		x = tf.placeholder(tf.float32, [None, height, width, channel])
		y = tf.placeholder(tf.float32)
		y_conv = rnn_graph(x, rnn_size, out_size, width, height, channel)
		y_conv_prediction = tf.argmax(y_conv, 1)
		y_real = tf.argmax(y, 1)
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
		accuracy = accuracy_graph(y, y_conv)
		xs, ys = get_file('./data/train1')
		image_batch, label_batch = get_batch(xs, ys, img_width=width, img_height=height, batch_size=batch_size, capacity=256,channel=channel)
		xs_val, ys_val = get_file('./data/test1')
		image_val_batch, label_val_batch = get_batch(xs_val, ys_val, img_width=width, img_height=height,batch_size=455, capacity=256,channel=channel)
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()

		coord = tf.train.Coordinator() 
		threads = tf.train.start_queue_runners(coord=coord, sess=sess)
		summary_writer = tf.summary.FileWriter('./logs/', graph=sess.graph, flush_secs=15)
		summary_writer2 = tf.summary.FileWriter('./logs/plot2/', flush_secs=15)
		tf.summary.scalar(name='loss_func', tensor=loss)
		tf.summary.scalar(name='accuracy', tensor=accuracy)
		merged_summary_op = tf.summary.merge_all()
 
		step = 0
		acc_rate = 0.98
		epoch_start_time = time()
		for i in range(num_epoch):
				batch_x, batch_y = sess.run([image_batch, label_batch])
				batch_y = onehot(batch_y)
 
				merged_summary,_,loss_show = sess.run([merged_summary_op,optimizer,loss], feed_dict={x: batch_x, y: batch_y})
				summary_writer.add_summary(merged_summary, global_step=i)
 
				if i % (int(7000//batch_size)) == 0:
						batch_x_test, batch_y_test = sess.run([image_val_batch, label_val_batch])
						batch_y_test = onehot(batch_y_test)
						batch_x_test = batch_x_test.reshape([-1, height, width, channel])
						merged_summary_val,acc,prediction_val_out,real_val_out,loss_show = sess.run([merged_summary_op,accuracy,y_conv_prediction,y_real,loss],feed_dict={x: batch_x_test, y: batch_y_test})
						summary_writer2.add_summary(merged_summary_val, global_step=i)
 
						lh1_right, lh2_right, lh3_right, lh4_right, lh5_right, lh6_right, lh7_right = 0, 0, 0, 0, 0, 0, 0
						lh1_wrong, lh2_wrong, lh3_wrong, lh4_wrong, lh5_wrong, lh6_wrong, lh7_wrong = 0, 0, 0, 0, 0, 0, 0
						for ii in range(len(prediction_val_out)):
								if prediction_val_out[ii] == real_val_out[ii]:
										if real_val_out[ii] == 0:
												lh1_right += 1
										elif real_val_out[ii] == 1:
												lh2_right += 1
										elif real_val_out[ii] == 2:
												lh3_right += 1
										elif real_val_out[ii] == 3:
												lh4_right += 1
										elif real_val_out[ii] == 4:
												lh5_right += 1
										elif real_val_out[ii] == 5:
												lh6_right += 1
										elif real_val_out[ii] == 6:
												lh7_right += 1
								else:
										if real_val_out[ii] == 0:
												lh1_wrong += 1
										elif real_val_out[ii] == 1:
												lh2_wrong += 1
										elif real_val_out[ii] == 2:
												lh3_wrong += 1
										elif real_val_out[ii] == 3:
												lh4_wrong += 1
										elif real_val_out[ii] == 4:
												lh5_wrong += 1
										elif real_val_out[ii] == 5:
												lh6_wrong += 1
										elif real_val_out[ii] == 6:
												lh7_wrong += 1
						print(step, "correct rate :", ((lh1_right) / (lh1_right + lh1_wrong)), ((lh2_right) / (lh2_right + lh2_wrong)),
									((lh3_right) / (lh3_right + lh3_wrong)), ((lh4_right) / (lh4_right + lh4_wrong)),
									((lh5_right) / (lh5_right + lh5_wrong)), ((lh6_right) / (lh6_right + lh6_wrong)),
									((lh7_right) / (lh7_right + lh7_wrong)))
						print(step, "accurcy",(((lh1_right) / (lh1_right + lh1_wrong))+((lh2_right) / (lh2_right + lh2_wrong))+
									((lh3_right) / (lh3_right + lh3_wrong))+((lh4_right) / (lh4_right + lh4_wrong))+
									((lh5_right) / (lh5_right + lh5_wrong))+((lh6_right) / (lh6_right + lh6_wrong))+
									((lh7_right) / (lh7_right + lh7_wrong)))/7)
 
 
						epoch_end_time = time()
						print("takes time:",(epoch_end_time-epoch_start_time), ' step:', step, ' accuracy:', acc," loss_fun:",loss_show)
						epoch_start_time = epoch_end_time
						if acc >= acc_rate:
								model_path = os.getcwd() + os.sep + '\models\\'+str(acc_rate) + "LSTM.model"
								saver.save(sess, model_path, global_step=step)
								break
						if step % 10 == 0 and step != 0:
								model_path = os.getcwd() + os.sep + '\models\\'  + str(acc_rate)+ "LSTM"+str(step)+".model"
								print(model_path)
								saver.save(sess, model_path, global_step=step)
						step += 1
 
		duration = time() - startTime
		print("total takes time:",duration)
		summary_writer.close()
 
		coord.request_stop() 
		coord.join(threads) 