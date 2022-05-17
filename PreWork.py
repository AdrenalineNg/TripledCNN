import os
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from numpy import *

# image file path
agricultural_train = []
airplane_train = []
baseballdiamond_train = []
beach_train = []
buildings_train = []
chaparral_train = []
denseresidential_train = []
forest_train = []
freeway_train = []
golfcourse_train = []
harbor_train = []
intersection_train = []
mediumresidential_train = []
mobilehomepark_train = []
overpass_train = []
parkinglot_train = []
river_train = []
runway_train = []
sparseresidential_train = []
storagetanks_train = []
tenniscourt_train = []

# image label
label_agricultural_train = []
label_airplane_train = []
label_baseballdiamond_train = []
label_beach_train = []
label_buildings_train = []
label_chaparral_train = []
label_denseresidential_train = []
label_forest_train = []
label_freeway_train = []
label_golfcourse_train = []
label_harbor_train = []
label_intersection_train = []
label_mediumresidential_train = []
label_mobilehomepark_train = []
label_overpass_train = []
label_parkinglot_train = []
label_river_train = []
label_runway_train = []
label_sparseresidential_train = []
label_storagetanks_train = []
label_tenniscourt_train = []


def get_train_file(file_dir):
	for file in os.listdir(file_dir + '/agricultural'):
		if file != '.DS_Store':
			agricultural_train.append(file_dir + '/agricultural' + '/' + file)
			label_agricultural_train.append(0)
	for file in os.listdir(file_dir + '/airplane'):
		if file != '.DS_Store':
			airplane_train.append(file_dir + '/airplane' + '/' + file)
			label_airplane_train.append(1)
	for file in os.listdir(file_dir + '/baseballdiamond'):
		if file != '.DS_Store':
			baseballdiamond_train.append(file_dir + '/baseballdiamond' + '/' + file)
			label_baseballdiamond_train.append(2)
	for file in os.listdir(file_dir + '/beach'):
		if file != '.DS_Store':
			beach_train.append(file_dir + '/beach' + '/' + file)
			label_beach_train.append(3)
	for file in os.listdir(file_dir + '/buildings'):
		if file != '.DS_Store':
			buildings_train.append(file_dir + '/buildings' + '/' + file)
			label_buildings_train.append(4)
	for file in os.listdir(file_dir + '/chaparral'):
		if file != '.DS_Store':
			chaparral_train.append(file_dir + '/chaparral' + '/' + file)
			label_chaparral_train.append(5)
	for file in os.listdir(file_dir + '/denseresidential'):
		if file != '.DS_Store':
			denseresidential_train.append(file_dir + '/denseresidential' + '/' + file)
			label_denseresidential_train.append(6)
	for file in os.listdir(file_dir + '/forest'):
		if file != '.DS_Store':
			forest_train.append(file_dir + '/forest' + '/' + file)
			label_forest_train.append(7)
	for file in os.listdir(file_dir + '/freeway'):
		if file != '.DS_Store':
			freeway_train.append(file_dir + '/freeway' + '/' + file)
			label_freeway_train.append(8)
	for file in os.listdir(file_dir + '/golfcourse'):
		if file != '.DS_Store':
			golfcourse_train.append(file_dir + '/golfcourse' + '/' + file)
			label_golfcourse_train.append(9)
	for file in os.listdir(file_dir + '/harbor'):
		if file != '.DS_Store':
			harbor_train.append(file_dir + '/harbor' + '/' + file)
			label_harbor_train.append(10)
	for file in os.listdir(file_dir + '/intersection'):
		if file != '.DS_Store':
			intersection_train.append(file_dir + '/intersection' + '/' + file)
			label_intersection_train.append(11)
	for file in os.listdir(file_dir + '/mediumresidential'):
		if file != '.DS_Store':
			mediumresidential_train.append(file_dir + '/mediumresidential' + '/' + file)
			label_mediumresidential_train.append(12)
	for file in os.listdir(file_dir + '/mobilehomepark'):
		if file != '.DS_Store':
			mobilehomepark_train.append(file_dir + '/mobilehomepark' + '/' + file)
			label_mobilehomepark_train.append(13)
	for file in os.listdir(file_dir + '/overpass'):
		if file != '.DS_Store':
			overpass_train.append(file_dir + '/overpass' + '/' + file)
			label_overpass_train.append(14)
	for file in os.listdir(file_dir + '/parkinglot'):
		if file != '.DS_Store':
			parkinglot_train.append(file_dir + '/parkinglot' + '/' + file)
			label_parkinglot_train.append(15)
	for file in os.listdir(file_dir + '/river'):
		if file != '.DS_Store':
			river_train.append(file_dir + '/river' + '/' + file)
			label_river_train.append(16)
	for file in os.listdir(file_dir + '/runway'):
		if file != '.DS_Store':
			runway_train.append(file_dir + '/runway' + '/' + file)
			label_runway_train.append(17)
	for file in os.listdir(file_dir + '/sparseresidential'):
		if file != '.DS_Store':
			sparseresidential_train.append(file_dir + '/sparseresidential' + '/' + file)
			label_sparseresidential_train.append(18)
	for file in os.listdir(file_dir + '/storagetanks'):
		if file != '.DS_Store':
			storagetanks_train.append(file_dir + '/storagetanks' + '/' + file)
			label_storagetanks_train.append(19)
	for file in os.listdir(file_dir + '/tenniscourt'):
		if file != '.DS_Store':
			tenniscourt_train.append(file_dir + '/tenniscourt' + '/' + file)
			label_tenniscourt_train.append(20)
	print("There are %d pictures in each class" %(len(agricultural_train)))
	
	image_list = np.hstack((agricultural_train, airplane_train, baseballdiamond_train, beach_train, buildings_train, chaparral_train, denseresidential_train, forest_train, freeway_train, golfcourse_train, harbor_train, intersection_train, mediumresidential_train, mobilehomepark_train, overpass_train, parkinglot_train, river_train, runway_train, sparseresidential_train, storagetanks_train, tenniscourt_train))
	label_list = np.hstack((label_agricultural_train, label_airplane_train, label_baseballdiamond_train, label_beach_train, label_buildings_train, label_chaparral_train, label_denseresidential_train, label_forest_train, label_freeway_train, label_golfcourse_train, label_harbor_train, label_intersection_train, label_mediumresidential_train, label_mobilehomepark_train, label_overpass_train, label_parkinglot_train, label_river_train, label_runway_train, label_sparseresidential_train, label_storagetanks_train, label_tenniscourt_train))
	temp = np.array([image_list, label_list])
	temp = temp.transpose()
	np.random.shuffle(temp)

	all_image_list = list(temp[:, 0])    # image path
	all_label_list = list(temp[:, 1])    # image laebl
	label_list = [int(i) for i in label_list]	
	return image_list, label_list
	
agricultural_test = []
airplane_test = []
baseballdiamond_test = []
beach_test = []
buildings_test = []
chaparral_test = []
denseresidential_test = []
forest_test = []
freeway_test = []
golfcourse_test = []
harbor_test = []
intersection_test = []
mediumresidential_test = []
mobilehomepark_test = []
overpass_test = []
parkinglot_test = []
river_test = []
runway_test = []
sparseresidential_test = []
storagetanks_test = []
tenniscourt_test = []

# image label
label_agricultural_test = []
label_airplane_test = []
label_baseballdiamond_test = []
label_beach_test = []
label_buildings_test = []
label_chaparral_test = []
label_denseresidential_test = []
label_forest_test = []
label_freeway_test = []
label_golfcourse_test = []
label_harbor_test = []
label_intersection_test = []
label_mediumresidential_test = []
label_mobilehomepark_test = []
label_overpass_test = []
label_parkinglot_test = []
label_river_test = []
label_runway_test = []
label_sparseresidential_test = []
label_storagetanks_test = []
label_tenniscourt_test = []


def get_test_file(file_dir):
	for file in os.listdir(file_dir + '/agricultural'):
		if file != '.DS_Store':
			agricultural_test.append(file_dir + '/agricultural' + '/' + file)
			label_agricultural_test.append(0)
	for file in os.listdir(file_dir + '/airplane'):
		if file != '.DS_Store':
			airplane_test.append(file_dir + '/airplane' + '/' + file)
			label_airplane_test.append(1)
	for file in os.listdir(file_dir + '/baseballdiamond'):
		if file != '.DS_Store':
			baseballdiamond_test.append(file_dir + '/baseballdiamond' + '/' + file)
			label_baseballdiamond_test.append(2)
	for file in os.listdir(file_dir + '/beach'):
		if file != '.DS_Store':
			beach_test.append(file_dir + '/beach' + '/' + file)
			label_beach_test.append(3)
	for file in os.listdir(file_dir + '/buildings'):
		if file != '.DS_Store':
			buildings_test.append(file_dir + '/buildings' + '/' + file)
			label_buildings_test.append(4)
	for file in os.listdir(file_dir + '/chaparral'):
		if file != '.DS_Store':
			chaparral_test.append(file_dir + '/chaparral' + '/' + file)
			label_chaparral_test.append(5)
	for file in os.listdir(file_dir + '/denseresidential'):
		if file != '.DS_Store':
			denseresidential_test.append(file_dir + '/denseresidential' + '/' + file)
			label_denseresidential_test.append(6)
	for file in os.listdir(file_dir + '/forest'):
		if file != '.DS_Store':
			forest_test.append(file_dir + '/forest' + '/' + file)
			label_forest_test.append(7)
	for file in os.listdir(file_dir + '/freeway'):
		if file != '.DS_Store':
			freeway_test.append(file_dir + '/freeway' + '/' + file)
			label_freeway_test.append(8)
	for file in os.listdir(file_dir + '/golfcourse'):
		if file != '.DS_Store':
			golfcourse_test.append(file_dir + '/golfcourse' + '/' + file)
			label_golfcourse_test.append(9)
	for file in os.listdir(file_dir + '/harbor'):
		if file != '.DS_Store':
			harbor_test.append(file_dir + '/harbor' + '/' + file)
			label_harbor_test.append(10)
	for file in os.listdir(file_dir + '/intersection'):
		if file != '.DS_Store':
			intersection_test.append(file_dir + '/intersection' + '/' + file)
			label_intersection_test.append(11)
	for file in os.listdir(file_dir + '/mediumresidential'):
		if file != '.DS_Store':
			mediumresidential_test.append(file_dir + '/mediumresidential' + '/' + file)
			label_mediumresidential_test.append(12)
	for file in os.listdir(file_dir + '/mobilehomepark'):
		if file != '.DS_Store':
			mobilehomepark_test.append(file_dir + '/mobilehomepark' + '/' + file)
			label_mobilehomepark_test.append(13)
	for file in os.listdir(file_dir + '/overpass'):
		if file != '.DS_Store':
			overpass_test.append(file_dir + '/overpass' + '/' + file)
			label_overpass_test.append(14)
	for file in os.listdir(file_dir + '/parkinglot'):
		if file != '.DS_Store':
			parkinglot_test.append(file_dir + '/parkinglot' + '/' + file)
			label_parkinglot_test.append(15)
	for file in os.listdir(file_dir + '/river'):
		if file != '.DS_Store':
			river_test.append(file_dir + '/river' + '/' + file)
			label_river_test.append(16)
	for file in os.listdir(file_dir + '/runway'):
		if file != '.DS_Store':
			runway_test.append(file_dir + '/runway' + '/' + file)
			label_runway_test.append(17)
	for file in os.listdir(file_dir + '/sparseresidential'):
		if file != '.DS_Store':
			sparseresidential_test.append(file_dir + '/sparseresidential' + '/' + file)
			label_sparseresidential_test.append(18)
	for file in os.listdir(file_dir + '/storagetanks'):
		if file != '.DS_Store':
			storagetanks_test.append(file_dir + '/storagetanks' + '/' + file)
			label_storagetanks_test.append(19)
	for file in os.listdir(file_dir + '/tenniscourt'):
		if file != '.DS_Store':
			tenniscourt_test.append(file_dir + '/tenniscourt' + '/' + file)
			label_tenniscourt_test.append(20)
	print("There are %d pictures in each class" %(len(agricultural_test)))
	
	image_list = np.hstack((agricultural_test, airplane_test, baseballdiamond_test, beach_test, buildings_test, chaparral_test, denseresidential_test, forest_test, freeway_test, golfcourse_test, harbor_test, intersection_test, mediumresidential_test, mobilehomepark_test, overpass_test, parkinglot_test, river_test, runway_test, sparseresidential_test, storagetanks_test, tenniscourt_test))
	label_list = np.hstack((label_agricultural_test, label_airplane_test, label_baseballdiamond_test, label_beach_test, label_buildings_test, label_chaparral_test, label_denseresidential_test, label_forest_test, label_freeway_test, label_golfcourse_test, label_harbor_test, label_intersection_test, label_mediumresidential_test, label_mobilehomepark_test, label_overpass_test, label_parkinglot_test, label_river_test, label_runway_test, label_sparseresidential_test, label_storagetanks_test, label_tenniscourt_test))
	temp = np.array([image_list, label_list])
	temp = temp.transpose()
	np.random.shuffle(temp)

	all_image_list = list(temp[:, 0])    # image path
	all_label_list = list(temp[:, 1])    # image laebl
	label_list = [int(i) for i in label_list]	
	return image_list, label_list
	
def get_batch(image, label, image_W, image_H, batch_size, capacity):
	image = tf.cast(image, tf.string)  
	label = tf.cast(label, tf.int32)
	input_queue = tf.train.slice_input_producer([image, label])
	label = input_queue[1]
	image_contents = tf.read_file(input_queue[0])  
 
	print(image_contents)
	image = tf.image.decode_jpeg(image_contents, channels=3)
	image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
	image = tf.image.per_image_standardization(image)
	
	# image_batch: 4D tensor [batch_size, width, height, 3], dtype = tf.float32
	# label_batch: 1D tensor [batch_size], dtype = tf.int32
	image.set_shape([256,256,3])
	image_batch, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=16, capacity=capacity)

	label_batch = tf.reshape(label_batch, [batch_size])
	image_batch = tf.cast(image_batch, tf.float32)
	print(image_batch)
	print(label_batch)
	return image_batch, label_batch

def PreWork():
	IMG_W = 256
	IMG_H = 256
	BATCH_SIZE = 21
	CAPACITY = 64
	train_dir = '/Users/ku/Desktop/SGCN/benchmark/dataset'
	# image_list, label_list, val_images, val_labels = get_file(train_dir)
	image_list, label_list = get_file(train_dir)
	image_batch, label_batch = get_batch(image_list, label_list, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)
	print(label_batch.shape)
	lists = ('agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt')
	with tf.Session() as sess:
		i = 0
		coord = tf.train.Coordinator()  # 创建一个线程协调器，用来管理之后在Session中启动的所有线程
		threads = tf.train.start_queue_runners(coord=coord)
		try:
			while not coord.should_stop() and i < 1:
				# 提取出两个batch的图片并可视化。
				img, label = sess.run([image_batch, label_batch])  # 在会话中取出img和label
				# img = tf.cast(img, tf.uint8)
				for j in np.arange(BATCH_SIZE):
					# np.arange()函数返回一个有终点和起点的固定步长的排列
					print('label: %d' % label[j])
					plt.imshow(img[j, :, :, :])
					title = lists[int(label[j])]
					plt.title(title)
					plt.show()
				i += 1
		except tf.errors.OutOfRangeError:
			print('done!')
		finally:
			coord.request_stop()
		coord.join(threads)
		
if __name__ == '__main__':
	PreWork()
