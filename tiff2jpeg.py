from PIL import Image
import os
import cv2
	
save_path = '/Users/ku/Desktop/SGCN/benchmark/dataset/'
load_path = '/Users/ku/Desktop/SGCN/benchmark/Images/'
lists = ('agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt')

for i in range(21):
	os.makedirs(save_path+lists[i])
		
for i in range(21):
	k = 1
	for file in os.listdir(load_path + lists[i]):
		#temp = tif.imread(load_path + lists[i]+'/'+ file)
		#tif.imsave(save_path+lists[i]+'/'+lists[i]+str(k)+'.jpeg', temp)
		name = lists[i]
		tmp = cv2.imread(load_path + lists[i]+'/'+ file)
		file_name_temp = name[:-4]
		cv2.imwrite(save_path+lists[i]+'/'+lists[i]+str(k)+'.jpeg', tmp)
		k = k+1


def is_jpg(filename):
	try:
		i=Image.open(filename)
		return i.format =='JPEG'
	except IOError:
		return Fals
	
save_path = '/Users/ku/Desktop/SGCN/benchmark/dataset/'
lists = ('agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings', 'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse', 'harbor', 'intersection', 'mediumresidential', 'mobilehomepark', 'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential', 'storagetanks', 'tenniscourt')

for i in range(21):
	for file in os.listdir(save_path + lists[i]):
		temp = is_jpg(save_path + lists[i]+'/'+ file)
		print(temp)