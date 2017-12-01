# A helper function to read and manipulate retianl images

# import tensorflow as tf
import numpy as np
import random
import os
import sys
from scipy import misc
from scipy import ndimage
import matplotlib.pyplot as plt

# wrap_counting
from wrap_counting import sampler

UNET_PATH = os.getcwd() + "/../"
sys.path.append(UNET_PATH)

DATA_PATH = os.getcwd() + "/../../RetinalDataJohn"
# DATA_PATH = os.getcwd() + "/../sampleData"
SEED = 1234

class DataProvider(object):

	def __init__(self, validationSize = 20, batchSize = 1):
		# super(DataProvider, self).__init__(a_min, a_max)
		# metaData dict
		self.trainData = None
		self.testData = None
		self.validData = None
		self.n_class = 2
		self.a_min = 0
		self.a_max = 255
		self.validationSize = validationSize
		self.trainSize = None
		self.testSize = None
		self.batchSize = batchSize
		self.sampler = None
		self.channels = 1
		self.n_class = 2

	def getBatchSize(self):
		return self.batchSize

	def getValidationSize(self):
		return self.validationSize


	def createMetaDataDict(self, path):

		files = os.listdir(path)
		# print(files)
		# metaData = {image:GT}
		metaData = {}
		images = []
		GTs = []
		# create a list of images and GTs
		for file in files:
			# print(file.split("-")[0])
			if file.split("-")[0] != "GT":
				images.append((file,file.split("_")[-1]))
			else:
				GTs.append((file,file.split("_")[-1]))

		# metaData = {image:GT}
		for img in images:
			for g in GTs:
				if g[1] == img[1]:
					metaData[img[0]] = g[0]

		return metaData

	def createAugmentedData(self, metaDataDict, dataPath):

		augDataImg = []
		augDataGt = []
		for aImg, aGt in metaDataDict.items():
			img = misc.imread(dataPath +"/" + aImg) # read image
			gt = misc.imread(dataPath +"/" + aGt) # read its GT
			augDataImg.append(img)
			augDataGt.append(gt)
			augImg, augGt = self.augmentData(img, gt)
			for i in range(len(augImg)):
				augDataImg.append(augImg[i])
				augDataGt.append(augGt[i])

			del img, gt, augImg, augGt

		return augDataImg, augDataGt

	def augmentData(self, img, gt):
		augImg = []
		augGt = []

		# flip up-down
		augImg.append(np.flipud(img))
		augGt.append(np.flipud(gt))

		# flip right-left
		augImg.append(np.fliplr(img))
		augGt.append(np.fliplr(gt))

		# rotate 90, 180 and 270 clockwise
		for i in range(1,4):
			augImg.append(ndimage.rotate(img, i*90))
			augGt.append(ndimage.rotate(gt, i*90))

		return augImg,augGt

	def readTrainData(self):
		#  read train Data
		train_path = DATA_PATH + "/train"
		trainMetaData = self.createMetaDataDict(train_path)
		self.trainData = self.createAugmentedData(trainMetaData, train_path)
		print("done reading data")
		# extract validation data
		self.validData = self.createValidationData()
		# get train size
		self.trainSize = len(self.trainData[0])

		# create sampler to get samples from train data
		self.sampler = sampler(self.batchSize, self.trainSize, seed = SEED)

	def readTestData(self):
		# read test Data
		test_path = DATA_PATH + "/test"
		testMetaData = self.createMetaDataDict(test_path)
		self.testData = self.createAugmentedData(testMetaData, test_path)
		# get train size
		self.testSize = len(self.testData[0])
		# print(self.testSize)

	def readData(self):
		self.readTrainData()	
		self.readTestData()

	def createValidationData(self):

		trainSize = len(self.trainData[0])
		# randInt = random.sample(range(trainSize), self.validationSize)
		randInt = range(self.validationSize)
		validImg = []
		validGT = []
		for r in randInt:
			validImg.append(self.trainData[0][r])
			validGT.append(self.trainData[1][r])
		# pop validation data from train data
		tempImg = []
		tempGT = []
		for i in range(trainSize):
			if i not in randInt:
				tempImg.append(self.trainData[0][i])
				tempGT.append(self.trainData[1][i])
		
		self.trainData = (tempImg, tempGT)
		return validImg, validGT

	def processLabels(self, label):
		nx = label.shape[1]
		ny = label.shape[0]
		# label = self.normalize(label)
		# print(label.dtype)
		labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
		labels[..., 1] = self.normalize(label)
		labels[..., 0] = self.normalize(~label)
		return labels

	def processData(self, data):
        # normalization
		data = self.normalize(data)
		return np.reshape(data, (data.shape[0], data.shape[1], self.channels))

	def normalize(self,data):
		# check if all zeros or ones
		if np.count_nonzero(data)==0:
			# print("zeros")
			return data
		if np.count_nonzero(data -1 )==0:
			# print("ones")
			return data
		data = np.clip(np.fabs(data), self.a_min, self.a_max)
		data -= np.amin(data)
		data /= (np.amax(data) + 1e-6)
		return data

	def cropImage(self,data):
		m, n = data.shape
		data = data[m/4:-m/4,n/4:-n/4]
		return data

	def getValidationData(self, batchSize,crop=True):
		if batchSize == -1:
			batchSize = self.validationSize

		if crop:
			nx = self.validData[0][0].shape[0]/2
			ny = self.validData[0][0].shape[1]/2
			X = np.zeros((batchSize, nx, ny, self.channels))
			Y = np.zeros((batchSize, nx, ny, self.n_class))
		else:
			nx = self.validData[0][0].shape[0]
			ny = self.validData[0][0].shape[1]
			X = np.zeros((batchSize, nx, ny, self.channels))
			Y = np.zeros((batchSize, nx, ny, self.n_class))		

		selected = random.sample(range(self.validationSize), batchSize)
		for idx, val in enumerate(selected):
			if crop: # crop before processing
				d = self.cropImage(self.validData[0][val])
				l = self.cropImage(self.validData[1][val])
			else:
				d = self.validData[0][val]
				l = self.validData[1][val]	
						
			X[idx] = self.processData(d)
			Y[idx] = self.processLabels(l)

		return X, Y

	def getTestData(self, batchSize, crop = True):
		if batchSize == -1:
			batchSize = self.testSize

		if crop:
			nx = self.testData[0][0].shape[0]/2
			ny = self.testData[0][0].shape[1]/2
			X = np.zeros((batchSize, nx, ny, self.channels))
			Y = np.zeros((batchSize, nx, ny, self.n_class))
		else:
			nx = self.testData[0][0].shape[0]
			ny = self.testData[0][0].shape[1]
			X = np.zeros((batchSize, nx, ny, self.channels))
			Y = np.zeros((batchSize, nx, ny, self.n_class))		

		selected = random.sample(range(self.testSize), batchSize)
		for idx, val in enumerate(selected):
			if crop: # crop before processing
				d = self.cropImage(self.testData[0][val])
				l = self.cropImage(self.testData[1][val])
			else:
				d = self.testData[0][val]
				l = self.testData[1][val]	
			
			print(d.shape)
			print(l.shape)	
			X[idx] = self.processData(d)
			Y[idx] = self.processLabels(l)

		return X, Y

	def __call__(self,crop = True):
		
		# print(self.sampler.getOrder())
		nextIdx = self.sampler.next_inds()
		# print(nextIdx)
		# train_data, labels = self._load_data_and_label()

		if crop:
			nx = self.trainData[0][0].shape[0]/2
			ny = self.trainData[0][0].shape[1]/2
			X = np.zeros((self.batchSize, nx, ny, self.channels))
			Y = np.zeros((self.batchSize, nx, ny, self.n_class))
		else:
			nx = self.trainData[0][0].shape[0]
			ny = self.trainData[0][0].shape[1]
			X = np.zeros((self.batchSize, nx, ny, self.channels))
			Y = np.zeros((self.batchSize, nx, ny, self.n_class))		

		for idx, val in enumerate(nextIdx):
			if crop: # crop before processing
				d = self.cropImage(self.trainData[0][val])
				l = self.cropImage(self.trainData[1][val])
			else:
				d = self.trainData[0][val]
				l = self.trainData[1][val]	
						
			X[idx] = self.processData(d)
			Y[idx] = self.processLabels(l)

		# print(type(X))
		return X, Y

	def getTrainSize(self):
		return self.trainSize


class DataProviderTiled(DataProvider):

	def __init__(self,validationSize = 20, batchSize = 1, splits = 8):
		super(DataProviderTiled, self).__init__(validationSize, batchSize)
		self.splits = splits


	# def createAugmentedData(self, metaDataDict, dataPath):

	# 	augDataImg = []
	# 	augDataGt = []
	# 	for aImg, aGt in metaDataDict.items():
	# 		img = misc.imread(dataPath +"/" + aImg) # read image
	# 		gt = misc.imread(dataPath +"/" + aGt) # read its GT
	# 		# tile img data
	# 		# print(self.splits)
	# 		imgTiles = self.split(img,self.splits)
	# 		for x in imgTiles:
	# 			augDataImg.append(x)

	# 		# tile gt data
	# 		gtTiles = self.split(gt,self.splits)
	# 		for x in gtTiles:
	# 			augDataGt.append(gt)

	# 		augImg, augGt = self.augmentData(img, gt)
	# 		for i in range(len(augImg)):
	# 			for x in self.split(augImg[i],self.splits):
	# 				augDataImg.append(x)
	# 			for x in self.split(augGt[i],self.splits):
	# 				augDataGt.append(x)

	# 		del img, gt, augImg, augGt

	# 	return augDataImg, augDataGt


	def createAugmentedData(self, metaDataDict, dataPath):

		augDataImg = []
		augDataGt = []
		for aImg, aGt in metaDataDict.items():
			img = misc.imread(dataPath +"/" + aImg) # read image
			gt = misc.imread(dataPath +"/" + aGt) # read its GT
			# tile img data
			# print(self.splits)
			imgTiles = self.split(img,self.splits)
			for x in imgTiles:
				augDataImg.append(x)

			# tile gt data
			gtTiles = self.split(gt,self.splits)
			for x in gtTiles:
				augDataGt.append(x)

			del img, gt

		return augDataImg, augDataGt




	def split(self,data, splits):
		tiles = []
		m = data.shape[0]/splits
		n = data.shape[1]/splits
		# print("m" , m)
		# print("n" , n)
		for i in range(splits):
			for j in range(splits):
				tiles.append(data[i*m:(i+1)*m,j*n:(j+1)*n])
		return tiles



			
def main():
	dp = DataProviderTiled(splits = 12 , batchSize = 10)
	dp.readTrainData()
	# x ,y = dp.getTestData(4, crop= False)
	x ,y = dp(crop = False)
	print(dp.getTrainSize())
	# # print(np.max(x))
	# # print(np.max(y))
	# # sanity check
	# print(y.shape)
	# # g = np.reshape(y[...,1],[-1,1])
	# # print(g.shape)
	fig, ax = plt.subplots(2, 2)
	ax[0][0].imshow(x[2,:,:,0],cmap=plt.cm.gray)
	ax[1][0].imshow(y[2,:,:,1],cmap=plt.cm.gray)
	ax[0][1].imshow(x[0,:,:,0],cmap=plt.cm.gray)
	ax[1][1].imshow(y[0,:,:,1],cmap=plt.cm.gray)
	plt.show()
	

if __name__ == '__main__':
	main()
