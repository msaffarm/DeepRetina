from __future__ import print_function, division, absolute_import, unicode_literals
import os 
import sys
UNET_PATH = os.getcwd() + "/../"
sys.path.append(UNET_PATH)

from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
from readImages import DataProvider
import matplotlib.pyplot as plt
import numpy as np
from plotting import plotter
from plotting import logger

BATCH_SIZE = 1
EPOCHS = 10
VALIDATION_SIZE = 10

DROPOUT_KEEP_PROB = 0.6 # keep_prob
DISPLAY_STEP = 2
OUTPUT_PATH = os.getcwd()+ "/" + "testResults"
MODEL_PATH = ""

# 564
# 102

def main():

	dp = DataProvider(batchSize = BATCH_SIZE, validationSize = VALIDATION_SIZE)
	dp.readData()
	print("DONE READING DATA")
	# calculate num of iterations
	iters = dp.getTrainSize()//BATCH_SIZE
	# unet
	net = unet.Unet(channels = 1, n_class = 2, layers = 3,\
	 features_root = 16, cost="cross_entropy", cost_kwargs={})

	# # trainer
	# options = {"momentum":0.2, "learning_rate":0.2,"decay_rate":0.95}

	# trainer = unet.Trainer(net, optimizer="momentum",plotter = plot, opt_kwargs=options )
	# # train model
	# path = trainer.train(dp, OUTPUT_PATH,training_iters = iters,epochs=EPOCHS,\
	# 	dropout=DROPOUT_KEEP_PROB, display_step = DISPLAY_STEP,restore = restore)

	path = os.getcwd() + "/retinaModel/model.cpkt"
	
	x_test, y_test = dp.getTestData(3, crop=False)
	prediction = net.predict(path, x_test)
	
	# # sanity check
	# fig, ax = plt.subplots(3, 3)
	# ax[0][0].imshow(x_test[0,:,:,0],cmap=plt.cm.gray)
	# ax[0][1].imshow(y_test[0,:,:,1],cmap=plt.cm.gray)
	# ax[0][2].imshow(np.argmax(prediction[0,...],axis =2),cmap=plt.cm.gray)
	# ax[1][0].imshow(x_test[1,:,:,0],cmap=plt.cm.gray)
	# ax[1][1].imshow(y_test[1,:,:,1],cmap=plt.cm.gray)
	# ax[1][2].imshow(np.argmax(prediction[1,...],axis =2),cmap=plt.cm.gray)
	# ax[2][0].imshow(x_test[2,:,:,0],cmap=plt.cm.gray)
	# ax[2][1].imshow(y_test[2,:,:,1],cmap=plt.cm.gray)
	# ax[2][2].imshow(np.argmax(prediction[2,...],axis =2),cmap=plt.cm.gray)
	# plt.show()

	# save test result as image
	# check for path
	if not os.path.lexists(OUTPUT_PATH):
		os.makedirs(OUTPUT_PATH)

	sampleSize = 3
	img = util.combine_img_prediction(x_test[0:sampleSize,...], y_test[0:sampleSize,...]\
		, prediction[0:sampleSize,...])

	util.save_image(img, "%s/%s.jpg"%(os.getcwd()+"/"+"testResults", "testSample"))

	print("Testing error rate: {:.2f}%".format(unet.error_rate(prediction, util.crop_to_shape(y_test, prediction.shape))))



if __name__ == '__main__':
	main()
