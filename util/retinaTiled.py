from __future__ import print_function, division, absolute_import, unicode_literals
import os 
import sys
UNET_PATH = os.getcwd() + "/../"
sys.path.append(UNET_PATH)

from tf_unet import image_gen
from tf_unet import unet
from tf_unet import util
from readImages import DataProvider, DataProviderTiled
from plotting import plotter
from plotting import logger

BATCH_SIZE = 50
EPOCHS = 20
VALIDATION_SIZE = 516

DROPOUT_KEEP_PROB = 0.6 # keep_prob
DISPLAY_STEP = 2
OUTPUT_PATH = "./retinaModel"

# 564
# 102

def main():
	restore = False
    # adding measurements (loggers) to plotter
	logs = []
	logs.append(logger(label = "train loss",color = "Red"))
	logs.append(logger(label = "validation loss",color = "Blue"))
	logs.append(logger(label = "validation error",color = "Black"))
	logs.append(logger(label = "dice score",color = "Green"))
	plot = plotter(logs, EPOCHS, os.getcwd() + "/" + "plots") 

	# data provider
	dp = DataProviderTiled(splits = 12, batchSize = BATCH_SIZE, validationSize = VALIDATION_SIZE)
	dp.readData()
	print("DONE READING DATA")
	# calculate num of iterations
	iters = dp.getTrainSize()//BATCH_SIZE


	# unet
	opt = {"class_weights":[0.99, 0.01]}
	net = unet.Unet(channels = 1, n_class = 2, layers = 3,\
	 features_root = 16, cost="cross_entropy", cost_kwargs={})

	# trainer
	options = {"momentum":0.2, "learning_rate":0.2,"decay_rate":0.95}


	trainer = unet.Trainer(net, optimizer="momentum",plotter = plot, opt_kwargs=options )
	# train model
	path = trainer.train(dp, OUTPUT_PATH,training_iters = iters,epochs=EPOCHS,\
		dropout=DROPOUT_KEEP_PROB, display_step = DISPLAY_STEP,restore = restore)

	# plot results
	plot.saveLoggers()
	plot.plotLoggers()
	print("DONE")

if __name__ == '__main__':
	main()
