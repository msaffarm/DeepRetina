import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import pickle as pk
import os


IMAGE_SIZE = 1024 # size of final image
DATA_FOLDERS = ["trainSet" + str(i+1) for i in range(5)]
DATA_PATH = "/home/msaffarm/RetinaData" # Data path
# DATA_PATH = "/home/msaffarm/DeepRetina/sampleData"


def main():
	partNums = len(DATA_FOLDERS)
 	# write images 
	for part in range(partNums):
		# list of images in Data path
		inputFolder = DATA_PATH + "/" + DATA_FOLDERS[part]
		images = os.listdir(inputFolder) 
		totalNum = len(images)
		print("Total number of files: " + str(totalNum))
		counter = 1
		# output path
		outputPath = DATA_PATH + "/" + DATA_FOLDERS[part] + "-comp"
		if not os.path.lexists(outputPath):
			os.makedirs(outputPath)
		# process images
		for image in images:
			print("Part " + str(part+1) + " Progress: " +\
			 "{0:.2f}".format(round((counter/totalNum)*100,2)) + "%")
			f = misc.imread(inputFolder +"/" + image) # read image
			r = misc.imresize(f,IMAGE_SIZE/f.shape[0]) #resize images
			misc.toimage(r).save(outputPath + "/" + image+'.jpg') # save resized image
			counter += 1


if __name__ == '__main__':
	main()
