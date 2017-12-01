import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle as pk


IMAGE_SIZE = 1024 # size of final image
DATA_PATH = os.getcwd()+"/sampleData" # Data path

images = os.listdir(DATA_PATH) # list of images in Data path
image = DATA_PATH +"/"+ images[0]
print(image)
f = misc.imread(image) # read image

print(f.shape)
h,w = f.shape[0],f.shape[1]
AspectRatio = w/h
print(AspectRatio)
f = misc.imresize(f,IMAGE_SIZE/h) 
print(f.shape)
# plt.imshow(f)
# plt.show()



img1 = tf.convert_to_tensor(f)

resized_image = tf.image.resize_image_with_crop_or_pad(img1,1024,1024)
# # resized_image = tf.image.resize_images(img1,[1024,1024])
# # resized_image = tf.image.central_crop(img1, 0.8)



with tf.Session() as sess:
	print(sess.run(resized_image))
	print(resized_image.eval().shape)
	plt.imshow(resized_image.eval())
	plt.show()

