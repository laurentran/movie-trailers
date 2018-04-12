from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import sys

def main():
	# dimensions of our images.
	img_width, img_height = 256, 256
	batch_size = 1
 
	#test_model = load_model('first_try.h5')
	test_model = load_model('vgg19_1_trained_w25epochs.h5')
 
	validation_data_dir='../data/train/'
 
	test_datagen = ImageDataGenerator(rescale=1. / 255)

	validation_generator = test_datagen.flow_from_directory(
    		validation_data_dir,
    		target_size=(img_width, img_height),
    		batch_size=batch_size,
    		class_mode='binary')

	print(len(validation_generator.filenames))
	predictions=test_model.predict_generator(validation_generator,len(validation_generator.filenames));
	print(predictions)


if __name__ == "__main__":
	main()


