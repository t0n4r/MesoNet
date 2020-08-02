import numpy as np
from classifiers import *
from pipeline import *
from tensorflow.keras import backend as K 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_width, img_height = 256, 256

train_data_dir = 'C:\\Users\\Murshed Al-Amin\\Desktop\\DeepFake\\deepfake_database\\train_test'
validation_data_dir = 'C:\\Users\\Murshed Al-Amin\\Desktop\\DeepFake\\deepfake_database\\validation'

n_train_samples = 12361
n_validation_samples = 7148
epochs = 10
batch_size = 16


if K.image_data_format() == 'channels_first': 
    input_shape = (3, img_width, img_height) 
else: 
    input_shape = (img_width, img_height, 3)


train_datagen = ImageDataGenerator( 
    rescale=1. / 255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True) 
  
test_datagen = ImageDataGenerator(rescale=1. / 255) 
  
train_generator = train_datagen.flow_from_directory( 
    train_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode='binary')
  
validation_generator = test_datagen.flow_from_directory( 
    validation_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode='binary') 

classifier = Meso4()

classifier.model.fit_generator(train_generator, 
    steps_per_epoch = n_train_samples // batch_size, 
    epochs = epochs, validation_data = validation_generator, 
    validation_steps = n_validation_samples // batch_size)

classifier.model.save_weights('meso4.h5')