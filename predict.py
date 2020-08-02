import numpy as np
from classifiers import *
from pipeline import *
from tensorflow.keras import backend as K 

from tensorflow.keras.preprocessing.image import ImageDataGenerator

classifier = Meso4()
classifier.load('weights/Meso4_DF')


dataGenerator = ImageDataGenerator(rescale=1./255)
generator = dataGenerator.flow_from_directory(
        'test_images',
        target_size=(256, 256),
        batch_size=2,
        class_mode='binary',
        subset='training')

# 3 - Predict
X, y = generator.next()

print (classifier.get_accuracy(X, y), '\n')
#print('Predicted :', classifier.predict(X), '\nReal class :', y)