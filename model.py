import pandas as pd
import numpy as np

import cv2
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

def readin_image_angle(data_row):
    steering_center = data_row[3]

    correction = 0.15
    steering_left = steering_center + correction
    steering_right = steering_center - correction

    img_center = cv2.cvtColor(cv2.imread(data_row[0].strip()),cv2.COLOR_BGR2RGB)
    img_left = cv2.cvtColor(cv2.imread(data_row[1].strip()),cv2.COLOR_BGR2RGB)
    img_right = cv2.cvtColor(cv2.imread(data_row[2].strip()),cv2.COLOR_BGR2RGB)

    return [img_center, img_left, img_right],[steering_center, steering_left, steering_right]

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for _,batch_sample in batch_samples.iterrows():
                feature_set,label_set = readin_image_angle(batch_sample)
                images.extend(feature_set)
                angles.extend(label_set)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


driving_log = pd.read_csv('driving_log.csv')

train_samples, validation_samples = train_test_split(driving_log, test_size=0.2)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
#model.add(Cropping2D(cropping=((60,34), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Convolution2D(32, 3, 3))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Activation('sigmoid'))
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('sigmoid'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='mse', optimizer='adam')
model.summary()
model.fit_generator(train_generator, samples_per_epoch=len(train_samples) * 3, 
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples) * 3, 
                    nb_epoch=1)
model.save('model.h5')
