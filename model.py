import argparse
import pandas as pd
import numpy as np

import cv2
import sklearn
from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D
from keras.layers.advanced_activations import ELU

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

resize_col = 200
resize_row = 160
def readin_image_angle(data_row):
    '''
    Main image factory for creating augmented images.
    Randomly select left or right image, so as image flipping.
    '''
    steering = data_row[3]
    correction = 0.25

    select = np.random.randint(3)
    img = cv2.cvtColor(cv2.imread(data_row[select].strip()),cv2.COLOR_BGR2HSV)

    if select == 1:
        steering += correction

    if select == 2:
        steering -= correction

    flip = np.random.randint(2)
    if flip:
        img = np.fliplr(img)
        steering = -steering

    return cv2.resize(img,(resize_col, resize_row),interpolation=cv2.INTER_AREA),steering

def generator(samples, batch_size=32):
    '''
    Generator for feeding the model with processed images.
    '''
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        offset = num_samples // 2
        batch_samples = samples[offset:offset+batch_size]

        images = []
        angles = []
        for _,batch_sample in batch_samples.iterrows():
            img,steering = readin_image_angle(batch_sample)
            images.append(img)
            angles.append(steering)

        X_train = np.array(images)
        y_train = np.array(angles)
        yield sklearn.utils.shuffle(X_train, y_train)

parser = argparse.ArgumentParser(description='Behavior model training.')

parser.add_argument('prefix', nargs='?', type=str, default='', help='Directory path which contains the image data.')
parser.add_argument('epoch', nargs='?', type=int, default=2, help='Number of epochs.')

args = parser.parse_args()

driving_log = pd.read_csv(args.prefix + 'driving_log.csv')

train_generator = generator(driving_log, batch_size=100)

# Use Nvidia model, introduce ELU for nonlinearity
model = Sequential()
model.add(Cropping2D(cropping=((60,23), (0,0)), input_shape=(resize_row,resize_col,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
model.add(ELU())
model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
model.add(ELU())
model.add(Convolution2D(64, 3, 3))
model.add(ELU())
model.add(Convolution2D(64, 3, 3))
model.add(ELU())
model.add(Flatten())
model.add(Dense(100))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(10))
model.add(ELU())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.summary()
print('{} samples'.format(len(driving_log)))
model.fit_generator(train_generator, samples_per_epoch=40000,
                    nb_epoch=args.epoch)
model.save('model.h5')
