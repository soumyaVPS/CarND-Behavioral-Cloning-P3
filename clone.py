import csv
from random import shuffle

import cv2
import numpy

import numpy as np
import sklearn
import sys


def read_csv (dir, samples,type="train") :
    name = dir + '/driving_log_train.csv'
    if type=='val':
        dir + '/driving_log_val.csv'
    with open(name) as csvfile:
        reader = list(csv.reader(csvfile))

        for line in reader:
            line[0] = dir + "/" + line[0].strip()
            line[1] = dir + "/" + line[1].strip()
            line[2] = dir + "/" + line[2].strip()

            samples.append(line)

train_samples=[]
validation_samples = []
new_model = sys.argv[1]
c_r_l = sys.argv[2]


for dir in sys.argv[2:] :
    read_csv(dir, train_samples)
    read_csv(dir, validation_samples)
print ("training samples", len(train_samples))
print("validation_samples", len(validation_samples))
def generator(samples, batch_size=32):
    global c_r_l
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for line in batch_samples:
                source_path = line[0]
                measurment = float(line[3])
                for c_r_l in range(3):
                    if c_r_l == 1:
                        source_path = line[1]
                        measurment += 0.25

                    elif c_r_l == 2:
                        source_path = line[2]
                        measurment -= 0.25
                    image = cv2.imread(source_path)
                    img_g = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                    ch = cv2.split(img_g)
                    images.append(ch[1])
                    angles.append(measurment)

                    #flip each image and angle
                    images.append(cv2.flip(ch[1], 1))
                    angles.append(measurment * -1.0)
            X_train = np.array(images)
            X_train = np.expand_dims(X_train,4)
            y_train = np.array(angles)
            yield X_train, y_train

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, K, MaxPooling2D, Dropout, Activation

if new_model == 'new':
    model = Sequential()
    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=(160, 320, 1)))
    model.add(Convolution2D(6, 5, 5, activation="relu", dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(16, 5, 5, activation="relu", dim_ordering='tf'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(84))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')

else:
    print("Load old model")
    model = load_model('model.h5')


#model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

    #model.fit_generator(train_generator, samples_per_epoch=len(train_samples),  nb_epoch=7)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)
#batch_size 32*6 192

model.fit_generator(train_generator,samples_per_epoch=len(train_samples)*6,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples)*6,
                    nb_epoch=2)

model.save('model.h5')
