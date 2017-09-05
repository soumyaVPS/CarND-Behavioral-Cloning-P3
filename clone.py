import csv
import cv2
import numpy

lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurments = []
for line in lines:
    source_path=line[0]
    image = cv2.imread(source_path)
    #print (image.shape);
    images.append(image)
    measurment = float(line[3])
    measurments.append(measurment)

X_train = numpy.array(images)
Y_train = numpy.array(measurments)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, K, MaxPooling2D

model = Sequential()
l1 = Lambda(lambda x: x/255.0 -0.5,input_shape=(160,320,3))
model.add(l1)
print ("Lambda shape", l1.output_shape)
l2 = Convolution2D(6, 5, 5, input_shape=(160, 320, 3),activation=K.relu)
model.add(l2)
print ("Conv2D 1 shape", l2.output_shape)

l3=MaxPooling2D((2, 2))
model.add(l3)
print ("Pooling shape", l3.output_shape)

l4 = Convolution2D(6, 5, 5, input_shape=(78, 158,6),activation=K.relu)
model.add(l4)
print ("Conv2D 2 shape", l4.output_shape)


l5 = Flatten()
model.add(l5)
print ("flatten shape", l5.output_shape)


l6 = Dense(1, activation=K.relu)
model.add(l6)
print ("Dense shape", l6.output_shape)


l7 = Dense(1, activation=K.relu)
model.add(l7)
print ("Dense shape", l7.output_shape)


l8 = Dense(1, activation=K.relu)
model.add(l8)
print ("Dense shape", l8.output_shape)


model.compile(loss='mse',optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
