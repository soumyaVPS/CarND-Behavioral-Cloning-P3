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
    print (image.shape);
    images.append(image)
    measurment = float(line[3])
    measurments.append(measurment)

X_train = numpy.array(images)
Y_train = numpy.array(measurments)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

model = Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5,input_shape=(160,320,3)))
model.add(Flatten())
model.add(Dense(1))
model.compile(loss='mse',optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
model.save('model.h5')