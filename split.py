import csv
import random

import sys

import numpy
data = []
dir = sys.argv[1]
with open(dir + '/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    data = list(reader)
random.shuffle(data)

train_data = data[:int(len(data)*0.8)]
val_data = data[int(len(data)*0.8):]
with open(dir+"/driving_log_train.csv","w") as f:
    wr = csv.writer(f, delimiter=",")
    for line in train_data:
        wr.writerow(line)

with open(dir+"/driving_log_val.csv","w") as f:
    wr = csv.writer(f,  delimiter=",")
    for line in val_data:
        wr.writerow(line)