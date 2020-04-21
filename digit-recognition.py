#!/usr/bin/env python

import os
import numpy as np
import pandas as pd
from PIL import Image

# Bangla digit dataset is available from https://www.kaggle.com/c/numta.

threshold = 160
binarizer = lambda x : 255 if x > threshold else 0

# Load TRAIN data from CSV files
train = pd.read_csv("numta/training-a.csv", header=0)

n = len(train.index)
width = 180
height = 180

train_x = np.zeros((n, width * height), dtype=int)
print(train_x.shape)

i = 0
for row in train.itertuples():
    filepath = 'numta/training-a/' + row.filename
    img = Image.open(filepath).convert("L").point(binarizer, mode='1')
    train_x[i] = np.array(img, dtype=int).ravel()
    i += 1

train_y = train['digit']
print(train_y.value_counts().sort_index())
train_y = train_y.to_numpy()

# Load TEST dataset
files = os.listdir('numta/testing-a/')
m = len(files)

test_x = np.zeros((m, width * height), dtype=int)

i = 0;
for file in files:
    img = Image.open('numta/testing-a/' + file).convert("L").point(binarizer, mode='1')
    test_x[i] = np.asarray(img, dtype=int).ravel()
    i += 1

# Run Deep Learning

from tensorflow import keras
from tensorflow.keras import layers, callbacks
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization
from kerastuner.applications import HyperResNet

def build_model(hp):
    model = keras.Sequential()

    for layer in range(hp.Int("layers", min_value = 1, max_value = 4, step = 1)):
        model.add(layers.Dense(
            units = hp.Int(
                "units_" + str(layer),
                min_value = 32,
                max_value = 512,
                step = 32),
            activation = "relu"))
        
        model.add(layers.Dropout(
            hp.Float(
                "dropout",
                0,
                0.5,
                step = 0.1)))

    model.add(layers.Dense(10, activation = "softmax"))
    model.compile(
        optimizer = keras.optimizers.Adam(
            hp.Choice(
                "learning_rate",
                values = [1e-2, 1e-3, 1e-4])
            ),
        loss = "sparse_categorical_crossentropy",
        metrics =[ "accuracy" ])

    return model

tuner = BayesianOptimization(build_model,
            objective = "val_loss",
            max_trials = 32,
            num_initial_points = 8,
            directory = "tuning",
            project_name = "bayesi")

tuner.search_space_summary()

tuner.search(
    train_x,
    train_y,
    batch_size = 1024,
    epochs = 30,
    validation_split = 0.1,
    callbacks = [
        callbacks.ReduceLROnPlateau(
            monitor = "val_loss",
            patience = 2),
        callbacks.EarlyStopping(
            monitor = "val_loss",
            patience = 4,
            restore_best_weights = True) ])

tuner.results_summary()
model = tuner.get_best_models(num_models = 1)[0]
model.summary()

hyperparameters = tuner.get_best_hyperparameters(num_trials = 1)[0].get_config()
print(hyperparameters["values"])

probabilities = model.predict(test_x)
test_y = model.predict_classes(test_x)

with open("prob-b.csv", "w") as prob_file:
    prob_file.write("id,prob0,prob1,prob2,prob3,prob4,prob5,prob6,prob7,prob8,prob9\n")
    for i in range(test_x.shape[0]):
        prob_file.write(str(i).zfill(5) + "," + ",".join([ str(p) for p in probabilities[i] ]) + "\n")

with open("pred-b.csv", "w") as pred_file:
    pred_file.write("id,pred\n")

    for i in range(test_x.shape[0]):
        pred_file.write(str(i).zfill(5) + "," + str(test_y[i]) + "\n")
