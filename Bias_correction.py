#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 14:24:44 2023

@author: asam
"""

import pandas as pd
dataframe1 = pd.read_excel('data.xlsx', sheet_name='Sheet1')
cal = dataframe1.to_numpy()
dataframe2 = pd.read_excel('data.xlsx', sheet_name='Sheet2')
val = dataframe2.to_numpy()
val1 =val
from sklearn.metrics import mean_absolute_error as mae
mae(cal[:,1],cal[:,0])


import numpy as np
import tensorflow as tf
keras = tf.keras

train_dataset = tf.data.Dataset.from_tensor_slices((cal[:,0], cal[:,1]))
test_dataset = tf.data.Dataset.from_tensor_slices((val[:,0], val[:,1]))


BATCH_SIZE = 371
SHUFFLE_BUFFER_SIZE = 10

train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

model1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(1,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

model2 = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Reshape((1, 1, 1)),
    keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="linear"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='linear'),
    tf.keras.layers.Dense(128, activation='linear'),
    tf.keras.layers.Dense(1)
])



model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.6),
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])



model.fit(train_dataset, epochs=10)
#model.evaluate(test_dataset)


arr5 = tf.data.Dataset.from_tensor_slices(val[:,0]).batch(BATCH_SIZE)
predictions1 = model.predict(arr5)
mae(predictions1 ,val[:,1])
