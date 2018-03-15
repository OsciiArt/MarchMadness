"""
https://www.kaggle.com/aharless/exclude-same-wk-res-from-nitin-s-surpriseme2-w-nn

"""

import glob, re
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime
from xgboost import XGBRegressor

from keras.layers import Embedding, Input, Dense
from keras import Model
import keras
import keras.backend as K

import matplotlib.pyplot as plt

from sklearn import *
import os, glob

if False:
    datafiles = sorted(glob.glob('../input/**'))
    datafiles = {os.path.basename(file)[:-4]: pd.read_csv(file, encoding='latin-1') for file in datafiles}
    rank = datafiles['MasseyOrdinals']
    rank_types = sorted(rank['SystemName'].unique())
    for i, rank_type in enumerate(rank_types):
        print("processeng: {}".format(rank_type))
        each_rank = rank[rank['SystemName']==rank_type]
        each_rank.columns = ['Season', 'DayNum', 'SystemName', 'Team', rank_type]
        if i==0:
            rank_reshaped = each_rank[['Season', 'DayNum', 'Team', rank_type]]
        else:
            rank_reshaped = pd.merge(rank_reshaped, each_rank[['Season', 'DayNum', 'Team', rank_type]],
                                     how='outer', on=['Season', 'DayNum', 'Team'])
    rank_reshaped = rank_reshaped.fillna(rank_reshaped.mean())
    rank_reshaped.to_csv("../additional/rank.csv", index=None)
else:
    rank_reshaped = pd.read_csv("../additional/ranking_reshape.csv")



#
value_scaler = preprocessing.MinMaxScaler() # TODO rankgaussian is better
X = value_scaler.fit_transform(rank_reshaped)


def get_model():
    inputs = Input((X.shape[1],))

    x = Dense(300, activation='relu')(inputs)
    x = Dense(10, activation='relu', name="feature")(x)
    x = Dense(300, activation='relu')(x)

    outputs = Dense(X.shape[1], activation='relu')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')

    return model


def x_generator(x, batch_size, shuffle=True):
    batch_index = 0
    n = x.shape[0]
    while True:
        if batch_index == 0:
            index_array = np.arange(n)
            if shuffle:
                index_array = np.random.permutation(n)

        current_index = (batch_index * batch_size) % n
        if n >= current_index + batch_size:
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = n - current_index
            batch_index = 0

        batch_x = x[index_array[current_index: current_index + current_batch_size]]

        yield batch_x


def mix_generator(x, batch_size, shuffle=True):
    gen1 = x_generator(x, batch_size, shuffle)
    gen2 = x_generator(x, batch_size, shuffle)
    while True:
        batch1 = next(gen1)
        batch2 = next(gen2)
        new_batch = batch1.copy()
        for i in range(batch1.shape[0]):
            swap_idx = np.random.choice(np.arange(163), int(163*0.07), replace=False)
            new_batch[i, swap_idx] = batch2[i, swap_idx]

        yield (new_batch, batch1)

def valid_generator(x, batch_size, shuffle = False):
    gen1 = x_generator(x, batch_size, shuffle = False)
    while True:
        batch1 = next(gen1)

        yield (batch1, batch1)


# cv

from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, CSVLogger, ReduceLROnPlateau, LearningRateScheduler

def get_callbacks(save_path, lr=0.001):
    csv_logger = CSVLogger(save_path + '_log.csv', append=True)
    # check_path = save_path + '_e{epoch:02d}_vl{val_loss:.5f}.hdf5'
    check_path = save_path
    save_checkpoint = ModelCheckpoint(filepath=check_path, monitor='loss', save_best_only=True)
    lerning_rate_schedular = ReduceLROnPlateau(patience=8, min_lr=lr * 0.00001)

    def lrs(epoch):
        if epoch<100:
            return 1e-4
        elif epoch<200:
            return 1e-5
        else:
            return 1e-6

    learning_rate_Schedular = LearningRateScheduler(lambda epoch: lrs(epoch))
    early_stopping = EarlyStopping(monitor='val_loss',
                                   patience=16,
                                   verbose=1,
                                   min_delta=1e-4,
                                   mode='min')
    Callbacks = [csv_logger,
                 save_checkpoint,
                 # learning_rate_Schedular,
                 # early_stopping
                 ]
    return Callbacks


import time
import os

from sklearn.model_selection import train_test_split, KFold

batch_size = 128
num_epoch = 10
num_fold = 5
format = "%H%M"
ts = time.strftime(format)
base_name = os.path.splitext(__file__)[0] + "_ts" + ts

save_path = base_name
model = get_model()

gen = mix_generator(X, batch_size)

# Fit model
weight_path = "../model/" + save_path + '.hdf5'
callbacks = get_callbacks(weight_path)
model.fit_generator(generator=gen,
                    steps_per_epoch=np.ceil(X.shape[0] / batch_size),
                    epochs=num_epoch,
                    verbose=1,
                    callbacks=callbacks,
                    )
x = model.get_layer("feature").output

model2 = Model(inputs=model.input, outputs=x)
model2.compile(loss='mse', optimizer='adam')
predict = model2.predict(X, batch_size)
np.save("../additional/rank_feat2.npy", predict)
print(predict.shape)