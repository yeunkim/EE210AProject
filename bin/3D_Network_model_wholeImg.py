# -*- coding: utf-8 -*-
"""
Runs Convolutional Neural Network on entire image
"""


import numpy as np
from keras import optimizers
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution3D, Dense, Dropout, Flatten, Concatenate, Merge, merge, regularizers, BatchNormalization
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from load_data import load
from preprocess import preprocess_data
from metadata import prepare_metadata


###### get IDs and labels ######
fn = '/home/yklocal/Downloads/OASIS3_noNA_ybin.csv'
dirimg = '/data/OASIS/wholeImg/'
dirjac = '/home/yklocal/Downloads/Original_cropped_pkl2/'
data = load(fn, dirimg, dirjac,176, 208, 176)
Y_train0 = data.y_train0
Y_test0 = data.y_test0

###### smooth and normalize ######
# X_train0 = smooth_data(data.train_data_matrix, 1)
# X_train0 = normalize(X_train0)
X_train0_pre = preprocess_data(data.train_data_matrix, 1)
X_train0 =X_train0_pre.data
X_test0_pre = preprocess_data(data.test_data_matrix, 1, mean=X_train0_pre.overallMean, sd=X_train0_pre.overallStd)
X_test0 =X_test0_pre.data

###### set data #######
X_train, Y_train = X_train0, Y_train0
X_test, Y_test = X_test0, Y_test0

####Some Hyperparameters
batch_size = 3
num_epochs = 25
num_classes = 2

conv_depth_1 = 45
conv_depth_2 = 45
conv_depth_3 = 60
conv_depth_4 = 128
conv_depth_5 = 256
conv_depth_6 = 512
drop_prob_1 = 0.5
drop_prob_2 = 0.25
hidden_size = 512

number_of_classes = 2
Y_train = np_utils.to_categorical(Y_train, number_of_classes)
Y_test = np_utils.to_categorical(Y_test, number_of_classes)

X_train = np.expand_dims(X_train0, axis=4)

train_set = X_train
test_set = np.expand_dims(X_test, axis=4)

inp = Input(shape=(X_train0.shape[1],X_train0.shape[2],X_train0.shape[3],1))

conv_1_3 = Convolution3D(conv_depth_1, 7,7,7, padding='same',activation='relu',subsample=(2,2,2), kernel_regularizer=regularizers.l1(0.3))(inp)

drop_1 = Dropout(drop_prob_1)(conv_1_3)

conv_2 = Convolution3D(conv_depth_2, 7,7,7, padding='same',activation='relu',subsample=(2,2,2), kernel_regularizer=regularizers.l1(0.3))(drop_1)
drop_2 = Dropout(drop_prob_1)(conv_2)

conv_3 = Convolution3D(conv_depth_3, 5,5,5, padding='same',activation='relu',subsample=(2,2,2), kernel_regularizer=regularizers.l1(0.3))(drop_2)
drop_3 = Dropout(drop_prob_1)(conv_3)

conv_4 = Convolution3D(conv_depth_4, 3,3,3, padding='same',activation='relu',subsample=(2,2,2))(drop_3)
drop_4 = Dropout(drop_prob_1)(conv_4)

flat = Flatten()(drop_4)
# hidden = Dense(hidden_size, activation='relu')(flat)
hidden = Dense(512, activation='relu')(flat)
norm = BatchNormalization()(hidden)

drop_7 = Dropout(drop_prob_2)(norm)

out = Dense(num_classes,activation='softmax', name='feat')(drop_7)

model = Model(inputs=inp, outputs=out)

opt = optimizers.Adam(lr=0.00001)
model.compile(loss='binary_crossentropy',
             optimizer=opt,
             metrics=['accuracy', 'mse'])

model.fit(train_set,Y_train, shuffle=True,
         batch_size=batch_size,epochs=num_epochs,
         verbose=1, validation_data=(test_set, Y_test))
