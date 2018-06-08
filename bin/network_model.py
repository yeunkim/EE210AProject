# -*- coding: utf-8 -*-
"""
Runs Convolutional Neural Network
"""


import numpy as np
from keras import optimizers
from keras.models import Model # basic class for specifying and training a neural network
from keras.layers import Input, Convolution3D, Dense, Dropout, Flatten, concatenate, Merge, merge, regularizers, BatchNormalization, initializers, MaxPool3D
from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from load_data import load
from preprocess import preprocess_data
from metadata import prepare_metadata
from sklearn.utils import shuffle

###### get IDs and labels ######
#fn = '/home/yklocal/Downloads/OASIS3_noNA_ybin.csv'
fn = '/home/yklocal/Downloads/new_Y_Data.csv'
dirimg = '/home/yklocal/Downloads/Original_cropped_pkl2/'
dirjac = '/home/yklocal/Downloads/Jac_cropped/'
data = load(fn, dirimg, dirjac, 128, 64, 64)
Y_train0 = data.y_train0
Y_test0 = data.y_test0


###### load in data ######


###### smooth and normalize ######
# X_train0 = data.train_data_matrix
# X_test0 = data.test_data_matrix
# X_train0 = normalize(X_train0)
X_train0_pre = preprocess_data(data.train_data_matrix, 0)
X_train0 =X_train0_pre.data
X_test0_pre = preprocess_data(data.test_data_matrix, 0, mean=X_train0_pre.overallMean, sd=X_train0_pre.overallStd)
X_test0 =X_test0_pre.data

X_train_jac0_pre = preprocess_data(data.jac_train_data_matrix, 0)
X_train_jac0 =X_train_jac0_pre.data
X_test_jac0_pre = preprocess_data(data.jac_test_data_matrix, 0, mean=X_train_jac0_pre.overallMean, sd=X_train_jac0_pre.overallStd)
X_test_jac0 =X_test_jac0_pre.data

###### set data #######
X_train, Y_train = X_train0, Y_train0
X_test, Y_test = X_test0, Y_test0
# X_train_jac0 = data.jac_train_data_matrix
# X_test_jac0 = data.jac_test_data_matrix

####Some Hyperparameters
batch_size = 15
num_epochs = 10
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

####Model Functional


number_of_classes = 2
np.random.seed(1)
np.random.shuffle(Y_train)
Y_train = np_utils.to_categorical(Y_train, number_of_classes)
np.random.seed(1)
np.random.shuffle(Y_test)
Y_test = np_utils.to_categorical(Y_test, number_of_classes)

np.random.seed(1)
np.random.shuffle(X_train0)
X_train = np.expand_dims(X_train0, axis=4)
train_set = X_train

# train_set = np.zeros((X_train0.shape[0], 1, X_train0.shape[1],X_train0.shape[2],X_train0.shape[3]))
# for h in xrange(X_train0.shape[0]):
#     train_set[h][0][:][:][:] = X_train0[h,:,:,:]
#
# test_set = np.zeros((X_test.shape[0], 1, X_test.shape[1],X_test.shape[2],X_test.shape[3]))
# for h in xrange(X_test.shape[0]):
#     test_set[h][0][:][:][:] = X_test[h,:,:,:]

# print(X_train.shape)

np.random.seed(1)
np.random.shuffle(X_train_jac0)
X_train_jac = np.expand_dims(X_train_jac0, axis=4)

np.random.seed(1)
np.random.shuffle(X_test_jac0)
X_test_jac = np.expand_dims(X_test_jac0, axis=4)

np.random.seed(1)
np.random.shuffle(X_test)
test_set = np.expand_dims(X_test, axis=4)



print(train_set)
print('\n')
print(Y_train)


# inp = Input(shape=(X_train.shape[1], X_train.shape[2],X_train.shape[3],X_train.shape[4]))
inp = Input(shape=(X_train0.shape[1],X_train0.shape[2],X_train0.shape[3],1))
jacinp = Input(shape=(X_train0.shape[1],X_train0.shape[2],X_train0.shape[3],1))

x = concatenate([inp, jacinp])

# conv_1_1 = Convolution3D(conv_depth_1, 7,7,7, padding='same',activation='relu',subsample=(2,2,2))(inp)
# conv_1_2 = Convolution3D(conv_depth_1, 6,6,6, padding='same',activation='relu',subsample=(2,2,2))(inp)
conv_1_3 = Convolution3D(conv_depth_1, 5,5,5, padding='same',
                         activation='relu',subsample=(2,2,2),
                         kernel_regularizer=regularizers.l1(0.3),
                         bias_initializer=initializers.Constant(0.01),
                         kernel_initializer=initializers.glorot_normal(seed=1)
                         # kernel_initializer='random_uniform'
                         )(x)

# conv_1 = Concatenate([conv_1_1, conv_1_2, conv_1_3])
# conv_1 = merge([conv_1_1, conv_1_2, conv_1_3], mode='concat', concat_axis=1)

drop_1 = Dropout(drop_prob_1)(conv_1_3)

conv_2 = Convolution3D(conv_depth_2, 5,5,5, padding='same',
                       activation='relu',subsample=(2,2,2),
                       kernel_regularizer=regularizers.l1(0.3),
                       bias_initializer=initializers.Constant(0.01),
                       kernel_initializer=initializers.glorot_normal(seed=1)
                       # kernel_initializer='random_uniform'
                       )(drop_1)
drop_2 = Dropout(drop_prob_1)(conv_2)

#max1 = MaxPool3D((8,8,8))(drop_2)

conv_3 = Convolution3D(conv_depth_3, 5,5,5, padding='same',
                       activation='relu',subsample=(2,2,2),
                       bias_initializer=initializers.Constant(0.01),
                       kernel_regularizer=regularizers.l1(0.3),
                       kernel_initializer=initializers.glorot_normal(seed=1)
                       # kernel_initializer='random_uniform'
                       )(drop_2)

drop_3 = Dropout(drop_prob_1)(conv_3)
#
conv_4 = Convolution3D(conv_depth_4, 3,3,3, padding='same',
                       activation='relu',subsample=(2,2,2),
                       bias_initializer=initializers.Constant(0.01),
                       kernel_regularizer=regularizers.l1(0.3),
                       kernel_initializer=initializers.glorot_normal(seed=1)
                       # kernel_initializer='random_uniform'
                       )(drop_3)
drop_4 = Dropout(drop_prob_1)(conv_4)

# max2 = MaxPool3D((4,4,4))(drop_4)
#
# conv_5 = Convolution3D(conv_depth_5, 3,3,3, padding='same',activation='relu',subsample=(2,2,2))(drop_4)
# drop_5 = Dropout(drop_prob_1)(conv_5)
#
# conv_6 = Convolution3D(conv_depth_5, 3,3,3, padding='same',activation='relu',subsample=(2,2,2))(drop_5)
# drop_6 = Dropout(drop_prob_1)(conv_6)

flat = Flatten()(drop_4)
# hidden = Dense(hidden_size, activation='relu')(flat)
hidden = Dense(512, activation='relu',
               bias_initializer=initializers.Constant(0.01),
               kernel_regularizer=regularizers.l1(0.3),
               kernel_initializer=initializers.glorot_normal(seed=1)
               # kernel_initializer='random_uniform'
               )(flat)
norm = BatchNormalization()(hidden)

drop_7 = Dropout(drop_prob_2)(norm)

out = Dense(num_classes,activation='softmax', name='feat')(drop_7)

model = Model(inputs=[inp, jacinp],outputs=out)
# model = Model(inputs=inp,outputs=out)

# print(model.get_weights())

# print(inp._keras_shape)
# print(conv_1_3._keras_shape)
# print(conv_2._keras_shape)
# print(conv_3._keras_shape)
# print(conv_4._keras_shape)
# print(conv_5._keras_shape)
# print(conv_6._keras_shape)

# intermediate_layer_model = Model(inputs=model.input,
#                                  outputs=model.get_layer('feat').output) ### pull out features

# intermediate_output = intermediate_layer_model.predict(train_set)
# intermediate_output_test = intermediate_layer_model.predict(test_set)


opt = optimizers.Adam(lr=0.000001)
model.compile(loss='binary_crossentropy',
             optimizer=opt,
             metrics=['accuracy', 'mse'])

# X_train_new, X_val_new, y_train_new,y_val_new = train_test_split(train_set, Y_train, test_size=0.2, random_state=4)
# print(Y_train.shape)


#
model.fit([train_set, X_train_jac],Y_train, shuffle=False,
         batch_size=batch_size,epochs=num_epochs,
         verbose=1, validation_data=([test_set, X_test_jac], Y_test))
print(model.evaluate([test_set, X_test_jac], Y_test, verbose=1))

# model.fit(train_set,Y_train, shuffle=True,
#          batch_size=batch_size,epochs=num_epochs,
#          verbose=1, validation_data=(test_set, Y_test))
# print(model.evaluate(test_set, Y_test, verbose=1))#


# va
#print(train_set)
#print('\n')
#print(Y_train)
# validation_data=(test_set, Y_test)
# print(Y_train)
# print('\n now test')
# print(Y_test)
# print('\n now predict')
print(model.predict([test_set, X_test_jac]))

modelname = '3epochs'
model.save("oasis_{0}.h5".format(modelname))

############# prepare metadata #######
# demogfn = '/Users/yeunkim/Data/OASIS/csv/X_DataFrame2.csv'
# measures = '/Users/yeunkim/Data/OASIS/csv/Y_DataFrame.csv'
#
# metadata =  prepare_metadata(demogfn, measures,data)
#
# print(intermediate_output.shape)