# -*- coding: utf-8 -*-
"""
test functions
"""


import numpy as np
# from keras.models import Model # basic class for specifying and training a neural network
# from keras.layers import Input, Convolution3D, Dense, Dropout, Flatten, Concatenate
# from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
from load_data import load
from preprocess import smooth_data,normalize
from metadata import prepare_metadata


###### get IDs and labels ######
# fn = '/home/yklocal/Downloads/OASIS3_noNA_ybin.csv'
# dirimg = '/home/yklocal/Downloads/Original_cropped_pkl2/'
fn = '/Users/yeunkim/Data/OASIS/OASIS3_noNA_ybin.csv'
dirimg = '/Users/yeunkim/Data/OASIS/Original_cropped_pkl2/'
data = load(fn, dirimg)


###### load in data ######

# X_train0 = smooth_data(data.train_data_matrix, 1)
# X_train0_norm= normalize(data.train_data_matrix)
#
# print(X_train0)
###### smooth and normalize ######


#### test org_data
test = prepare_metadata('/Users/yeunkim/Data/OASIS/csv/X_DataFrame2.csv', '/Users/yeunkim/Data/OASIS/csv/Y_DataFrame.csv',data)

print(test.Y_test_demog)
