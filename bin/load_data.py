# -*- coding: utf-8 -*-
"""
Partitions training and test data labels/IDs
"""

import pandas as pd
import numpy as np
import pickle
import os

class load(object):

    def __init__(self, fn, dirimg, dirjac, x, y, z):
        self.fn = fn
        self.dirimg = dirimg
        self.dirjac = dirjac
        self.y_test0_id = []
        self.y_test0 =[]
        self.y_train0_id =[]
        self.y_train0 =[]
        self.train_data_matrix =[]
        self.test_data_matrix =[]
        self.jac_train_data_matrix = []
        self.jac_test_data_matrix =[]
        self.x = x
        self.y = y
        self.z = z

        self.read_csv()
        self.load_img()
        self.load_jacdet()

    def read_csv(self):
        ### Partitioning demographic data : train vs test

        newdemog = pd.read_csv(self.fn)

        # print(newdemog)

        NC = newdemog["Subject"][newdemog["y_bin"] == 0]
        AD = newdemog["Subject"][newdemog["y_bin"] == 1]

        y_test0_nc = NC[0:15]
        y_test0_ad = AD[0:15]

        y_test0_id = np.concatenate([y_test0_nc.values, y_test0_ad.values], axis=0)  # ID for test
        y_test0 = np.concatenate((np.zeros((1, 15)), np.ones((1, 15))), axis=1).flatten()

        y_train0_nc = NC[15:]
        y_train0_ad = AD[15:]

        y_train0_id = np.concatenate([y_train0_nc.values, y_train0_ad.values], axis=0)  # ID for train
        y_train0 = np.concatenate((np.zeros((1, len(y_train0_nc))), np.ones((1, len(y_train0_ad)))), axis=1).flatten()

        self.y_test0_id = y_test0_id
        self.y_test0 = y_test0
        self.y_train0_id = y_train0_id
        self.y_train0 = y_train0

    def load_img(self):
        for root, dirs, files in os.walk(self.dirimg):
            num_train = self.y_train0_id.shape[0]  ###
            num_test = self.y_test0_id.shape[0]  ###
            train_data_matrix = np.ones((num_train, self.x, self.y, self.z))
            test_data_matrix = np.ones((num_test, self.x, self.y, self.z))

            train_i = 0
            test_i = 0
            for file in files:
                #       subjID.append(file.split('.')[0])
                #       path = os.path.join(root,dirs)
                filename = self.dirimg + file

                if np.isin(file[0:9], self.y_train0_id):  ###
                    with open(filename, 'rb') as f:
                        train_data_matrix[train_i, :, :, :] = pickle.load(f)
                        train_i += 1
                elif np.isin(file[0:9], self.y_test0_id):
                    with open(filename, 'rb') as f:
                        test_data_matrix[test_i, :, :, :] = pickle.load(f)
                        test_i += 1
                else:
                    continue
        self.train_data_matrix = train_data_matrix
        self.test_data_matrix = test_data_matrix

    def load_jacdet(self):
        # print('hi')
        for root, dirs, files in os.walk(self.dirjac):
            num_train = self.y_train0_id.shape[0]  ###
            num_test = self.y_test0_id.shape[0]  ###
            jac_train_data_matrix = np.ones((num_train, self.x, self.y, self.z))
            jac_test_data_matrix = np.ones((num_test, self.x, self.y, self.z))

            train_i = 0
            test_i = 0
            for file in files:
                #       subjID.append(file.split('.')[0])
                #       path = os.path.join(root,dirs)
                filename = self.dirjac + file

                if np.isin(file[0:9], self.y_train0_id):  ###
                    with open(filename, 'rb') as f:
                        jac_train_data_matrix[train_i, :, :, :] = pickle.load(f)
                        train_i += 1
                elif np.isin(file[0:9], self.y_test0_id):
                    with open(filename, 'rb') as f:
                        jac_test_data_matrix[test_i, :, :, :] = pickle.load(f)
                        test_i += 1
                else:
                    continue
        self.jac_train_data_matrix = jac_train_data_matrix
        self.jac_test_data_matrix = jac_test_data_matrix


