# -*- coding: utf-8 -*-
"""
Performs Gaussian smoothing and normalization
"""

import numpy as np
from scipy.ndimage import filters

class preprocess_data(object):

    def __init__(self, data_matrix, smooth_sz, **keyword_parameters):
        self.overallMean = 0
        self.overallStd = 0
        self.data = []
        self.dataSmooth = []

        self.smooth_data(data_matrix, smooth_sz)
        self.normalize(data_matrix, **keyword_parameters)


    def smooth_data(self,data_matrix, smooth_sz):
        dataSmooth = np.zeros((data_matrix.shape[0],data_matrix.shape[1],data_matrix.shape[2],data_matrix.shape[3]))
        for i in range(data_matrix.shape[0]):
          dataSmooth[i] = filters.gaussian_filter(data_matrix[i,:,:,:],smooth_sz)
        self.dataSmooth = dataSmooth


    def normalize(self, data_matrix, **keyword_parameters):
        data = np.zeros((data_matrix.shape[0], data_matrix.shape[1], data_matrix.shape[2], data_matrix.shape[3]))

        if "mean" in keyword_parameters:
            for i in range(data_matrix.shape[0]):
                tmp = (self.dataSmooth[i] - np.mean(self.dataSmooth[i])) / np.std(self.dataSmooth[i])
                # tmp_data = (tmp - self.overallMean) / self.overallStd
                data[i] = tmp
            # self.overallMean = np.mean(data, axis=None)
            # self.overallStd = np.std(data, axis=None)
            data = (data - keyword_parameters['mean']) / keyword_parameters['sd']

        else:

            for i in range(data_matrix.shape[0]):
                tmp = (self.dataSmooth[i] - np.mean(self.dataSmooth[i])) / np.std(self.dataSmooth[i])
                # tmp_data = (tmp - self.overallMean) / self.overallStd
                data[i] = tmp
            self.overallMean = np.mean(data, axis=None)
            self.overallStd = np.std(data, axis=None)

            data = (data - self.overallMean) / self.overallStd
            # print(data)
        # return data
        # print(data)
        self.data = data
