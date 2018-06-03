# -*- coding: utf-8 -*-
"""
Performs Gaussian smoothing and normalization
"""

import numpy as np
from scipy.ndimage import filters


def smooth_data(data_matrix, smooth_sz):
    dataSmooth = np.zeros((data_matrix.shape[0],data_matrix.shape[1],data_matrix.shape[2],data_matrix.shape[3]))
    for i in range(data_matrix.shape[0]):
      dataSmooth[i] = filters.gaussian_filter(data_matrix[i,:,:,:],smooth_sz)
    return dataSmooth


def normalize(data_matrix):
    data = np.zeros((data_matrix.shape[0], data_matrix.shape[1], data_matrix.shape[2], data_matrix.shape[3]))
    overallMean = np.mean(data_matrix, axis=None)
    overallStd = np.std(data_matrix, axis=None)
    for i in range(data_matrix.shape[0]):
        tmp = (data_matrix[i] - np.mean(data_matrix[i])) / np.std(data_matrix[i])
        tmp_data = (tmp - overallMean) / overallStd
        data[i] = tmp
    return data

