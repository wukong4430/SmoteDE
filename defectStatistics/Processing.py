#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 kicc <wukong4430@gmail.com>
#           Create time: 2019-01-14 10:32
#
# Distributed under terms of the MIT license.

"""
数据读取
"""


import os
import csv
import pandas as pd
import numpy as np

class Processing(object):

    """读取csv数据，转成numpy"""

    def __init__(self):
        """定义数据的文件夹路径 """
        self.dataPath= '../datasetcsv'
        # self.dataPath = '../tinydataset'
        
    def dataload(self):
        """get data from csv

        :path: file locates
        :returns: pandas frame

        """
        dataset = pd.core.frame.DataFrame()

        folderPath = self.dataPath+'/'
        
        for root, dirs, files in os.walk(folderPath):

            for file in files:
                filePath = os.path.join(root, file)
                dataset = pd.read_csv(filePath)
                yield dataset, file


def convert2numpy(dataset):
    """pandas 2 numpy.array

    :dataset: dataset from Processing().dataload()
    :returns: numpy.array

    """
    dataset = dataset.iloc[:, 3:] # eliminate the first 3 columns.
    dataset = np.array(dataset)
    columns = dataset.shape[1]

    dataX, datay = dataset[:, 0:columns-1], dataset[:, -1]
    return dataX, datay
        



if __name__ == '__main__':
    main()
