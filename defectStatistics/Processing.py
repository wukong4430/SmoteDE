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
        # self.dataPath= '../../datasetcsv'
        self.dataPath = '../../pre-post-csv'
        
    def dataload(self):
        """get data from csv

        :path: file locates
        :returns: pandas frame

        """
        dataset = pd.core.frame.DataFrame()

        folderPath = self.dataPath+'/'
        
        for root, dirs, files in os.walk(folderPath):

            for file in files:
                print(f'file = {file}')
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
    print(f'shape of dataset = {dataset.shape}')
    
    columns = dataset.shape[1]

    dataX, datay = dataset[:, 0:columns-1], dataset[:, -1]
    return dataX, datay
        

def convert2numpy_twoY(dataset):
    """pandas 2 numpy.array
       对应于3月12日的任务,
       pre是一个y1, 位于第三列
       post是一个y2, 位于第四列
       前两列信息没用

    :dataset: dataset from Processing().dataload()
    :returns: numpy.array

    """
    dataset = dataset.iloc[:, 2:] # eliminate the first 2 columns.
    dataset = np.array(dataset)
    print(f'shape of dataset = {dataset.shape}')
    columns = dataset.shape[1]

    dataX, pre, post = dataset[:, 2:columns], dataset[:, 0], dataset[:, 1]
    print(f'pre = {pre}')
    print(f'post = {post}')
    return dataX, pre, post


if __name__ == '__main__':
    for dataset, filename in Processing().dataload():
        print(f'filename = {filename}')

        X, y_pre, y_post = convert2numpy_twoY(dataset)
        print(X.shape)
        print(y_pre.shape)
        print(y_post.shape)
        # break

