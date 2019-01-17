#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 kicc <wukong4430@gmail.com>
#           Create time: 2019-01-14 10:15
#
# Distributed under terms of the MIT license.

"""
#Instance represents the number of modules in the release,
#Instance=count(rows)
#Defects represents the total number of defects in the release,
#Defects=sum(bugs)
%Defect represents the percentage of defective-prone modules in the release,
%Defect=#(bug>0)/#Instance * 100%
Max is the maximum value of defects in the release,
Max=max(bugs)
Avg is the average value of defects of all defective-prone modules in the release
(就是#defects这一列的值除以（#Instances乘以%defects）). 类似于下面这个表。
Avg=#Defect/(#Instance * %Defects)
"""
import os
from Processing import Processing, convert2numpy
import fire
import pandas as pd
import numpy as np
import csv

class Statistics(object):

    """Statistics for 5 elements."""

    def __init__(self, data):
        """initial."""
        self.data = data

    def numberInstance(self):
        """ #Instance

        :returns: rows

        """
        rows = self.data.shape[0]
        self.number_instance = rows
        return int(rows)

    def numberDefect(self):
        """ #Defects sum(bugs)

        """
        number_defect = np.sum(self.data)
        return int(number_defect)

    def percentDefect(self):
        number_instance = self.numberInstance()
        percent_defect = np.count_nonzero(self.data) / number_instance
        return np.around(percent_defect * 100, decimals=1)

    def max_(self):
        max_bug = np.max(self.data)
        return int(max_bug)

    def avg_(self):
        number_defect = self.numberDefect()
        number_instance = self.numberInstance()
        percent_defect = self.percentDefect()
        avg_bug = number_defect / (number_instance * percent_defect) * 100.0
        return np.around(avg_bug, decimals=2)


def save_(f, filename, **data):
    """ save result into csv file.

    :f: file path
    :data:
    :returns:

    """
    ele1 = data['number_instance']
    ele2 = data['number_defect']
    ele3 = data['percent_defect']
    ele4 = data['max_bug']
    ele5 = data['avg_bug']
    Release = data['Release']
    print(ele5)
    # dataFrame = pd.DataFrame(**data)
    dataFrame = pd.DataFrame({'Release': Release,
                              '#Instance': ele1,
                              '#Defects': ele2,
                              '%Defects': ele3,
                              'Max': ele4,
                              'Avg': ele5})
    dataFrame.to_csv(f, index=False, sep=',')


def fiveElements():
    """ 统计五个元素


    """
    all_filenames = []
    all_instances = []
    all_defects = []
    all_percent_defects = []
    all_maxs = []
    all_avgs = []
    for dataset, filename in Processing().dataload():
        data = convert2numpy(dataset)
        statics = Statistics(data[1])
        # #Instance
        # print(filename)
        number_instance = statics.numberInstance()
        number_defect = statics.numberDefect()
        percent_defect = statics.percentDefect()
        max_bug = statics.max_()
        avg_bug = statics.avg_()
        # print('\n')
        # print('#Instance :{0}'.format(number_instance))
        # print('#Defects :{0}'.format(number_defect))
        # print('%Defects :{:.1f}'.format(percent_defect))
        # print('max_bug :{0}'.format(max_bug))
        # print('avg_bug :{:.2f}'.format(avg_bug))

        all_filenames.append(filename)
        all_instances.append(number_instance)
        all_defects.append(number_defect)
        all_percent_defects.append(percent_defect)
        all_maxs.append(max_bug)
        all_avgs.append(avg_bug)

    print(all_filenames)
    data_ = {'Release': all_filenames,
             'number_instance': all_instances,
             'number_defect': all_defects,
             'percent_defect': all_percent_defects,
             'max_bug': all_maxs,
             'avg_bug': all_avgs}

    save_(r'fiveElements.csv', filename, **data_)
    pass


def proportion_(dataset, filename):
    """ 统计每个release中各个数量模型的比例
    :returns: 写入文件

    """
    data = convert2numpy(dataset)
    statics = Statistics(data[1])
    max_bug = statics.max_()
    number_instance = statics.numberInstance()
    unique, counts = np.unique(data[1], return_counts=True)
    counter = dict(zip(unique, counts))
    print(counter)
    res = dict()
    res['Release'] = filename
    for i in range(max_bug + 1):
        if i>=9:
            break
        if i in counter:
            res[str(i)] = np.around(counter[i]/number_instance*100.0, decimals=4)
        else:
            res[str(i)] = 0
    if max_bug>8:
        sum_=0
        for j in range(9, max_bug+1):
            if j in counter:
                sum_+=counter[j]
        res['gt 8'] = np.around(sum_/number_instance*100.0, decimals=4)
    dataFrame = pd.DataFrame(res, index=[0])
    with open('proportion.csv', 'a') as f:
        dataFrame.to_csv(f, index=False, header=False)

    pass

def proportion():

    # 先加入header
    with open('proportion.csv', 'w') as f:
    
        columns = ['Release', 'Count 0', 'Count 1', 'Count 2',
                   'Count 3', 'Count 4', 'Count 5',
                   'Count 6', 'Count 7', 'Count 8',
                   'Count gt 8']
        writer = csv.DictWriter(f, fieldnames=columns)

        writer.writeheader()
    # 添加数据
    for dataset, filename in Processing().dataload():
        proportion_(dataset, filename)
        pass

def main():

    # 任务2, 计算比例
    proportion()


    # 任务1， 计算五个元素
    fiveElements()










if __name__ == '__main__':
    fire.Fire()
