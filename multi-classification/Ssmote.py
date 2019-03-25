#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 kicc <wukong4430@gmail.com>
#           Create time: 2019-01-16 19:48
#
# Distributed under terms of the MIT license.

from concatModule import ConcatMinority
from statistics import Statistics
from sklearn.neighbors import NearestNeighbors
from utils import tuple2dict, arrayConcat
import numpy as np


class Ssmote(object):
    """
    基于ssmote算法合成数据
    导入数据(X, y) & 用concatMonirity 计算出各个模块的比例分布

    经过N轮迭代，生成目标数据.
    例如，有原始数据：
        缺陷个数 0     1     2     3+4     5
        模块个数 100   50    25    15      10

        经过一轮迭代，得：
        缺陷个数 0     1     2     3+4     5
        模块个数 100   50    25    15      20=10+10

        经过二轮迭代，得：
        缺陷个数 0     1     2     3+4           5
        模块个数 100   50    25    30=15+15      20=10+10

        ...
        经过5=j轮迭代后，得：
        缺陷个数 0     1     2           3+4              5
        模块个数 100   50    50=25+25    45=15+15+15      30=10+10+10
    """

    def __init__(self, X, y, ratio, k=5, r=0.1):
        """ 传入文件， 对应比例分布
            X: features
            y: defects [5,4,4,3,3,2,2,2,1,1,1,1,0,0,0,0,...]

            比例和y不对应，只为说明参数特征
            proportion: [('0', 80.0), ('1', 10.0), ('2', 5), ('3+4', 3.5), ('5', 1.5)]
        """
        self.X = X
        self.y = y
        self.ratio = ratio
        self._proportion = self.concat_()
        self.instanceSize, self.n_attrs = X.shape
        self.k = k
        self.r = r

        self._proportion_modified = None

    def proportion_(self, X, y):
        """ 计算各个模块的比例

        """
        statics = Statistics(y)
        max_bug = statics.max_()
        number_instance = statics.numberInstance()
        unique, counts = np.unique(y, return_counts=True)
        counter = dict(zip(unique, counts))
        res = dict()
        for i in range(max_bug + 1):
            if i >= 9:
                break
            if i in counter:
                res[str(i)] = np.around(counter[i] /
                                        number_instance * 100.0, decimals=4)
            else:
                res[str(i)] = 0
        if max_bug > 8:
            sum_ = 0
            for j in range(9, max_bug + 1):
                if j in counter:
                    sum_ += counter[j]
            # 令999代表大于8的模块
            res['999'] = np.around(sum_ / number_instance * 100.0, decimals=4)

        return res

    def concat_(self):
        """ 调用ConcatMinority 合并小于ratio的模块

        """
        proportion = self.proportion_(self.X, self.y)
        con = ConcatMinority(data=proportion, ratio=self.ratio)
        # after 是合并后的proportion
        after = con.concat()

        return after

    def getSynNumber(self, arg1=None):
        """ 根据self._proportion 计算经过全部迭代后需要生成多少数据
            如 0      1      2      3+4      5
               100    50     25     15       10

            经过5轮后
                0     1      2      3+4      5
    需要生成    0     0      25     30       20

        :arg1: TODO
        :returns: 需要生成的proportion

        """
        _proportion = sorted(
            self._proportion,
            key=lambda x: x[1],
            reverse=True)
        _proportion_dict = tuple2dict(_proportion)
        rounds = len(_proportion)
        # print('rounds :', rounds)
        for _ in range(rounds):
            tmp = _proportion[-1]
            change = (tmp[0], tmp[1] + _proportion_dict[tmp[0]])
            _proportion[-1] = change
            # print('change =', change)

            # 每次迭代后, 都需要根据数据个数重新进行排序
            _proportion = sorted(_proportion, key=lambda x: x[1], reverse=True)

        # 减去原本的样本，得到需要生成的样本个数
        for idx, item in enumerate(_proportion):
            change = (item[0], item[1] - _proportion_dict[item[0]])
            _proportion[idx] = change
        return _proportion

    def getUnsyn(self, arg1=None):
        """ 获得每个类别的数据，用这些数据进行合成
            类别类似于0 1 2 3+4 5..
            根据类别，将原数据集中的对应该类的所有数据都取出来

        :arg1: TODO
        :returns: 类别对应的trainingX, trainingy, 以及需要合成的个数synSize.
                  与self.over_sampling() 中的参数所对应

        """
        # 获取需要合成的样本数
        # 如[('0', 0.0), ('1', 0.0), ('2', 25), ('5', 30), ('3+4', 15)]
        _syn_proportion = self.getSynNumber()
        print('_syn_proportion =', _syn_proportion)

        # if not self._proportion_modified:
        #     self._proportion_modified = sorted(
        #         self._proportion, key=lambda x: x[1])
        # else:
        #     self._proportion_modified = sorted(
        #         self._proportion_modified, key=lambda x: x[1])

        instanceSize = self.instanceSize

        # # 获取需要合成的类
        # _class = self._proportion_modified[-1][0]

        all_trainingX = []
        all_trainingy = []
        # 需要合成的个数，size = p*self.index
        all_size = []
        for (_class, size) in _syn_proportion:
            if size == 0:
                continue
            if len(_class) == 1:
                # 如果没有经过合并
                _slice = np.where(self.y == int(_class))
                trainingX = self.X[_slice]
                trainingy = self.y[_slice]

                # 添加， 不执行over_sampling
                all_trainingX.append(trainingX)
                all_trainingy.append(trainingy)
                size_around = np.around(size * instanceSize / 100)
                all_size.append(size_around)
            else:
                _classes = _class.split('+')
                tmpX = [self.X[np.where(self.y == int(c))] for c in _classes]
                tmpy = [self.y[np.where(self.y == int(c))] for c in _classes]

                trainingX = arrayConcat(tmpX)
                trainingy = arrayConcat(tmpy)
                # for x, y in zip(tmpX, tmpy):
                #     trainingX = np.vstack((trainingX, x))
                #     trainingy = np.vstack((trainingy, y))

                all_trainingX.append(trainingX)
                all_trainingy.append(trainingy)
                size_around = np.around(size * instanceSize / 100)
                # print('size_around =', size_around)
                all_size.append(size_around)

        return all_trainingX, all_trainingy, all_size

    def synthesis(self, arg1=None):
        """ 遍历all_trainigX, all_trainingy
            以及每个类需要合成的样本个数
            调用self.over_sampling()

        :arg1: TODO
        :returns: none

        """

        all_trainingX, all_trainingy, all_size = self.getUnsyn()
        self.finalX, self.finaly = self.X, self.y
        for trainingX, trainingy, synSize in zip(
                all_trainingX, all_trainingy, all_size):
            self.over_sampling(trainingX, trainingy, synSize)

        return self.finalX, self.finaly

    def over_sampling(self, trainingX, trainingy, synSize):
        """ 执行一遍上采样
        :synSize: 需要合成的个数
        :returns: 新增加的数据

        """
        # 需要合成的数据为p * index, 设index=1
        self.index = 1
        p = synSize / self.index
        if len(trainingX) == 1:
            p = 1
        p = int(p)
        # trainingX =  # Todo
        # trainingy =  # Todo
        self.syntheticX = np.zeros((int(synSize), self.n_attrs))
        self.syntheticY = []

        self.count = p * self.index - 1
        for i in range(p):
            k = min(trainingX.shape[0], self.k)
            # k = self.k
            # 使用自定义的取邻近k个函数
            nnarray = self.nearestNeighbors(
                self.r, k, targetPoint=trainingX[int(i / p)], allPoints=trainingX)

            self._populate(trainingX, trainingy, int(i / p), nnarray, self.r)

        self.syntheticX = self.syntheticX[::-1]
        self.syntheticY = np.array(self.syntheticY)

        # print('生成的X:', self.syntheticX.shape)
        # print('生成的y:', self.syntheticY.shape)
        # print('之前的X:', self.finalX.shape)
        # print('之前的y:', self.finaly.shape)
        self.finalX = np.vstack((self.finalX, self.syntheticX))
        self.finaly = np.hstack((self.finaly, self.syntheticY))

    def _populate(self, trainingX, trainingy, i, nnarray, r):
        """ 从trainingX[i]的k个邻居中随机选取index次，生成index个合成的样本

        :trainingX: 所有的用于合成的数据
        :trainingy:
        :i: 与self.over_sampling for 中的i对应
        :nnarray: 距离目标点最近的数据
        :r:
        :returns: 生成的新数据

        """
        for j in range(self.index):
            nn = np.random.randint(0, self.k)
            nn = min(nn, len(nnarray) - 1)

            # print('\nnnarray =', nnarray)
            # print('nn =', nn)
            # print('i =', i)
            # print('trainingX.shape =', trainingX.shape)
            # print('trainingX:', trainingX)
            # print('trainingy:', trainingy)
            dif = trainingX[nnarray[nn]] - trainingX[i]
            gap = np.random.rand(1, self.n_attrs)
            self.syntheticX[self.count] = trainingX[i] + gap.flatten() * dif

            dist1 = (float)(
                (np.sum(abs(self.syntheticX[self.count] - trainingX[i])**r))**(1 / r))
            dist2 = (float)(
                (np.sum(abs(self.syntheticX[self.count] - trainingX[nnarray[nn]])**r))**(1 / r))

            if(dist1 + dist2 != 0):
                y = (dist1 * trainingy[nnarray[nn]] +
                     dist2 * trainingy[i]) * 1.0 / (dist1 + dist2)
                self.syntheticY.append(y)
            else:
                self.syntheticY.append(self.y[i] * 1.0)

            self.count -= 1

    def nearestNeighbors(self, r, k, targetPoint, allPoints):
        """获得距离目标点最近的k个点的标号
           r: float
           k: int
           targetPoint: np.array[float]
           allPoints: List[np.array[float]]
            res = List[np.array[float]]
        """
        candidate = []
        index = 1 / r
        targetPoint = np.asarray(targetPoint)
        allPoints = np.asarray(allPoints)
        for idx, point in enumerate(allPoints):
            subtraction = abs(point - targetPoint)
            result = np.sum(subtraction**r)
            candidate.append((result**index, idx))
        candidate = sorted(candidate, key=lambda x: x[0])
        res = [i[1] for i in candidate]

        return res[1:int(k + 1)]


def constructData(length):
    """ 生成fake数据

    :length: 生成数据的长度
    :returns: TODO

    """

    X = np.array([[1, 1]] * length)
    y = np.zeros(length)
    X[0:3], y[0:3] = [5, 5], 5
    X[3:5], y[3:5] = [4, 4], 4
    X[5:8], y[5:8] = [3, 3], 3
    X[8:15], y[8:15] = [2, 2], 2
    X[15:30], y[15:30] = [1, 1], 1
    # print(X, y)
    return X, y


def main():
    # X = np.array([[1, 1], [8, 8], [9, 9], [10, 10], [7, 9], [13, 13], [1, 2], [8, 2], [9, 2], [9, 2], [7, 2], [7, 2],
    #               [3, 4], [4, 3], [6, 2], [7, 3], [3, 5], [4, 5], [6, 5], [7, 5], [2, 1], [1, 3], [1, 2], [4, 1], [1, 6], [3, 4], [4, 3], [6, 2], [7, 3], [3, 5], [4, 5], ])
    # y = np.array([2, 3, 5, 1, 3, 1, 4, 3, 0, 0, 0, 0,
    #               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ])
    X, y = constructData(100)
    proportion = [('0', 70.0), ('1', 15.0), ('2', 7), ('3+4', 5), ('5', 3)]
    # proportion = [('0', 100.0), ('1', 50.0), ('2', 25), ('3+4', 15), ('5', 10)]

    ssmote = Ssmote(X=X, y=y, ratio=4)
    # synProportion = con.getSynNumber()
    # print(synProportion)
    all_X, all_y, all_size = ssmote.getUnsyn()
    print(all_X)
    print(all_y)
    print(all_size)
    print('-' * 20, '下面开始合成', '-' * 20)

    resX, resy = ssmote.synthesis()
    print(resX, resy)


if __name__ == '__main__':
    main()
