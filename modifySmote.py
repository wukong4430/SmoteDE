# -*- coding: utf-8 -*-
# @Author: Kicc
# @Date:   2018-11-23 12:31:01
# @Last Modified by:   Kicc
# @Last Modified time: 2018-12-06 14:42:05

from sklearn.neighbors import NearestNeighbors
import numpy as np


class Smote:
    def __init__(self, X, Y, ratio=0.5, k=5, r=0.1):
        """ ratio: [0.5, 1.0, 2.0, 4.0]
            k: [1-20], int
            r: [0.1, 5], float
        """
        # self.n_samples includes defective and non-defective data.
        self.instancesize, self.n_attrs = X.shape
        self.X = X
        self.Y = Y
        # self.ratio is the desired ratio between the rare instances and the normal instances
        self.ratio = ratio
        self.k = k
        self.r = r

    # 这个函数只返回合成的数据
    def over_sampling(self):

        # 获取原始数据集中normal instances和rare instances的个数
        normalinstancesize, rareinstanceX, rareinstanceY = self.refreshData(
            self.X, self.Y)
        rareinstancesize = self.instancesize - normalinstancesize

        if self.ratio < 2 * (rareinstancesize / (self.instancesize - rareinstancesize)):
            p = round(self.ratio * (self.instancesize -
                                    rareinstancesize) - rareinstancesize)

            # rareinstanceX, rareinstanceY 中抽取p个数据，即为用于合成新rare instances的traininginstances
            keep = np.random.permutation(rareinstancesize)[:p]
            traininginstancesX = rareinstanceX[keep]
            traininginstancesY = rareinstanceY[keep]
            self.index = 1

        else:
            p = rareinstancesize
            self.index = int((self.ratio * (self.instancesize -
                                            rareinstancesize) - rareinstancesize) / rareinstancesize)
            traininginstancesX = rareinstanceX
            traininginstancesY = rareinstanceY
        '''
        print('number of normalinstances :', normalinstancesize)
        print('number of rareinstances :', rareinstancesize)
        print('p :', p)
        print('index :', self.index)
        print('总共需要合成的rare instances为:', p * self.index)
        print('从原始数据中选择的参与合成的rare instances为：', traininginstancesX)
        '''
        # 总共需要合成的rare instances为p*index个
        self.syntheticX = np.zeros((p * self.index, self.n_attrs))
        self.syntheticY = []

        # 因为在knn算法中，取instance[i]的k个近邻会包含instance[i]，所以取k+1个，然后去掉instance[i]
        neighbors = NearestNeighbors(
            n_neighbors=self.k + 1).fit(traininginstancesX)
        # print('从原始数据中选择的参与合成的rare instances为：', traininginstancesX)

        self.count = p * self.index - 1
        for i in range(p):
            # 存储traininginstances[i]的k+1个近邻的下标
            # knnarray = neighbors.kneighbors(
            #     traininginstancesX[i].reshape(1, -1), return_distance=False)[0]
            # nnarray = knnarray[1:]
            # print(nnarray.shape)
            # print('nnarray=', nnarray)
            # print('knnarray=', knnarray)

            # 使用自定义的取邻近k个函数
            nnarray = self.nearestNeighbors(
                self.r, self.k, targetPoint=traininginstancesX[i], allPoints=traininginstancesX)
            # print(nnarray)

            # print('\n选择第', i, '个traininginstanceX为',
            #       traininginstancesX[i], ',它的近邻为:')
            # print('knnarray', knnarray, 'nnarray:', nnarray)
            # for j in nnarray:
            #     print('第', j, '个', traininginstancesX[j])

            self.__populate(traininginstancesX, traininginstancesY, i, nnarray)

        return self.syntheticX, self.syntheticY

        self.syntheticX = self.syntheticX[::-1]
        self.syntheticY = np.array(self.syntheticY)
        return self.syntheticX, self.syntheticY

    # 这个函数返回合成的数据加原始数据
    def over_sampling_addorginaldata(self):

        # 获取原始数据集中normal instances和rare instances的个数
        normalinstancesize, rareinstanceX, rareinstanceY = self.refreshData(
            self.X, self.Y)
        rareinstancesize = self.instancesize - normalinstancesize

        if self.ratio < 2 * (rareinstancesize / (self.instancesize - rareinstancesize)):
            p = round(self.ratio * (self.instancesize -
                                    rareinstancesize) - rareinstancesize)

            # rareinstanceX, rareinstanceY 中抽取p个数据，即为用于合成新rare instances的traininginstances
            keep = np.random.permutation(rareinstancesize)[:p]
            traininginstancesX = rareinstanceX[keep]
            traininginstancesY = rareinstanceY[keep]
            self.index = 1

        else:
            p = rareinstancesize
            self.index = int((self.ratio * (self.instancesize -
                                            rareinstancesize) - rareinstancesize) / rareinstancesize)
            traininginstancesX = rareinstanceX
            traininginstancesY = rareinstanceY
        '''
        print('number of normalinstances :', normalinstancesize)
        print('number of rareinstances :', rareinstancesize)
        print('p :', p)
        print('index :', self.index)
        print('总共需要合成的rare instances为:', p * self.index)
        print('从原始数据中选择的参与合成的rare instances为：', traininginstancesX)
        '''
        # 总共需要合成的rare instances为p*index个
        self.syntheticX = np.zeros((p * self.index, self.n_attrs))
        self.syntheticY = []
        # 因为在knn算法中，取instance[i]的k个近邻会包含instance[i]，所以取k+1个，然后去掉instance[i]
        # minkowski 距离
        # neighbors = NearestNeighbors(
        #     n_neighbors=self.k + 1, p=self.p).fit(traininginstancesX)

        self.count = p * self.index - 1
        for i in range(p):
            # 存储traininginstances[i]的k+1个近邻的下标
            # knnarray = neighbors.kneighbors(
            #     traininginstancesX[i].reshape(1, -1), return_distance=False)[0]
            # nnarray = knnarray[1:]

            nnarray = self.nearestNeighbors(
                self.r, self.k, targetPoint=traininginstancesX[i], allPoints=traininginstancesX)

            '''
            print('\n选择第', i, '个traininginstanceX为',
                  traininginstancesX[i], ',它的近邻为:')
            print('knnarray', knnarray, 'nnarray:', nnarray)
            for j in nnarray:
                print('第', j, '个', traininginstancesX[j])
            '''

            self.__populate(traininginstancesX, traininginstancesY, i, nnarray)

        return self.syntheticX, self.syntheticY

        self.syntheticX = self.syntheticX[::-1]
        self.syntheticY = np.array(self.syntheticY)

        syntheticY = np.array(self.syntheticY)
        Y = np.hstack((self.Y, syntheticY))
        X = np.vstack((self.X, self.syntheticX))
        return X, Y

    # 从traininginstancesX[i]的k个邻居中随机选取index次，生成index个合成的样本
    def __populate(self, traininginstancesX, traininginstancesY, i, nnarray):
        for j in range(self.index):
            nn = np.random.randint(0, self.k)
            nn = min(nn, len(nnarray) - 1)
            # print('nnarray =', nnarray)
            # print('self.k={0}, nn={1}'.format(self.k, nn))
            # print('nnarray[nn] =', nnarray[nn])  # index out of range
            # print('traininginstancesX[nnarray[nn]] =',
            # traininginstancesX[nnarray[nn]])
            dif = traininginstancesX[nnarray[nn]] - traininginstancesX[i]
            gap = np.random.rand(1, self.n_attrs)
            self.syntheticX[self.count] = traininginstancesX[i] + \
                gap.flatten() * dif

            '''
            print(
                '\n选择用于合成的', traininginstancesX[i], '一个近邻为的traininginstanceX为', traininginstancesX[nnarray[nn]])
            print(traininginstancesX[nnarray[nn]], '与',
                  traininginstancesX[i], '之间的dif :', dif)
            print('产生的随机向量为:', gap.flatten())
            print('合成的instance的向量为', self.syntheticX[self.count])
            '''

            dist1 = (float)(np.linalg.norm(
                self.syntheticX[self.count] - traininginstancesX[i]))
            dist2 = (float)(np.linalg.norm(
                self.syntheticX[self.count] - traininginstancesX[nnarray[nn]]))

            if (dist1 + dist2 != 0):
                self.syntheticY.append(
                    (dist1 * traininginstancesY[nnarray[nn]] + dist2 * traininginstancesY[i]) * 1.0 / (dist1 + dist2))
            else:
                self.syntheticY.append(self.Y[i] * 1.0)

            '''
            print(traininginstancesX[i], traininginstancesY[i],
                  '和合成数据', self.syntheticX[self.count], '之间距离为', dist1)
            print(traininginstancesX[nnarray[nn]], traininginstancesY[nnarray[nn]],
                  '和合成数据', self.syntheticX[self.count], '之间距离为', dist2)
            print('合成的数据为：', self.syntheticX[self.count], self.syntheticY[i])
            '''
            self.count -= 1

    def refreshData(self, dataX, dataY):
        '''
        dataX: 原始数据集的X
        dataY: 原始数据集的y
        返回缺陷数目为0的instances个数，返回缺陷数目大于0的instance的dataX 和 datay
        '''
        bugDataX = []
        bugDataY = []
        count = 0
        dataY = np.matrix(dataY).T
        dataX = np.array(dataX)
        for i in range(len(dataY)):
            if dataY[i] > 0:
                bugDataX.append(dataX[i])
                bugDataY.append(int(dataY[i]))
            else:
                count += 1

        bugDataX = np.array(bugDataX)
        bugDataY = np.array(bugDataY)
        return count, bugDataX, bugDataY

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
            subtraction = point - targetPoint
            result = np.sum(subtraction**r)
            candidate.append((result**index, idx))
        candidate = sorted(candidate, key=lambda x: x[0])
        res = [i[1] for i in candidate]

        return res[1:int(k + 1)]


def main():
    X = np.array([[1, 1], [8, 8], [9, 9], [9, 7], [7, 9], [7, 7], [1, 2], [8, 2], [9, 2], [9, 2], [7, 2], [7, 2],
                  [3, 4], [4, 3], [6, 2], [7, 3], [3, 5], [4, 5], [6, 5], [7, 5], [2, 1], [1, 3], [1, 2], [4, 1], [1, 6], [3, 4], [4, 3], [6, 2], [7, 3], [3, 5], [4, 5], ])

    y = np.array([2, 3, 5, 1, 3, 1, 0, 0, 0, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ])

    smote_X, smote_y = Smote(X=X, Y=y, ratio=1.0, k=2, r=2).over_sampling()

    print('smote_X :', smote_X)
    print('smote_y :', smote_y)


if __name__ == '__main__':
    main()
