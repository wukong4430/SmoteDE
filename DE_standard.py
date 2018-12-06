# -*- coding: utf-8 -*-
# @Author: Kicc
# @Date:   2018-11-24 20:51:54
# @Last Modified by:   kicc
# @Last Modified time: 2018-12-05 15:15:43


import numpy as np
import matplotlib.pyplot as plt
import math
import random
from sklearn.linear_model import BayesianRidge
from modifySmote import Smote

"""
所有的基因都是实数，且范围一致

"""


class DE:
    def __init__(self, NP=100, F=0.6, CR=0.7, generation=2000, len_x=10, value_up_range=-5.12, value_down_range=5.12, X=None, y=None):
        self.NP = NP   # 种群数量
        self.F = F   # 缩放因子
        self.CR = CR   # 交叉概率
        self.generation = generation   # 遗传代数
        self.len_x = len_x
        self.value_up_range = value_up_range
        self.value_down_range = value_down_range

        self.np_list = self.initialtion()

        self.trainX, self.trainy, self.validX, self.validy, self.testX, self.testy = dataProcess(
            X=X, y=y).generateData()

    def initialtion(self):
        # 种群初始化

        np_list = []   # 种群，染色体
        for i in range(0, self.NP):
            x_list = []   # 个体，基因
            for j in range(0, self.len_x):
                x_list.append(self.value_down_range + random.random() *
                              (self.value_up_range - self.value_down_range))
            np_list.append(x_list)
        return np_list

    def substract(self, a_list, b_list):
        # 列表相减
        return [a - b for (a, b) in zip(a_list, b_list)]

    def add(self, a_list, b_list):
        # 列表相加
        return [a - b for (a, b) in zip(a_list, b_list)]

    def multiply(self, a, b_list):
        # 列表的数乘
        return [a * b for b in b_list]

    def mutation(self, np_list):
        """
        # 变异
        # 保证取出来的i,r1,r2,r3互不相等
        返回中间的变异种群
        """

        v_list = []
        for i in range(0, self.NP):
            r1 = random.randint(0, self.NP - 1)
            while r1 == i:
                r1 = random.randint(0, self.NP - 1)
            r2 = random.randint(0, self.NP - 1)
            while r2 == r1 | r2 == i:
                r2 = random.randint(0, self.NP - 1)
            r3 = random.randint(0, self.NP - 1)
            while r3 == r2 | r3 == r1 | r3 == i:
                r3 = random.randint(0, self.NP - 1)

            v_list.append(self.add(np_list[r1], self.multiply(
                self.F, self.substract(np_list[r2], np_list[r3]))))
        return v_list

    def crossover(self, np_list, v_list):
        """
        np_list: 第g代初始种群
        v_list: 变异后的中间体
        """
        u_list = []
        for i in range(0, self.NP):
            vv_list = []
            for j in range(0, self.len_x):  # len_x 是基因个数
                if (random.random() <= self.CR) or (j == random.randint(0, self.len_x - 1)):
                    vv_list.append(v_list[i][j])
                else:
                    vv_list.append(np_list[i][j])
            # 保证每个染色体至少有一个基因遗传给下一代，强制取出一个变异中间体的基因
            tmp = random.randint(0, self.len_x - 1)
            vv_list[tmp] = v_list[i][tmp]
            u_list.append(vv_list)
        return u_list

    def selection(self, u_list, np_list):
        """根据适应度函数，从初始化种群 或者 交叉后种群中选择

        """
        for i in range(0, self.NP):
            if self.object_function(u_list[i]) <= self.object_function(np_list[i]):
                np_list[i] = u_list[i]
            else:
                np_list[i] = np_list[i]
        return np_list

    def process(self):
        np_list = self.np_list
        min_x = []
        min_f = []
        for i in range(0, self.NP):
            xx = []
            xx.append(self.object_function(np_list[i]))
        # 将初始化的种群对应的min_f和min_xx加入
        min_f.append(min(xx))
        min_x.append(np_list[xx.index(min(xx))])

        # 迭代循环
        for i in range(0, self.generation):
            v_list = self.mutation(np_list)  # 变异
            u_list = self.crossover(np_list, v_list)  # 杂交
            # 选择， 选择完之后的种群就是下一个迭代开始的种群
            np_list = self.selection(u_list, np_list)
            for i in range(0, self.NP):
                xx = []
                xx.append(self.object_function(np_list[i]))
            min_f.append(min(xx))
            min_x.append(np_list[xx.index(min(xx))])

        # 输出
        min_ff = min(min_f)
        # 用min_f.index()根据最小值min_ff找对应的染色体，说明不一定最后的染色体是最好的
        min_xx = min_x[min_f.index(min_ff)]
        print('the minimum point x =', min_xx)
        print('the minimum value y =', min_ff)

        # 画图
        x_label = np.arange(0, self.generation + 1, 1)
        plt.plot(x_label, min_f, color='blue')
        plt.xlabel('iteration')
        plt.ylabel('fx')
        plt.savefig('./iteration-f.png')
        plt.show()

    def object_function(self, x):
        """
        适应度函数注册
        x是一个list
        x^2 - (10*cos(2*pi*x)+10)
        """
        f = 0
        for i in range(0, len(x)):
            f = f + (x[i] ** 2 - (10 * math.cos(2 * np.pi * x[i])) + 10)
        return f

    def smoteObj(self, smoteParam):
        """传入所有参数计算一个fpa值
        smoteParam: [ratio, k, r]
        param: ratio: smote的比例
               k: 最邻近个数
               p: minkowski 指标

        """
        def getFPA(bbr, smote_X, smote_y, validX, validy):
            brr = brr.fit(smote_X, smote_y)
            brr_pred_y = np.around(brr.predict(validX))
            brr_fpa = PerformanceMeasure(validy, brr_pred_y).FPA()
            return brr_fpa

        ratio = smoteParam[0]
        k = smoteParam[1]
        r = smoteParam[2]
        smote_X, smote_y = Smote(
            X=self.trainX, Y=self.trainy, ratio=ratio, k=k, r=r).over_sampling()

        bbr = BayesianRidge()

        # get the FPA with bbr model.
        fpa = getFPA(bbr=brr, smote_X=smote_X, smote_y=smote_y,
                     validX=self.validX, validy=self.validy)

        return fpa


class dataProcess:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def generateData(self):
        """将数据集分成5份, 3份训练集， 1份测试集， 1份验证集

        """
        if self.X == None:
            return None, None, None, None, None, None

        return trainX, trainy, validX, validy, testX, testy


if __name__ == '__main__':
    # 初始化
    # NP, F, CR, generation, len_x, value_up_range, value_down_range = initpara()
    # np_list = initialtion(NP)
    # main(np_list)
    de = DE()
    de.process()
