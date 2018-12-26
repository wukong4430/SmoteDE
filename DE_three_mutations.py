# -*- coding: utf-8 -*-
# @Author: Kicc
# @Date:   2018-12-12 12:35:46
# @Last Modified by:   Kicc
# @Last Modified time: 2018-12-26 12:54:10

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from modifySmote import Smote
from sklearn.model_selection import train_test_split
from PerformanceMeasure import PerformanceMeasure
from Processing import Processing

"""
"""


class DE_three_mutations:
    def __init__(self,
                 NP=100,
                 F_CR=[(1.0, 0.1), (1.0, 0.9), (0.8, 0.2)],
                 generation=10,
                 len_x=20,
                 value_up_range=20.0,
                 value_down_range=-20.0,
                 X=None,
                 y=None):

        self.NP = NP   # 种群数量
        self.F_CR = F_CR   # 缩放因子
        self.generation = generation   # 遗传代数
        self.len_x = len_x
        self.value_up_range = value_up_range
        self.value_down_range = value_down_range

        self.np_list = self.initialtion()
        self.training_data_X = X
        self.training_data_y = y

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
        return [a + b for (a, b) in zip(a_list, b_list)]

    def multiply(self, a, b_list):
        # 列表的数乘
        return [a * b for b in b_list]

    def random_distinct_integers(self, number):
        """ 从range(0, self.NP-1) 中选出number个不相同的整数。

        """
        res = set()
        while len(res) != int(number):
            res.add(random.randint(0, self.NP - 1))

        return list(res)

    def mutation_crossover_one(self, np_list):
        """变异+交叉算法 第一种


        """
        F_CR = random.choice(self.F_CR)
        F = F_CR[0]
        CR = F_CR[1]

        v_list = []
        for i in range(0, self.NP):
            r123 = self.random_distinct_integers(3)
            r1 = r123[0]
            r2 = r123[1]
            r3 = r123[2]

            sub = self.substract(np_list[r2], np_list[r3])
            mul = self.multiply(F, sub)
            add = self.add(np_list[r1], mul)
            v_list.append(add)

        # crossover
        u_list = []
        for i in range(0, self.NP):
            vv_list = []
            for j in range(0, self.len_x):  # len_x 是基因个数
                if (random.random() <= CR) or (j == random.randint(0, self.len_x - 1)):
                    vv_list.append(v_list[i][j])
                else:
                    vv_list.append(np_list[i][j])
            # 保证每个染色体至少有一个基因遗传给下一代，强制取出一个变异中间体的基因
            tmp = random.randint(0, self.len_x - 1)
            vv_list[tmp] = v_list[i][tmp]
            u_list.append(vv_list)
        return u_list

    def mutation_crossover_two(self, np_list):
        """ 变异+交叉算法 第一种


        """
        F_CR = random.choice(self.F_CR)
        F = F_CR[0]
        CR = F_CR[1]
        F1 = random.random()

        v_list = []
        for i in range(0, self.NP):
            r12345 = self.random_distinct_integers(5)
            r1 = r12345[0]
            r2 = r12345[1]
            r3 = r12345[2]
            r4 = r12345[3]
            r5 = r12345[4]

            sub1 = self.substract(np_list[r2], np_list[r3])
            sub2 = self.substract(np_list[r4], np_list[r5])
            mul1 = self.multiply(F1, sub1)
            mul2 = self.multiply(F, sub2)
            add1 = self.add(np_list[r1], mul1)
            add2 = self.add(add1, mul2)
            v_list.append(add2)

        u_list = self.crossover(np_list, v_list, CR)
        return u_list

    def mutation_crossover_three(self, np_list):
        """ 变异+交叉算法 第一种


        """
        F_CR = random.choice(self.F_CR)
        F = F_CR[0]

        v_list = []
        for i in range(0, self.NP):
            r123 = self.random_distinct_integers(3)
            r1 = r123[0]
            r2 = r123[1]
            r3 = r123[2]
            sub1 = self.substract(np_list[r2], np_list[r3])
            sub2 = self.substract(np_list[r1], np_list[i])
            mul1 = self.multiply(F, sub1)
            mul2 = self.multiply(random.random(), sub2)
            add1 = self.add(mul1, mul2)
            add2 = self.add(add1, np_list[i])
            v_list.append(add2)

        return v_list

    def crossover(self, np_list, v_list, CR):
        """
        np_list: 第g代初始种群
        v_list: 变异后的中间体
        """
        u_list = []
        for i in range(0, self.NP):
            vv_list = []
            for j in range(0, self.len_x):  # len_x 是基因个数
                if (random.random() <= CR) or (j == random.randint(0, self.len_x - 1)):
                    vv_list.append(v_list[i][j])
                else:
                    vv_list.append(np_list[i][j])
            # 保证每个染色体至少有一个基因遗传给下一代，强制取出一个变异中间体的基因
            tmp = random.randint(0, self.len_x - 1)
            vv_list[tmp] = v_list[i][tmp]
            u_list.append(vv_list)
        return u_list

    def selection(self, u_list1, u_list2, u_list3, np_list):
        """根据适应度函数，从初始化种群 或者 交叉后种群中选择

        """
        for i in range(0, self.NP):
            fpa1 = self.Objfunction(u_list1[i])
            fpa2 = self.Objfunction(u_list2[i])
            fpa3 = self.Objfunction(u_list3[i])
            fpa4 = self.Objfunction(np_list[i])
            max_fpa = max(fpa1, fpa2, fpa3, fpa4)
            if max_fpa == fpa1:
                np_list[i] = u_list1[i]
            elif max_fpa == fpa2:
                np_list[i] = u_list2[i]
            elif max_fpa == fpa3:
                np_list[i] = u_list3[i]
            else:
                np_list[i] = np_list[i]
        return np_list

    def process(self, objfunc):
        np_list = self.np_list
        max_x = []  # 保存每次迭代的最佳染色体
        max_f = []  # 保存每次迭代的最佳染色体对应的标量
        for i in range(0, self.NP):
            xx = []
            xx.append(objfunc(np_list[i]))
        # 将初始化的种群对应的max_f和max_xx加入
        max_f.append(max(xx))
        max_x.append(np_list[xx.index(max(xx))])

        # 迭代循环
        for i in range(0, self.generation):
            print("iteration {0}".format(i))
            u_list1 = self.mutation_crossover_one(np_list)  # 变异+交叉
            u_list2 = self.mutation_crossover_two(np_list)  # 变异+交叉
            u_list3 = self.mutation_crossover_three(np_list)  # 变异+交叉
            # 选择， 选择完之后的种群就是下一个迭代开始的种群
            np_list = self.selection(u_list1, u_list2, u_list3, np_list)
            for i in range(0, self.NP):
                xx = []
                xx.append(objfunc(np_list[i]))
            max_f.append(max(xx))
            max_x.append(np_list[xx.index(max(xx))])

        # 输出
        max_ff = max(max_f)
        # 用max_f.index()根据最小值max_ff找对应的染色体，说明不一定最后的染色体是最好的
        max_xx = max_x[max_f.index(max_ff)]
        print('the maximum point x =', max_xx)
        print('the maximum value y =', max_ff)

        # 画图
        x_label = np.arange(0, self.generation + 1, 1)
        plt.plot(x_label, max_f, color='blue')
        plt.xlabel('iteration')
        plt.ylabel('fx')
        plt.savefig('./iteration-f.png')
        plt.show()

        return max_xx

    def Objfunction(self, Param):
        """
        传入所有参数计算一个fpa值
        """
        pred_y = []
        for test_x in self.training_data_X:
            pred_y.append(float(np.dot(test_x, Param)))
        fpa = PerformanceMeasure(self.training_data_y, pred_y).FPA()

        return fpa


if __name__ == '__main__':
    # 初始化
    # NP, F, CR, generation, len_x, value_up_range, value_down_range = initpara()
    # np_list = initialtion(NP)
    # main(np_list)
    for dataset, filename in Processing().import_single_data():
        training_data_X, training_data_y, testing_data_X, testing_data_y = Processing(
        ).separate_data(dataset)
        de = DE_three_mutations(X=training_data_X,
                                y=training_data_y)
        de.process(de.Objfunction)
