# -*- coding: utf-8 -*-
import numpy as np

from Processing import Processing

from PerformanceMeasure import PerformanceMeasure
from DE_three_mutations import DE_three_mutations

"""
参照论文A Learning-to-Rank Approach to Software Defect Prediction
学习的模型为y=a0*x0+a1*x1+a2*x2+....+a19*x19,其中x0-x19为软件模块的20维的特征，
总共20个基因, a0,a1,a2,,,.a19.
要求出这个学习的模型，其实就是要求出a0,a1,a2,,,.a19。
然后将测试集软件模块的20维特征向量带入y=a0*x0+a1*x1+a2*x2+....+a19*x19
可求得测试集软件模块的预测缺陷个数。然后利用预测缺陷个数和真实缺陷个数对比

"""


class LTR:
    def __init__(self):
        self.W = None

    def fit(self, training_data_X, training_data_y):
        """拟合，求出最佳的W

        """
        self.training_data_X = training_data_X
        self.training_data_y = training_data_y
        de = DE_three_mutations(X=training_data_X, y=training_data_y)
        self.W = de.process(self.objfunc)
        return self

    def objfunc(self, Param):
        """适应度函数

        """
        pred_y = []
        for train_x in self.training_data_X:
            pred_y.append(float(np.dot(train_x, Param)))
        fpa = PerformanceMeasure(self.training_data_y, pred_y).FPA()

        return fpa

    def predict(self, testing_data_X):
        pred_y = []
        for test_x in testing_data_X:
            pred_y.append(float(np.dot(test_x, self.W)))
        return pred_y


if __name__ == '__main__':
    # 初始化
    # NP, F, CR, generation, len_x, value_up_range, value_down_range = initpara()
    # np_list = initialtion(NP)
    # main(np_list)
    for dataset, filename in Processing().import_single_data():
        training_data_X, training_data_y, testing_data_X, testing_data_y = Processing(
        ).separate_data(dataset)

        ltr = LTR().fit(training_data_X, training_data_y)
        ltr_pred_y = ltr.predict(testing_data_X)
        ltr_fpa = PerformanceMeasure(testing_data_y, ltr_pred_y).FPA()

        print("FPA = ", ltr_fpa)
