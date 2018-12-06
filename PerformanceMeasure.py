# -*- coding: utf-8 -*-
# @Author: Kicc
# @Date:   2018-11-23 13:29:21
# @Last Modified by:   Kicc
# @Last Modified time: 2018-11-23 13:29:49

import numpy as np

class PerformanceMeasure():

    def __init__(self, real_list, pred_list):
        self.real = real_list
        self.pred = pred_list
        self.aae_value  = []
        self.fpa_value=0

    def AAE(self):
        '''
        求每一类模块上的平均绝对误差（average absolute error）
        real_list指测试集中每个模块的真实缺陷个数
        pred_list指训练出的回归模型对测试集中每个模块进行预测得出的预测值
        如real_list=[2,3,0,0,1,1,0,5,3]
         pred_list=[1,1,1,0,1,0,0,3,4]
         输出结果就为0:0.33, 1:0.5,  2:1,  3:1.5,  5:2
        '''
        only_r = np.array(list(set(self.real)))
        # only_r=[0,1,2,3,5]

        for i in only_r:
            r_index = np.where(self.real == i)
            # i=0的时候，r_index=(array([2, 3, 6]), ) 得到是一个tuple

            sum = 0

            # i=0的时候，k = [2, 3, 6]
            k = r_index[0]
            sum = abs(self.real[k] - self.pred[k]).sum()
            avg = sum * 1.0 / len(k)
            self.aae_value.append(avg)

        # 直接返回字典
        aae_result = dict(zip(only_r, self.aae_value))
        return aae_result

    def FPA(self):
        '''
        有四个模块m1,m2,m3,m4，真实缺陷个数分别为1，4，2，1,self.real=[1，4，2，1]
        预测出m1缺陷个数为0，m2缺陷个数为3，m3缺陷个数为5，m4缺陷个数为1,self.pred=[0,3,5,1]
        预测出的排序为m3>m2>m4>m1
        fpa=1/4 *1/8 *(4*2+3*4+2*1+1*1)=0.718
        '''
        K = len(self.real)
        N = np.sum(self.real)
        sort_axis = np.argsort(self.pred)
        testBug = np.array(self.real)
        testBug = testBug[sort_axis]
        P = sum(np.sum(testBug[m:]) / N for m in range(K + 1)) / K
        return P

''''
if __name__ == '__main__':
    real=np.array([2,3,0,0,1,1,0,5,3])
    pred=np.array([1,5,1,0,1,0,0,7,2])
    aeeresult=PerformanceMeasure(real,pred).AEE()
    print (aeeresult)
    real=np.array([1,4,2,1])
    pred=np.array([0,3,5,1])
    print (PerformanceMeasure(real,pred).FPA())
'''