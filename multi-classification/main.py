#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 kicc <wukong4430@gmail.com>
#           Create time: 2019-01-18 12:45
#
# Distributed under terms of the MIT license.

"""
获取csv数据，调用ssmote合成数据
"""
from Processing import Processing, convert2numpy
from statistics import proportion_
from concatModule import ConcatMinority
from Ssmote import Ssmote

def main():
    for dataset, filename in Processing().dataload():
        print(filename)
        # 计算各模块的比例
        proportion = proportion_(dataset, filename)
        print(proportion)

        # 合并小于4%的模块
        con = ConcatMinority(data=proportion, ratio=4)
        # after是合并后的proportion
        after = con.concat()
        print(after, '\n')

        X, y = convert2numpy(dataset)
        print('X shape', X.shape)
        print('传入的proportion：', after)
        # ssmote合成新数据
        ssmote = Ssmote(X=X, y=y, proportion=after)
        synX, synY = ssmote.synthesis()
        print(synX, synY)
        print('-'*30, '一个文件完成啦！', '-'*30)
    print('o')


if __name__ == '__main__':
    main()
