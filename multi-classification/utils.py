#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 kicc <wukong4430@gmail.com>
#           Create time: 2019-01-17 14:05
#
# Distributed under terms of the MIT license.

import numpy as np
"""
基础函数
"""

def arrayConcat(array):
    """ 将[array([[8,8], [7,9]]), array([[10,10], [13, 13]])]
        改成 array([[8,8], [7,9], [10, 10], [13, 13]])
    :returns: TODO

    """
    listX = [list(x) for x in array]
    res = []
    for l in listX:
        res.extend(l)
    res = np.asarray(res)
    return res

def tuple2dict(list_of_tuple):
    """ convert List[tuple] -> dict

    :list_of_tuple: List[tuple]
    :returns: dict

    """
    res = dict()
    for t in list_of_tuple:
        res[t[0]] = t[1]
    return res


if __name__ == '__main__':
    _proportion = [('0', 80.0), ('1', 10.0), ('2', 5), ('3+4', 3.5), ('5', 1.5)]
    res = tuple2dict(_proportion)
    print(res)
    # main()
    tmpX = [np.array([[8,8], [7,9]]), np.array([[10,10], [13, 13]])]
    res = arrayConcat(tmpX)
    print(res)
