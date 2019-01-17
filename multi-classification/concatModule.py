#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 kicc <wukong4430@gmail.com>
#           Create time: 2019-01-16 17:26
#
# Distributed under terms of the MIT license.

"""
按照大于3%的比例合并小于3%的类
如
缺陷个数0: 80%，
缺陷个数1: 10%，
缺陷个数2: 5%，
缺陷个数3: 2%,
缺陷个数4：1.5%
缺陷个数5：1.5%

将缺陷个数3和4的合并起来(2%+1.5%>3%)，最终缺陷个数5的自成一类

"""


class ConcatMinority(object):

    """合并少数项"""

    def __init__(self, data, ratio):
        """ 初始化原本的数据

        :data: {'0':x0, '1':x1, '2':x2, '3':x3, '4':x4,...}
        sigma(x1~xn)=1

        """
        self._data = self.dict2tuple(data)
        self._ratio = ratio

    def dict2tuple(self, data):
        t = list()
        for key, value in data.items():
            t.append((key, value))
        return t

    def concat(self):
        """ 进行合并
        :returns: {'0': x0, '1': x1, '2': x2, '3+4':x3+x4,...}

        """
        after = []
        j = 0
        for idx, item in enumerate(self._data):
            # 跳过为0的模块
            if item[1] == 0:
                continue

            # j用来跳过合并的部分
            if idx < j:
                continue

            # print(item)
            if item[1] >= self._ratio:
                after.append(item)
                continue

            # 需要合并的部分
            else:
                # 用于保存新的数据
                concat_key = item[0]
                sum_ = item[1]
                j = idx + 1
                while j < len(self._data) and sum_ < self._ratio:
                    sum_ += self._data[j][1]
                    concat_key += ('+' + self._data[j][0])
                    j += 1
                after.append((concat_key, sum_))

        return after


def main():
    d = {'0': 80.0, '1': 10.0, '2': 5, '3': 2.0, '4': 1.5, '5': 1.5}
    con = ConcatMinority(data=d, ratio=3)
    after = con.concat()
    print(after)


if __name__ == '__main__':
    main()
