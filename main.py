# -*- coding: utf-8 -*-
# @Author: Kicc
# @Date:   2018-12-06 13:00:17
# @Last Modified by:   Kicc
# @Last Modified time: 2018-12-06 13:18:59

import numpy as np
from DE2 import DE
# from DE_standard import DE
from Processing import Processing
from sklearn.linear_model import BayesianRidge


def bootstrap(dataset):

    training_data_X, training_data_y, testing_data_X, testing_data_y = Processing(
    ).separate_data(dataset)

    brr = BayesianRidge()

    de = DE(NP=100,
            F=0.6,
            CR=0.7,
            generation=2000,
            len_x=3,
            ratioRange=[0.5, 1.0, 2.0, 4.0],
            kRange=list(range(1, 21)),
            value_up_range=5.0,
            value_down_range=0.1,
            X=training_data_X,
            y=training_data_y,
            classifier=brr)

    de.process()


if __name__ == '__main__':
    for dataset, filename in Processing().import_single_data():
        bootstrap(dataset=dataset)
        break
