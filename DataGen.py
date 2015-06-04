# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 21:21:33 2015

@author: leftword
"""

import numpy as np

class DataGenerator:
    data_dim = 0
    data_size = 0
    def __init__(self):
        pass

    def get_default_data_1(self):
        self.data_dim = 2
        data = np.array([[0,0,1], [2,2,1], [-1,-1,1], [2,-3,1], [3,-3,1], [1,-1,-1],[1,-2,-1], [3,3,-1], [3,0,-1],[4,0,-1]])
        self.data_size = len(data)
        return data

    def get_default_data_2(self):
        self.data_dim = 2
        data = np.array([[0.5,-2.5,1], [3,1,1], [-1.5,-1,-1], [1,3.5,-1]])
        self.data_size = len(data)
        return data

    def get_default_data_3(self):
        self.data_dim = 2
        data = np.array([[0,1,1], [-1,0,1], [-1.1,-2,1], [-2,2,1], [-1.5,4,1], [0,5,1], [1,3,1], [1.5,4,1],
                         [2,-3,-1], [-2,-3,-1], [2,1,-1], [3.1,3,-1], [3,5,-1], [1,6,-1], [-2,5,-1], [-3,3,-1]])
        self.data_size = len(data)
        return data