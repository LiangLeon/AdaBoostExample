# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 21:21:33 2015

@author: leftword
"""

import numpy as np

class DataGenerator:
    data_dim = 2
    def __init__(self, data_size=None, dim=None):
        if dim != None:
            self.data_dim = dim
        pass
        
    def get_default_data(self):
        self.data_dim = 2
        data = np.array([[0,0,1], [2,2,1],[3,0,-1],[4,0,-1]])
        return data
            