# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 21:18:02 2015

@author: leftword
"""
import numpy as np

import DataGen
import DecisionStumps

if __name__ == '__main__':
    MyDataGen = DataGen.DataGenerator()
    data = MyDataGen.get_default_data()
    weights = np.linspace(1, 1, data.shape[0])
    weights /= data.shape[0]
    MyDS = DecisionStumps.DecisionStumps(data.shape[0], MyDataGen.data_dim, data, weights) 
    print MyDS.get_hypothesis()