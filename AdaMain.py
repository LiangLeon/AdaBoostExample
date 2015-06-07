# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 21:18:02 2015

@author: leftword
"""

import DataGen
import AdaBoost

if __name__ == '__main__':
    MyDataGen = DataGen.DataGenerator()
    data = MyDataGen.get_default_data_3()
    positive = data[data[:,-1]==1]
    negative = data[data[:,-1]!=1]
    Boost = AdaBoost.AdaBoost()
    g_s, a_s = Boost.get_adaboost_model(data, desired_error=0.01, plot_result = True)
    
