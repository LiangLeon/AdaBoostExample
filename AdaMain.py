# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 21:18:02 2015

@author: leftword
"""

import DataGen
import AdaBoost

if __name__ == '__main__':
    MyDataGen = DataGen.DataGenerator()
    data = MyDataGen.separable_2d(3213,20,plot=True)
    #data = MyDataGen.separable_2d_circle(2332,100,[7.0,3.0,2],plot=True)  
    positive = data[data[:,-1]==1]
    negative = data[data[:,-1]!=1]
    Boost = AdaBoost.AdaBoost()
    g_s, a_s, error, cur_round = Boost.get_adaboost_model(data, desired_error=0.001, plot_result = True, plot_g = True)
    
