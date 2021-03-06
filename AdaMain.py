# -*- coding: utf-8 -*-
"""
Created on Tue Jun 02 21:18:02 2015

@author: leftword
"""

import DataGen
import AdaBoost

if __name__ == '__main__':
    MyDataGen = DataGen.DataGenerator()
    #data = MyDataGen.get_default_data_1(plot=True)
    #data = MyDataGen.separable_2d(227,10,plot=None)
    data = MyDataGen.separable_2d_circle(13233,600,[7.0,3.0,2],plot=True)  
    positive = data[data[:,-1]==1]
    negative = data[data[:,-1]!=1]
    Boost = AdaBoost.AdaBoost()
    g_s, a_s, error, cur_round = Boost.get_adaboost_model(data, desired_error=0.02, plot_result = True, plot_g = True)
    #Boost.ADA_boost_show_demo_plot(data,g_s,a_s)
